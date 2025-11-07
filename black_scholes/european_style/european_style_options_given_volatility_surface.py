# pinn_black_scholes_given_sigma.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.distributions.normal import Normal

# ---------------------
# Repro / device
# ---------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# Parameters
# ---------------------
K_val = 150.0
r_val = 0.01
T_maturity = 30.0 / 365.0
load_if_exists = True

n_collocation = 3000
n_terminal = 400
n_data = 600
n_epochs = 4000
lr = 1e-3

pde_weight = 1.0
term_weight = 100.0
data_weight = 1.0

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
ckpt_path = "models/bs_pinn_european_given_sigma.pth"

# ---------------------
# Volatility surface (given)
# Example: σ(S,t) = 0.2 + 0.1 * exp(-((S/K-1)^2)/(2*0.3^2)) * (1 - t/T)
# ---------------------
def sigma_surface(S, t, K=K_val, T=T_maturity):
    return 0.2 + 0.1 * torch.exp(-((S / K - 1.0) ** 2) / (2 * 0.3 ** 2)) * (1.0 - t / T)

# ---------------------
# Black–Scholes analytic price
# ---------------------
def bs_price_call(S, K, t, T, r, sigma):
    eps = 1e-8
    tau = torch.clamp(T - t, min=eps)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S / (K + 1e-16)) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau + 1e-16)
    d2 = d1 - sigma * sqrt_tau
    norm = Normal(0.0, 1.0)
    return S * norm.cdf(d1) - K * torch.exp(-r * tau) * norm.cdf(d2)

# ---------------------
# Neural network model for V(S,t)
# ---------------------
class PriceNet(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S_scaled, t_scaled):
        return self.net(torch.cat([S_scaled, t_scaled], dim=1))

price_net = PriceNet().to(device)
optimizer = optim.Adam(price_net.parameters(), lr=lr)

# ---------------------
# Scaling
# ---------------------
def scale_S(S, K): return S / (K + 1e-12)
def scale_t(t, T): return t / (T + 1e-12)

# ---------------------
# PDE residual
# ---------------------
def pde_residual_given_sigma(model, S, t, K, T, r):
    S_req = S.clone().detach().requires_grad_(True).to(device)
    t_req = t.clone().detach().requires_grad_(True).to(device)

    sigma = sigma_surface(S_req, t_req, K, T).to(device)
    S_s = scale_S(S_req, K)
    t_s = scale_t(t_req, T)
    V = model(S_s, t_s)

    V_S = torch.autograd.grad(V, S_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_t = torch.autograd.grad(V, t_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S_req, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]

    residual = V_t + 0.5 * sigma**2 * S_req**2 * V_SS + r * S_req * V_S - r * V
    return residual

# ---------------------
# Loss terms
# ---------------------
def terminal_loss(model, S, K, T):
    t_T = torch.full_like(S, T)
    V_pred = model(scale_S(S, K), scale_t(t_T, T))
    payoff = torch.maximum(S - K, torch.zeros_like(S))
    return torch.mean((V_pred - payoff) ** 2)

def boundary_loss(model, K, T):
    # Enforce V(0,t)=0 and dV/dS(large)=1 approximately
    S0 = torch.zeros((100,1), device=device)
    t_vals = torch.linspace(0, T, 100, device=device).view(-1,1)
    V0 = model(scale_S(S0,K), scale_t(t_vals,T))
    loss_0 = torch.mean(V0**2)

    S_inf = torch.full((100,1), 3*K, device=device)
    t_vals2 = torch.linspace(0,T,100,device=device).view(-1,1)
    V_inf = model(scale_S(S_inf,K), scale_t(t_vals2,T))
    loss_inf = torch.mean((V_inf - (S_inf - K * torch.exp(-r_val*(T-t_vals2))))**2)
    return loss_0 + loss_inf

# ---------------------
# Data points (synthetic)
# ---------------------
def make_mock_data(n, K, T, r):
    S = torch.rand((n,1), device=device)*3*K
    t = torch.rand((n,1), device=device)*T
    sigma = sigma_surface(S, t, K, T)
    prices = bs_price_call(S, K, t, T, r, sigma)
    return S, t, sigma, prices

S_data, t_data, sigma_data, price_data = make_mock_data(n_data, K_val, T_maturity, r_val)

# ---------------------
# Load or train
# ---------------------
start_epoch = 0
if load_if_exists and os.path.exists(ckpt_path):
    print(f"Found model at {ckpt_path}, loading...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    price_net.load_state_dict(checkpoint["model_state"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"✅ Loaded trained model ({start_epoch} epochs). Skipping training.")
else:
    print("Training new model...")

if start_epoch == 0:
    # Collocation & terminal samples
    S_coll = torch.rand((n_collocation,1), device=device)*3*K_val
    t_coll = torch.rand((n_collocation,1), device=device)*T_maturity
    S_term = torch.linspace(0.01,3*K_val,n_terminal,device=device).view(-1,1)

    for epoch in range(1, n_epochs+1):
        optimizer.zero_grad()

        res = pde_residual_given_sigma(price_net, S_coll, t_coll, K_val, T_maturity, r_val)
        L_pde = torch.mean(res**2)
        L_term = terminal_loss(price_net, S_term, K_val, T_maturity)
        L_bc = boundary_loss(price_net, K_val, T_maturity)

        S_s = scale_S(S_data, K_val)
        t_s = scale_t(t_data, T_maturity)
        V_pred_data = price_net(S_s, t_s)
        L_data = torch.mean((V_pred_data - price_data)**2)

        loss = pde_weight*L_pde + term_weight*L_term + data_weight*L_data + 0.1*L_bc
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | total {loss.item():.3e} | pde {L_pde.item():.3e} | term {L_term.item():.3e} | data {L_data.item():.3e}")

    torch.save({'model_state': price_net.state_dict(), 'epoch': n_epochs}, ckpt_path)
    print("✅ Saved model to", ckpt_path)

# ---------------------
# Evaluation & visualization
# ---------------------
price_net.eval()
S_plot = torch.linspace(0.01, 2*K_val, 300, device=device).view(-1,1)
t_zero = torch.zeros_like(S_plot)
sigma_vals = sigma_surface(S_plot, t_zero).cpu().numpy().flatten()

S_s = scale_S(S_plot, K_val)
t_s = scale_t(t_zero, T_maturity)
with torch.no_grad():
    V_pred = price_net(S_s, t_s).cpu().numpy().flatten()
    V_true = bs_price_call(S_plot.cpu(), torch.tensor(K_val), t_zero.cpu(), T_maturity, r_val, sigma_surface(S_plot,t_zero)).cpu().numpy().flatten()

plt.figure(figsize=(8,5))
plt.plot(S_plot.cpu().numpy(), V_true, label='Analytic BS (given σ)', linewidth=2)
plt.plot(S_plot.cpu().numpy(), V_pred, '--', label='PINN prediction', linewidth=2)
plt.xlabel('Underlying Price S')
plt.ylabel('Option Price V(S,0)')
plt.title('European Call: PINN vs Black–Scholes (given volatility surface)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("figures/pinn_given_sigma_price.png", dpi=200)
plt.show()

# ---------------------
# Plot volatility surface
# ---------------------
S_grid = torch.linspace(0.0, 2*K_val, 80)
t_grid = torch.linspace(0.0, T_maturity, 80)
S_mesh, t_mesh = torch.meshgrid(S_grid, t_grid, indexing="ij")
sigma_grid = sigma_surface(S_mesh, t_mesh).cpu().numpy()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_mesh.cpu().numpy(), t_mesh.cpu().numpy(), sigma_grid, cmap='viridis', alpha=0.9)
ax.set_xlabel('Underlying Price S')
ax.set_ylabel('Time t')
ax.set_zlabel('Volatility σ(S,t)')
ax.set_title('Given Volatility Surface σ(S,t)')
plt.tight_layout()
plt.savefig("figures/pinn_given_sigma_surface.png", dpi=200)
plt.show()
