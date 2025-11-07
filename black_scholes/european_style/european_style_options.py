# pinn_black_scholes.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import os, math, random

# ---------------------
# Repro / device
# ---------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------
# Problem / training params
# ---------------------
K_val = 150.0
r_val = 0.01
sigma_const = 0.2
T_maturity = 30.0 / 365.0
use_real_market_data = False

n_collocation = 3000
n_terminal = 400
n_data = 600
n_epochs = 4000
lr = 1e-3

pde_weight = 1.0
term_weight = 100.0
data_weight = 1.0

# ---------------------
# Model persistence paths
# ---------------------
os.makedirs("models", exist_ok=True)
model_path = "models/bs_pinn_european.pth"

# ---------------------
# Scaling helpers
# ---------------------
def scale_inputs(S, t, K, T):
    return S / (K + 1e-12), t / (T + 1e-12)

# ---------------------
# PINN model
# ---------------------
class PINN(nn.Module):
    def __init__(self, input_dim=3, width=64, depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S, t, sigma):
        x = torch.cat([S, t, sigma], dim=1)
        return self.net(x)

# ---------------------
# Black-Scholes analytic
# ---------------------
def bs_price_call_torch(S, K, t, T, r, sigma):
    eps = 1e-8
    tau = (T - t).clamp(min=eps)
    sqrt_tau = torch.sqrt(tau)
    d1 = (torch.log(S / (K + 1e-16)) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau + 1e-16)
    d2 = d1 - sigma * sqrt_tau
    norm = Normal(0.0, 1.0)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    price = S * cdf_d1 - K * torch.exp(-r * tau) * cdf_d2
    intrinsic = torch.maximum(S - K, torch.zeros_like(S))
    return torch.where(tau < 1e-6, intrinsic, price)

# ---------------------
# PDE residual
# ---------------------
def pde_residual(model, S, t, sigma, r):
    S_req = S.clone().detach().to(device).requires_grad_(True)
    t_req = t.clone().detach().to(device).requires_grad_(True)
    sigma_in = sigma.clone().detach().to(device)
    V = model(S_req, t_req, sigma_in)
    V_S = torch.autograd.grad(V, S_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_t = torch.autograd.grad(V, t_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S_req, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]
    return V_t + 0.5 * (sigma_in ** 2) * (S_req ** 2) * V_SS + r * S_req * V_S - r * V

# ---------------------
# Terminal condition
# ---------------------
def terminal_loss(model, S_samples, K, r, sigma, T):
    t_T = torch.full_like(S_samples, T, device=device)
    V_pred_T = model(S_samples, t_T, sigma)
    payoff = torch.maximum(S_samples - K, torch.zeros_like(S_samples))
    return torch.mean((V_pred_T - payoff) ** 2)

# ---------------------
# Sampling helpers
# ---------------------
def sample_collocation(n, S_min, S_max, T):
    S = torch.rand((n, 1), device=device) * (S_max - S_min) + S_min
    t = torch.rand((n, 1), device=device) * (T - 0.0)
    return S, t

def make_mock_market_data(n, S_min, S_max, K, T, r, sigma):
    S = torch.rand((n, 1), device=device) * (S_max - S_min) + S_min
    t_now = torch.rand((n, 1), device=device) * (T - 0.0)
    K_tensor = torch.full_like(S, K, device=device)
    sigma_t = torch.full_like(S, sigma, device=device)
    prices = bs_price_call_torch(S, K_tensor, t_now, T, r, sigma_t)
    return S, t_now, K_tensor, sigma_t, prices

# ---------------------
# Build model
# ---------------------
model = PINN(input_dim=3, width=64, depth=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Try loading
if os.path.exists(model_path):
    print(f"Loading saved model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print("No saved model found — training new model.")

# ---------------------
# Data setup
# ---------------------
S_min, S_max = 0.01, 3.0 * K_val
S_coll, t_coll = sample_collocation(n_collocation, S_min, S_max, T_maturity)
K_coll = torch.full_like(S_coll, K_val, device=device)
r_coll = torch.full_like(S_coll, r_val, device=device)
sigma_coll = torch.full_like(S_coll, sigma_const, device=device)

S_terminal = torch.linspace(0.01, 3.0 * K_val, n_terminal, device=device).view(-1, 1)
K_term = torch.full_like(S_terminal, K_val, device=device)
sigma_term = torch.full_like(S_terminal, sigma_const, device=device)

if not use_real_market_data:
    S_data, t_data, K_data, sigma_data, price_data = make_mock_market_data(n_data, 0.1, 2.0 * K_val, K_val, T_maturity, r_val, sigma_const)
else:
    raise NotImplementedError("Implement market-data loading here.")

# ---------------------
# Training
# ---------------------
history = {'loss': [], 'pde': [], 'term': [], 'data': []}

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()

    def wrapped_model(S_phys, t_phys, sigma_phys):
        S_s = S_phys / (K_coll + 1e-12)
        t_s = t_phys / (T_maturity + 1e-12)
        sigma_s = sigma_phys / 0.5
        return model(S_s.to(device), t_s.to(device), sigma_s.to(device))

    residual = pde_residual(wrapped_model, S_coll, t_coll, sigma_coll, r_val)
    L_pde = torch.mean(residual ** 2)

    def wrapped_model_T(S_phys, t_phys, sigma_phys):
        S_s = S_phys / (K_term + 1e-12)
        t_s = t_phys / (T_maturity + 1e-12)
        sigma_s = sigma_phys / 0.5
        return model(S_s.to(device), t_s.to(device), sigma_s.to(device))

    L_term = terminal_loss(wrapped_model_T, S_terminal, K_term, r_val, sigma_term, T_maturity)

    S_s = (S_data / (K_data + 1e-12)).to(device)
    t_s = (t_data / (T_maturity + 1e-12)).to(device)
    sigma_s = (sigma_data / 0.5).to(device)
    V_pred_data = model(S_s, t_s, sigma_s)
    L_data = torch.mean((V_pred_data - price_data) ** 2)

    loss = pde_weight * L_pde + term_weight * L_term + data_weight * L_data
    loss.backward()
    optimizer.step()

    history['loss'].append(loss.item())
    history['pde'].append(L_pde.item())
    history['term'].append(L_term.item())
    history['data'].append(L_data.item())

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:5d} | total {loss.item():.3e} | pde {L_pde.item():.3e} | term {L_term.item():.3e} | data {L_data.item():.3e}")

# ---------------------
# Save model
# ---------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_path)
print(f"Model saved to {model_path}")

# ---------------------
# Evaluation & plots
# ---------------------
model.eval()
S_grid_phys = torch.linspace(0.01, 2.0 * K_val, 300, device=device).view(-1, 1)
t_zero_phys = torch.zeros_like(S_grid_phys, device=device)
K_grid = torch.full_like(S_grid_phys, K_val, device=device)
sigma_grid = torch.full_like(S_grid_phys, sigma_const, device=device)

S_grid_s = (S_grid_phys / (K_grid + 1e-12))
t_grid_s = (t_zero_phys / (T_maturity + 1e-12))
sigma_grid_s = (sigma_grid / 0.5)

with torch.no_grad():
    V_pred = model(S_grid_s, t_grid_s, sigma_grid_s).cpu().numpy().flatten()
    V_true = bs_price_call_torch(S_grid_phys.cpu(), K_grid.cpu(), t_zero_phys.cpu(), T_maturity, r_val, sigma_grid.cpu()).cpu().numpy().flatten()

plt.figure(figsize=(8,5))
plt.plot(S_grid_phys.cpu().numpy(), V_true, label='Black-Scholes (analytic)', linewidth=2)
plt.plot(S_grid_phys.cpu().numpy(), V_pred, '--', label='PINN prediction', linewidth=2)
plt.xlabel('Underlying price S')
plt.ylabel('Option price V(S, t=0)')
plt.title('PINN vs Black–Scholes (t=0)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.semilogy(history['loss'], label='total')
plt.semilogy(history['pde'], label='pde')
plt.semilogy(history['term'], label='terminal')
plt.semilogy(history['data'], label='data')
plt.legend(); plt.title('Training history (log scale)')
plt.grid(True)
plt.show()
