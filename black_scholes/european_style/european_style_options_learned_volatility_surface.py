# pinn_european_learned_sigma.py
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

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
T_maturity = 30.0 / 365.0
use_real_market_data = False   # placeholder if you later plug market data

# training hyperparams (tune as needed)
n_collocation = 3000
n_terminal = 400
n_data = 600
n_epochs = 3000
lr = 1e-3

pde_weight = 1.0
term_weight = 100.0
data_weight = 1.0

# ---------------------
# Directories / checkpoints / figures
# ---------------------
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
ckpt_path = "models/bs_pinn_eur_with_sigma.pth"

# ---------------------
# Helpers: scaling
# ---------------------
def scale_S(S, K):
    return S / (K + 1e-12)

def scale_t(t, T):
    return t / (T + 1e-12)

# ---------------------
# Networks
# ---------------------
class PriceNet(nn.Module):
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

    def forward(self, S_scaled, t_scaled, sigma_scaled):
        x = torch.cat([S_scaled, t_scaled, sigma_scaled], dim=1)
        return self.net(x)

class SigmaNet(nn.Module):
    def __init__(self, input_dim=2, width=32, depth=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        # softplus ensures positive output
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, S_scaled, t_scaled):
        x = torch.cat([S_scaled, t_scaled], dim=1)
        raw = self.net(x)
        return self.softplus(raw) + 1e-6  # small floor to avoid zeros

# ---------------------
# Instantiate models + optimizer
# ---------------------
price_net = PriceNet(input_dim=3, width=64, depth=3).to(device)
sigma_net = SigmaNet(input_dim=2, width=32, depth=2).to(device)
optimizer = optim.Adam(list(price_net.parameters()) + list(sigma_net.parameters()), lr=lr)

# Attempt to load checkpoint
if os.path.exists(ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    price_net.load_state_dict(ckpt["price_state"])
    sigma_net.load_state_dict(ckpt["sigma_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
else:
    print("No checkpoint found; will train from scratch.")
    start_epoch = 1

# ---------------------
# Black-Scholes analytic (torch) for reference / mock data
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
# Sampling helpers
# ---------------------
def sample_collocation(n, S_min, S_max, T):
    S = torch.rand((n, 1), device=device) * (S_max - S_min) + S_min
    t = torch.rand((n, 1), device=device) * T
    return S, t

def make_mock_market_data(n, S_min, S_max, K, T, r, sigma):
    S = torch.rand((n, 1), device=device) * (S_max - S_min) + S_min
    t_now = torch.rand((n, 1), device=device) * T
    K_tensor = torch.full_like(S, K, device=device)
    sigma_t = torch.full_like(S, sigma, device=device)
    prices = bs_price_call_torch(S, K_tensor, t_now, T, r, sigma_t)
    return S, t_now, K_tensor, sigma_t, prices

# Prepare data (mock)
S_min, S_max = 0.01, 3.0 * K_val
S_coll, t_coll = sample_collocation(n_collocation, S_min, S_max, T_maturity)
S_terminal = torch.linspace(0.01, 3.0 * K_val, n_terminal, device=device).view(-1,1)
if not use_real_market_data:
    S_data, t_data, K_data, sigma_data, price_data = make_mock_market_data(n_data, 0.1, 2.0 * K_val, K_val, T_maturity, r_val, sigma_const := 0.2)
else:
    raise NotImplementedError("Plug market data here if desired.")

# ---------------------
# Loss helpers
# ---------------------
def pde_residual_price(price_net, sigma_net, S_phys, t_phys, K_ref, T_ref, r):
    """
    Compute PDE residual for batch of physical S and t.
    We scale S,t before feeding into nets (scale factors built-in).
    """
    # require grad wrt physical S and t
    S_req = S_phys.clone().detach().to(device).requires_grad_(True)
    t_req = t_phys.clone().detach().to(device).requires_grad_(True)

    # scaling
    S_scaled = scale_S(S_req, K_ref)
    t_scaled = scale_t(t_req, T_ref)

    # get sigma (unscaled physical sigma) predicted by sigma_net
    sigma_pred = sigma_net(S_scaled, t_scaled)  # (N,1) positive
    # also create sigma_scaled for price_net input normalization (divide by 0.5)
    sigma_scaled = sigma_pred / 0.5

    # price net forward (depends on scaled inputs)
    V = price_net(S_scaled, t_scaled, sigma_scaled)

    # first derivatives wrt physical variables (autograd will propagate through scaling)
    V_S = torch.autograd.grad(V, S_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_t = torch.autograd.grad(V, t_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    # second derivative
    V_SS = torch.autograd.grad(V_S, S_req, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]

    # PDE residual (use sigma_pred as the physical sigma)
    res = V_t + 0.5 * (sigma_pred ** 2) * (S_req ** 2) * V_SS + r * S_req * V_S - r * V
    return res, sigma_pred, V

def terminal_loss(price_net, S_phys, K_ref, T_ref):
    t_T_phys = torch.full_like(S_phys, T_ref, device=device)
    S_scaled = scale_S(S_phys, K_ref)
    t_scaled = scale_t(t_T_phys, T_ref)
    # for terminal evaluation we need a sigma input; use constant baseline or sigma_net
    # here we call sigma_net to be consistent with learned vol
    sigma_pred = sigma_net(S_scaled, t_scaled)
    V_pred_T = price_net(S_scaled, t_scaled, sigma_pred / 0.5)
    payoff = torch.maximum(S_phys - K_ref, torch.zeros_like(S_phys))
    return torch.mean((V_pred_T - payoff) ** 2)

def boundary_loss(price_net, K_ref, T_ref):
    S_min = torch.tensor([[1e-8]], device=device)
    S_max = torch.tensor([[5.0 * K_ref]], device=device)
    t_vals = torch.linspace(0.0, T_ref, 10, device=device)[:, None]
    S_min_rep = S_min.repeat(t_vals.shape[0],1)
    S_max_rep = S_max.repeat(t_vals.shape[0],1)

    S_min_scaled = scale_S(S_min_rep, K_ref)
    t_scaled = scale_t(t_vals, T_ref)
    sigma_min = sigma_net(S_min_scaled, t_scaled)
    V_low = price_net(S_min_scaled, t_scaled, sigma_min / 0.5)

    S_max_scaled = scale_S(S_max_rep, K_ref)
    sigma_max = sigma_net(S_max_scaled, t_scaled)
    V_high = price_net(S_max_scaled, t_scaled, sigma_max / 0.5)

    bc1 = torch.mean((V_low - 0.0) ** 2)  # approx 0 at S->0
    asym = (S_max_rep - K_ref * torch.exp(-r_val * (T_ref - t_vals)))
    bc2 = torch.mean((V_high - asym) ** 2)
    return bc1 + bc2

# ---------------------
# Training loop
# ---------------------
history = {'loss': [], 'pde': [], 'term': [], 'data': []}

for epoch in range(start_epoch, n_epochs + 1):
    price_net.train()
    sigma_net.train()
    optimizer.zero_grad()

    # sample PDE collocation batch
    idx = torch.randperm(S_coll.shape[0])[:1024]
    S_pde = S_coll[idx]
    t_pde = t_coll[idx]

    res, sigma_pred_batch, V_batch = pde_residual_price(price_net, sigma_net, S_pde, t_pde, K_val, T_maturity, r_val)
    L_pde = torch.mean(res**2)

    # terminal batch (random subset)
    idx_t = torch.randperm(S_terminal.shape[0])[:512]
    S_term_batch = S_terminal[idx_t]
    L_term = terminal_loss(price_net, S_term_batch, K_val, T_maturity)

    L_bc = boundary_loss(price_net, K_val, T_maturity)

    # data loss (mock)
    S_s = scale_S(S_data, K_data)
    t_s = scale_t(t_data, T_maturity)
    sigma_s = sigma_data / 0.5
    V_pred_data = price_net(S_s, t_s, sigma_s)
    L_data = torch.mean((V_pred_data - price_data) ** 2)

    total_loss = pde_weight * L_pde + term_weight * L_term + data_weight * L_data + L_bc

    total_loss.backward()
    optimizer.step()

    history['loss'].append(total_loss.item())
    history['pde'].append(L_pde.item())
    history['term'].append(L_term.item())
    history['data'].append(L_data.item())

    if epoch % 200 == 0 or epoch == start_epoch:
        print(f"Epoch {epoch:5d} | total {total_loss.item():.3e} | pde {L_pde.item():.3e} | term {L_term.item():.3e} | data {L_data.item():.3e}")

    # optional: periodic checkpoint
    if epoch % 1000 == 0:
        torch.save({
            "epoch": epoch,
            "price_state": price_net.state_dict(),
            "sigma_state": sigma_net.state_dict(),
            "optim_state": optimizer.state_dict()
        }, ckpt_path)
        print("Checkpoint saved at epoch", epoch)

# final save
torch.save({
    "epoch": n_epochs,
    "price_state": price_net.state_dict(),
    "sigma_state": sigma_net.state_dict(),
    "optim_state": optimizer.state_dict()
}, ckpt_path)
print("Final checkpoint saved:", ckpt_path)

# ---------------------
# Evaluation & visualize learned volatility surface
# ---------------------
price_net.eval()
sigma_net.eval()

S_plot = np.linspace(0.01, 2.0 * K_val, 120)
t_plot = np.linspace(0.0, T_maturity, 120)
S_grid, t_grid = np.meshgrid(S_plot, t_plot)
inp = np.hstack([S_grid.flatten()[:,None], t_grid.flatten()[:,None]])
with torch.no_grad():
    S_in = torch.tensor(scale_S(torch.tensor(inp[:,0:1], dtype=torch.float32, device=device), K_val), device=device)
    t_in = torch.tensor(scale_t(torch.tensor(inp[:,1:2], dtype=torch.float32, device=device), T_maturity), device=device)
    sigma_out = sigma_net(S_in, t_in).cpu().numpy().reshape(S_grid.shape)

# save volatility surface figure
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, t_grid, sigma_out, cmap='viridis', linewidth=0, antialiased=False)
ax.set_xlabel('S'); ax.set_ylabel('t'); ax.set_zlabel('σ(S,t)')
ax.set_title('Learned volatility surface (σ_net)')
fig.savefig('figures/learned_vol_surface.png', dpi=240, bbox_inches='tight')
print("Saved learned vol surface to figures/learned_vol_surface.png")
plt.close(fig)

# plot price at t=0 vs analytic BS (constant sigma baseline)
S_eval = np.linspace(0.01, 2.0 * K_val, 300)[:,None]
with torch.no_grad():
    S_eval_t = torch.tensor(S_eval, dtype=torch.float32, device=device)
    S_eval_s = scale_S(S_eval_t, K_val)
    t_zero = torch.zeros_like(S_eval_s)
    sigma_for_eval = torch.tensor((sigma_const := 0.2), dtype=torch.float32, device=device).repeat(S_eval_s.shape[0],1) / 0.5
    V_pin = price_net(S_eval_s, t_zero, sigma_for_eval).cpu().numpy().flatten()

# analytic
with torch.no_grad():
    S_torch = torch.tensor(S_eval, dtype=torch.float32)
    V_bs = bs_price_call_torch(S_torch, torch.tensor(K_val), torch.zeros_like(S_torch), T_maturity, r_val, torch.tensor(0.2)).cpu().numpy().flatten()

plt.figure(figsize=(8,5))
plt.plot(S_eval.flatten(), V_bs, label='Black-Scholes (σ=0.2)', linewidth=2)
plt.plot(S_eval.flatten(), V_pin, '--', label='PINN (learned σ)', linewidth=2)
plt.xlabel('S'); plt.ylabel('Call price V(S,0)')
plt.title('Price at t=0: PINN vs Black-Scholes (baseline)')
plt.legend(); plt.grid(True)
plt.savefig('figures/price_t0.png', dpi=240, bbox_inches='tight')
print("Saved price comparison to figures/price_t0.png")
plt.show()
