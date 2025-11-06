# PINN

## 🧠 Physics-Informed Neural Networks (PINNs) for American-Style Option Pricing

### Overview

A **Physics-Informed Neural Network (PINN)** is a neural architecture that embeds **known physical or financial laws** directly into its training process.  
Instead of learning purely from data, the PINN minimizes a composite **loss function** that enforces the governing **partial differential equation (PDE)** — in this case, the **Black–Scholes equation** — as a *soft constraint*.  

This approach allows the network to:
- Learn the structure of option prices under no-arbitrage conditions,
- Generalize better than purely data-driven models, and
- Provide continuous, differentiable pricing surfaces across strike, time, and volatility domains.

---

### ⚙️ The Black–Scholes PDE

For a European-style call option with price $V(S, t)$, spot price $S$, time $t$, constant risk-free rate $r$, and volatility $\sigma$, the Black–Scholes equation is:

```math
\frac{\partial V}{\partial t}
+ \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S}
- r V = 0
```

subject to the **terminal condition** at maturity $T$:

```math
V(S, T) = \max(S - K, 0)
```

for strike price $K$.

The PINN learns a function $\hat{V}_\theta(S, t, \sigma, K, r)$ whose derivatives (computed via automatic differentiation) are constrained to satisfy this PDE throughout the training domain.

---

### 🧩 Loss Function Design

The total training loss combines three key components:

```math
\mathcal{L} = 
\lambda_{\text{PDE}} \cdot \mathcal{L}_{\text{PDE}}
+ \lambda_{\text{terminal}} \cdot \mathcal{L}_{\text{terminal}}
+ \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}
```

where:

- **PDE residual loss** enforces the Black–Scholes differential constraint:

  ```math
  \mathcal{L}_{\text{PDE}} = \mathbb{E}\left[\left(
  V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS} + r S V_S - r V
  \right)^2\right]
  ```

- **Terminal loss** enforces the payoff condition $V(S, T) = \max(S-K, 0)$  
- **Data loss** (optional) fits observed market or synthetic prices

Training is performed using stochastic gradient descent (e.g., Adam), with all spatial and temporal derivatives computed via PyTorch’s `autograd`.

---

### 🇺🇸 American Options Extension

For **American-style options**, early exercise introduces a *free boundary condition*:

```math
V(S, t) \geq \max(S - K, 0)
```

and the PDE becomes an **inequality-constrained** problem (a *complementarity formulation*):

```math
\min\left(
\frac{\partial V}{\partial t}
+ \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S}
- r V,\;
V - (S - K)
\right) = 0
```

In the PINN framework, this condition is typically handled by:
- Adding a **penalty term** that enforces $V \geq \text{payoff}$,
- Or training two subnetworks (one for the continuation region, one for the exercise region),
- Or using a *ReLU barrier* to ensure nonnegative exercise premium.

Support for this is currently in development.

---

### 📊 Visualization and Validation

After training, we visualize the PINN's predictions against analytic **Black–Scholes solutions** (for European-style options) under a constant volatility assumption:

```python
plt.plot(S_grid, V_true, label="Black–Scholes (analytic)")
plt.plot(S_grid, V_pred, '--', label="PINN prediction")
plt.xlabel("Underlying Price S")
plt.ylabel("Option Price V(S, t)")
plt.legend()
plt.grid(True)
```

## European-Style Options (PINN) — Explanation of the Code

This section describes how the repository implements a Physics-Informed Neural Network (PINN) for **European-style** option pricing, and explains the key pieces of the reference implementation (model, losses, training loop, and evaluation).

### 1 — What the PINN learns

The model approximates the option pricing function
$$
V(S, t; K, r, \sigma)
$$
where:
- `S` is the underlying asset price (spot),
- `t` is the current time (measured in years),
- `K` is the option strike,
- `r` is the risk-free rate,
- `σ` is volatility (constant or per-sample).

We train a neural network `V̂_θ` that takes (scaled) inputs and outputs the option price.

---

### 2 — Governing PDE (enforced in the loss)

The Black–Scholes PDE (used for the PDE residual term) is rendered using the math fenced block below:

```math
\frac{\partial V}{\partial t}
+ \tfrac{1}{2}\sigma^{2} S^{2} \frac{\partial^{2} V}{\partial S^{2}}
+ r S \frac{\partial V}{\partial S}
- r V = 0
```

The PINN minimizes the mean-square of this PDE **residual** over randomly sampled collocation points `(S, t)` in the training domain.

---

### 3 — Loss components

The overall loss is a weighted sum of three terms:

- **PDE residual loss**  
  Mean squared PDE residual across collocation points:
  ```text
  L_pde = mean( (V_t + 0.5 * sigma^2 * S^2 * V_SS + r*S*V_S - r*V)^2 )
  ```

- **Terminal (payoff) loss**  
  Enforce the terminal condition at maturity `T`:
  ```math
  V(S, T) = \max(S - K, 0)
  ```
  Practically:
  ```python
  L_term = mean( (V(S, T) - max(S - K, 0))**2 )
  ```

- **Data loss (optional)**  
  Fit observed market prices (or synthetic BS prices) at sampled `(S, t)`:
  ```python
  L_data = mean( (V_pred - market_price)**2 )
  ```

Final loss:
```text
Loss = λ_pde * L_pde + λ_term * L_term + λ_data * L_data
```

---

### 4 — Key implementation details (what the code does)

Below are the important functions and design choices from the reference implementation.

#### Model
A fully connected feed-forward network that accepts *scaled* inputs `(S_scaled, t_scaled, sigma_scaled)` and outputs the scalar price:

```python
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
        x = torch.cat([S, t, sigma], dim=1)  # (N, 3)
        return self.net(x)                   # (N, 1)
```

**Note:** scaling (S/K and t/T) is applied before passing inputs to the network. Scaling stabilizes training.

#### PDE residual (autograd)
We compute derivatives using `torch.autograd.grad`. The residual function returns the PDE residual for a batch of `(S,t)`:

```python
def pde_residual(model, S, t, sigma, r):
    S_req = S.clone().detach().requires_grad_(True)
    t_req = t.clone().detach().requires_grad_(True)
    sigma_in = sigma.clone().detach()

    V = model(S_req_scaled, t_req_scaled, sigma_scaled)  # model expects scaled inputs

    V_S = torch.autograd.grad(V, S_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_t = torch.autograd.grad(V, t_req, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S_req, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]

    residual = V_t + 0.5 * (sigma_in ** 2) * (S_req ** 2) * V_SS + r * S_req * V_S - r * V
    return residual
```

In practice, the code wraps the model so that raw physical variables `S` and `t` are scaled internally before being passed to the neural net.

#### Terminal condition loss
Evaluate the network at `t = T` and penalize deviation from the intrinsic payoff:

```python
def terminal_loss(model, S_terminal, K, r, sigma, T):
    t_T = torch.full_like(S_terminal, T)
    V_pred_T = model(S_terminal_scaled, t_T_scaled, sigma_scaled)
    payoff = torch.maximum(S_terminal - K, torch.zeros_like(S_terminal))
    return torch.mean( (V_pred_T - payoff)**2 )
```

---

### 5 — Training loop (high-level skeleton)

The training loop alternates the computation of the three loss terms and backpropagates the weighted sum:

```python
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # collocation samples (S_coll, t_coll)
    residual = pde_residual(wrapped_model, S_coll, t_coll, sigma_coll, r)
    L_pde = torch.mean(residual**2)

    # terminal loss
    L_term = terminal_loss(wrapped_model_T, S_terminal, K_term, r, sigma_term, T_maturity)

    # optional data loss
    V_pred_data = model(S_data_scaled, t_data_scaled, sigma_data_scaled)
    L_data = torch.mean( (V_pred_data - price_data)**2 )

    loss = pde_weight * L_pde + term_weight * L_term + data_weight * L_data
    loss.backward()
    optimizer.step()
```

**Practical notes:**
- Use a relatively large `term_weight` (terminal loss) to ensure the network respects the payoff.
- Train first with `Adam`, optionally fine-tune with `LBFGS` for better convergence.
- Sample collocation points broadly across the `S` and `t` domain (e.g., `S ∈ [0, 3K]` and `t ∈ [0, T]`).

---

### 6 — Evaluation & visualization

After training:
- Evaluate `V̂` on a grid `S_grid × t_grid` and compare to analytic Black–Scholes results (when `σ` is constant) to validate correctness.
- Typical plots:
  - `V_PINN(S, t=0)` vs. analytic `V_BS(S, t=0)`.
  - Training history (total loss and components).
  - Residual heatmap `|PDE_residual(S,t)|`.

Example plotting code snippet:

```python
# Evaluate on a grid at t = 0
S_grid = torch.linspace(0.01, 2*K_val, 300).view(-1,1).to(device)
t_zero = torch.zeros_like(S_grid).to(device)
# scale inputs and run model...
V_pred = model(S_grid_scaled, t_zero_scaled, sigma_grid_scaled).cpu().numpy().flatten()

# analytic Black–Scholes
V_true = bs_price_call_torch(S_grid.cpu(), K_grid.cpu(), t_zero.cpu(), T_maturity, r_val, sigma_const).cpu().numpy().flatten()

plt.plot(S_grid.cpu().numpy(), V_true, label='Black–Scholes (analytic)')
plt.plot(S_grid.cpu().numpy(), V_pred, '--', label='PINN prediction')
plt.xlabel('Underlying price S')
plt.ylabel('Option price V(S,t=0)')
plt.legend(); plt.grid(True)
```

---

### 7 — How to plug in real market data

To use actual option-market data:
1. Build tensors of `(S, t, K, sigma, market_price)` where:
   - `S` = current spot (or historical spot when price snapshot was taken)
   - `t` = current time (or the time corresponding to the market price), or *time-to-maturity*
   - `sigma` = implied volatility (optional — can be used as input) or leave as constant
   - `market_price` = mid-price (or other observed price)
2. Replace the mock `S_data, t_data, K_data, sigma_data, price_data` used in the demo with your actual tensors (all shaped `(N,1)`).
3. Keep the PDE residual and terminal loss components — they act as regularizers that enforce no-arbitrage structure.

---

### 8 — Tips & troubleshooting

- **Scaling:** Always scale `S` and `t` before passing to the network (e.g. `S_scaled = S/K`, `t_scaled = t/T`). This makes training stable.
- **Small maturities:** For extremely small time-to-maturity, guard against numerical issues in the analytic price and in `tau = T - t` (clamp with `eps`).
- **Weights:** Tune `λ_pde`, `λ_term`, and `λ_data`. If the network reproduces market prices but violates the PDE, increase `λ_pde`.
- **Local volatility:** To learn a `σ(S,t)`, replace `sigma` as an input with a small subnetwork `sigma_net(S_scaled, t_scaled)` that outputs a positive volatility (e.g. via `softplus`). Train both subnets jointly with the PDE residual.
