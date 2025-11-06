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
