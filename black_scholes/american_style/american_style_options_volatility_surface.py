import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Black–Scholes parameters
# ---------------------------
K = 50.0
r = 0.05
T = 1.0  # maturity (years)

# ---------------------------
# Example non-constant volatility surface σ(S, t)
# ---------------------------
def sigma_surface(S, t):
    """
    Simple illustrative volatility surface:
      - higher vol for deep ITM or OTM
      - mild term structure (vol decays as t → T)
    """
    base = 0.2
    skew = 0.1 * tf.exp(-((S / K - 1.0) ** 2) / 0.2)
    term = 0.05 * (1.0 - t / T)
    return base + skew + term

# ---------------------------
# Neural network V(S, t)
# ---------------------------
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer((2,)),  # S, t
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(1, None)
    ])

# ---------------------------
# Model persistence
# ---------------------------
model_path = "models/bs_pinn_volatility_surface.keras"
os.makedirs("models", exist_ok=True)

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
else:
    print("No saved model found — training new model.")
    model = create_model()

# ---------------------------
# PDE residual loss
# ---------------------------
def bs_pde_loss(model, S, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([S, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([S, t])
            V = model(tf.concat([S, t], axis=1))
        V_S = tape1.gradient(V, S)
        V_t = tape1.gradient(V, t)
    V_SS = tape2.gradient(V_S, S)
    del tape1, tape2

    sigma = sigma_surface(S, t)
    pde = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return tf.reduce_mean(tf.square(pde))

# ---------------------------
# Terminal condition: V(S, T) = max(S - K, 0)
# ---------------------------
def terminal_loss(model, S):
    t = tf.ones_like(S) * T
    V = model(tf.concat([S, t], axis=1))
    payoff = tf.maximum(S - K, 0)
    return tf.reduce_mean(tf.square(V - payoff))

# ---------------------------
# Boundary conditions
# ---------------------------
def boundary_loss(model):
    S_min = tf.constant([[1e-8]])
    S_max = tf.constant([[5*K]])
    t = tf.linspace(0.0, T, 10)[:, None]

    V_low = model(tf.concat([S_min * tf.ones_like(t), t], axis=1))
    V_high = model(tf.concat([S_max * tf.ones_like(t), t], axis=1))
    bc1 = tf.reduce_mean(tf.square(V_low))  # should be 0
    bc2 = tf.reduce_mean(tf.square(V_high - (S_max - K * tf.exp(-r * (T - t)))))  # asymptotic
    return bc1 + bc2

# ---------------------------
# Training loop
# ---------------------------
optimizer = tf.keras.optimizers.Adam(1e-3)
epochs = 3000

if not os.path.exists(model_path):
    for epoch in range(epochs):
        S = tf.random.uniform((256, 1), 0.0, 2*K)
        t = tf.random.uniform((256, 1), 0.0, T)
        with tf.GradientTape() as tape:
            L_pde = bs_pde_loss(model, S, t)
            L_terminal = terminal_loss(model, tf.random.uniform((256, 1), 0.0, 2*K))
            L_boundary = boundary_loss(model)
            loss = L_pde + L_terminal + L_boundary
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d}: loss={loss.numpy():.6e} | PDE={L_pde.numpy():.3e} | Term={L_terminal.numpy():.3e}")

    # Save after training
    model.save(model_path)
    print(f"Model saved to {model_path}")
else:
    print("Using loaded model for evaluation.")

# ---------------------------
# Compare against constant-vol Black–Scholes
# ---------------------------
def bs_price(S, K, T, t, r, sigma):
    tau = np.maximum(T - t, 1e-8)
    d1 = (np.log(S / K + 1e-16) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

S_vals = np.linspace(0.0, 2*K, 200)[:, None]
t_vals = np.zeros_like(S_vals)
V_pred = model(tf.concat([S_vals, t_vals], axis=1)).numpy().flatten()
V_true = bs_price(S_vals.flatten(), K, T, 0.0, r, 0.2)

plt.figure(figsize=(8,5))
plt.plot(S_vals, V_true, label='Constant-vol (analytical)')
plt.plot(S_vals, V_pred, '--', label='PINN (non-constant σ)')
plt.xlabel('Underlying Price S')
plt.ylabel('Option Value')
plt.legend()
plt.grid(True)
plt.title('American Option PINN — Non-Constant Volatility Surface')
plt.show()

# ---------------------------
# Visualize volatility surface σ(S, t)
# ---------------------------
S_plot = np.linspace(0.0, 2*K, 100)
t_plot = np.linspace(0.0, T, 100)
S_grid, t_grid = np.meshgrid(S_plot, t_plot)
sigma_vals = sigma_surface(tf.constant(S_grid, dtype=tf.float32),
                           tf.constant(t_grid, dtype=tf.float32)).numpy()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, t_grid, sigma_vals, cmap='viridis', alpha=0.9)
ax.set_xlabel('Underlying Price S')
ax.set_ylabel('Time t')
ax.set_zlabel('Volatility σ(S,t)')
ax.set_title('Example Non-Constant Volatility Surface')
# Save the figure to a file (e.g., PNG)
fig.savefig('figures/volatility_surface.png', dpi=300, bbox_inches='tight')
plt.show()
