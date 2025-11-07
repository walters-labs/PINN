import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------
# Constants
# ---------------------------
K = 50.0
r = 0.05
T = 1.0  # maturity (years)

# ---------------------------
# Volatility surface neural network σ(S,t)
# ---------------------------
def create_vol_surface_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer((2,)),  # inputs: (S, t)
        tf.keras.layers.Dense(32, 'tanh'),
        tf.keras.layers.Dense(32, 'tanh'),
        tf.keras.layers.Dense(1, 'softplus')  # softplus to ensure positivity
    ])

vol_surface_model = create_vol_surface_model()

# ---------------------------
# Option price neural network V(S,t)
# ---------------------------
def create_price_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer((2,)),  # inputs: (S, t)
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(64, 'tanh'),
        tf.keras.layers.Dense(1, None)
    ])

price_model = create_price_model()

# ---------------------------
# PDE residual loss with learned vol surface
# PDE: V_t + 0.5 * sigma(S,t)^2 * S^2 * V_SS + r * S * V_S - r * V = 0
# ---------------------------
def pde_loss(price_model, vol_surface_model, S, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([S, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([S, t])
            inputs = tf.concat([S, t], axis=1)
            V = price_model(inputs)
        V_S = tape1.gradient(V, S)
        V_t = tape1.gradient(V, t)
    V_SS = tape2.gradient(V_S, S)
    del tape1, tape2

    sigma = vol_surface_model(tf.concat([S, t], axis=1))
    pde_residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return tf.reduce_mean(tf.square(pde_residual))

# ---------------------------
# Terminal condition loss: V(S,T) = max(S - K, 0)
# ---------------------------
def terminal_loss(price_model, S):
    t_T = tf.ones_like(S) * T
    V_pred = price_model(tf.concat([S, t_T], axis=1))
    payoff = tf.maximum(S - K, 0)
    return tf.reduce_mean(tf.square(V_pred - payoff))

# ---------------------------
# Boundary condition loss
# ---------------------------
def boundary_loss(price_model):
    S_min = tf.constant([[1e-8]])
    S_max = tf.constant([[5 * K]])
    t_vals = tf.linspace(0.0, T, 10)[:, None]

    V_low = price_model(tf.concat([S_min * tf.ones_like(t_vals), t_vals], axis=1))
    V_high = price_model(tf.concat([S_max * tf.ones_like(t_vals), t_vals], axis=1))

    bc_low = tf.reduce_mean(tf.square(V_low))  # V(0, t) ~ 0 for call
    bc_high = tf.reduce_mean(tf.square(V_high - (S_max - K * tf.exp(-r * (T - t_vals)))))  # asymptotic
    return bc_low + bc_high

# ---------------------------
# Training setup
# ---------------------------
optimizer = tf.keras.optimizers.Adam(1e-3)
epochs = 4000
batch_size = 256

for epoch in range(epochs):
    # Sample collocation points for PDE and terminal losses
    S_pde = tf.random.uniform((batch_size, 1), 0.0, 2 * K)
    t_pde = tf.random.uniform((batch_size, 1), 0.0, T)
    S_term = tf.random.uniform((batch_size, 1), 0.0, 2 * K)

    with tf.GradientTape() as tape:
        L_pde = pde_loss(price_model, vol_surface_model, S_pde, t_pde)
        L_term = terminal_loss(price_model, S_term)
        L_bc = boundary_loss(price_model)

        total_loss = L_pde + 100.0 * L_term + L_bc

    grads = tape.gradient(total_loss, price_model.trainable_variables + vol_surface_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, price_model.trainable_variables + vol_surface_model.trainable_variables))

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: total_loss={total_loss.numpy():.6e} PDE={L_pde.numpy():.3e} Terminal={L_term.numpy():.3e} Boundary={L_bc.numpy():.3e}")

# Create the directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# ---------------------------
# Visualize learned volatility surface
# ---------------------------
S_plot = np.linspace(0.0, 2 * K, 100)
t_plot = np.linspace(0.0, T, 100)
S_grid, t_grid = np.meshgrid(S_plot, t_plot)
input_grid = tf.constant(np.hstack((S_grid.flatten()[:, None], t_grid.flatten()[:, None])), dtype=tf.float32)
sigma_vals = vol_surface_model(input_grid).numpy().reshape(S_grid.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, t_grid, sigma_vals, cmap='viridis', alpha=0.9)
ax.set_xlabel('Underlying Price S')
ax.set_ylabel('Time t')
ax.set_zlabel('Learned Volatility σ(S,t)')
ax.set_title('Learned Volatility Surface from Data-driven PINN')
plt.savefig('figures/learned_volatility_surface.png')
plt.show()

# ---------------------------
# Visualize option price at t=0
# ---------------------------
S_vals = np.linspace(0.0, 2 * K, 200)[:, None]
t_zeros = np.zeros_like(S_vals)
inputs = tf.constant(np.hstack((S_vals, t_zeros)), dtype=tf.float32)
V_pred = price_model(inputs).numpy().flatten()

plt.figure(figsize=(8, 5))
plt.plot(S_vals, V_pred, label='PINN Predicted Option Price')
plt.xlabel('Underlying Price S')
plt.ylabel('Option Price V(S,0)')
plt.title('Option Price Surface at t=0')
plt.grid(True)
plt.legend()
plt.savefig('figures/option_price_t0.png')
plt.show()
