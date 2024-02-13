import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import chart_utils

LR = 1e-2
FIGSIZE = (10, 3.5)
LINEWIDTH = 4
LEGEND_ALPHA = 0.9
YSCALE = "linear"
YMIN = -20
YMAX = 500
LEGEND_YPOS = 0.9

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Define the function and MSE after the reset
def special_scalar_function(x, w):
    # Avoid division by zero
    return (w**2 - 1) ** 2 * w * x + (w**2 - 4) * x**2 / (w**2 + 1e-9)


# Define the model class
class Model(torch.nn.Module):
    def __init__(self, init_h_value):
        super(Model, self).__init__()
        self.h = torch.nn.Parameter(torch.tensor([init_h_value], dtype=torch.float32))

    def forward(self, x):
        return special_scalar_function(x, self.h)


# Define the function to generate synthetic data
def generate_data(x, true_h):
    return Model(true_h).forward(
        torch.tensor(x, dtype=torch.float32)
    ).detach().numpy() + np.random.normal(scale=0.5, size=x.shape)


# Define the function to run gradient descent
def run_gradient_descent(model, x_input, y_data, epochs=500, learning_rate=LR):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    loss_history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_input)
        loss = loss_fn(y_pred, y_data)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return model.h.item(), loss_history


# Generate synthetic data
print("Generating synthetic data...")
x_data = np.linspace(-2, 2, 100)
np.random.shuffle(x_data)
x_tensor = torch.tensor(x_data, dtype=torch.float32)
pre_train_y = generate_data(x_data, 1)
fine_tune_y = generate_data(x_data, 2)
pre_train_y_tensor = torch.tensor(pre_train_y, dtype=torch.float32)
fine_tune_y_tensor = torch.tensor(fine_tune_y, dtype=torch.float32)

# Initialize and train models for pre-training
print("Pre-training models...")
pre_train_plus_model = Model(0.5)
pre_train_minus_model = Model(-0.5)
optimal_h_plus, loss_history_plus = run_gradient_descent(
    pre_train_plus_model, x_tensor, pre_train_y_tensor
)
optimal_h_minus, loss_history_minus = run_gradient_descent(
    pre_train_minus_model, x_tensor, pre_train_y_tensor
)

# Re-initialize and fine-tune models
print("Fine-tuning models...")
pre_train_plus_model = Model(optimal_h_plus)
pre_train_minus_model = Model(optimal_h_minus)
_, loss_history_plus_fine = run_gradient_descent(
    pre_train_plus_model, x_tensor, fine_tune_y_tensor
)
_, loss_history_minus_fine = run_gradient_descent(
    pre_train_minus_model, x_tensor, fine_tune_y_tensor
)

# Calculate Loss Gap Ratio:
# (area difference between the fine-tuning losses)
# / (area difference between horizontal line at y = L_ft_init and the base ft loss)
loss_gap_ratio = np.trapz(
    np.abs(np.array(loss_history_plus_fine) - np.array(loss_history_minus_fine))
) / np.trapz(np.abs(np.array(loss_history_plus_fine) - loss_history_plus_fine[0]))
print(f"Loss Gap Ratio: {loss_gap_ratio:.3f}")

# Plotting
print("Plotting...")
chart_utils.initialize_plot_no_markers()  # Remove markers

# Plot Pre-training loss curves
color_positive = chart_utils.get_color_from_palette(3)
color_negative = chart_utils.get_color_from_palette(2)

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

# Plot Pre-training loss curves
ax = axes[0]
sns.lineplot(
    loss_history_plus,
    label="Initial $h$ = 0.5",
    linewidth=LINEWIDTH * 0.75,
    color=color_positive,
    ax=ax,
)
sns.lineplot(
    loss_history_minus,
    label="Initial $h$ = -0.5",
    linewidth=LINEWIDTH,
    color=color_negative,
    linestyle="--",
    ax=ax,
)
ax.set_yscale(YSCALE)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel("Pre-Training Steps")
ax.set_ylabel("MSE Loss")
ax.set_title("Pre-Training Loss")
ax.legend(framealpha=LEGEND_ALPHA, loc="upper right", bbox_to_anchor=(1, LEGEND_YPOS))

# Plot Fine-tuning loss curves
ax = axes[1]
sns.lineplot(
    loss_history_plus_fine,
    label="From $h$ = 0.5",
    linewidth=LINEWIDTH,
    color=color_positive,
    ax=ax,
)
sns.lineplot(
    loss_history_minus_fine,
    label="From $h$ = -0.5",
    linewidth=LINEWIDTH,
    color=color_negative,
    linestyle="--",
    ax=ax,
)
# Shade in the difference between the curves
ax.fill_between(
    range(len(loss_history_plus_fine)),
    loss_history_plus_fine,
    loss_history_minus_fine,
    color="mediumpurple",
    alpha=0.3,
    hatch="//",
    label="Loss Gap",
)
ax.set_yscale(YSCALE)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel("Fine-Tuning Steps")
ax.set_title("Fine-Tuning Loss")
ax.legend(framealpha=LEGEND_ALPHA, loc="upper right", bbox_to_anchor=(1, LEGEND_YPOS))

# Plot the loss landscape
ax = axes[2]


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Generate noisy data for the latest function at w = 1 and w = 2
x_data = np.linspace(-2, 2, 100)
noise_scale = 0.5
selected_w_values = np.array([1, 2])  # w values of interest
data_latest_creative = {
    w: special_scalar_function(x_data, w)
    + np.random.normal(scale=noise_scale, size=x_data.shape)
    for w in selected_w_values
}

# Compute MSE loss for the latest function for w = 1 and w = 2
w_fine_range = np.linspace(-2.5, 2.5, 200)  # Fine range for plotting MSE
mse_latest_creative = {
    w: [
        mse(data_latest_creative[w], special_scalar_function(x_data, wt))
        for wt in w_fine_range
    ]
    for w in selected_w_values
}

# Plotting the MSE loss for the latest function for w = 1 and w = 2 with different colors
# Define distinct colors for w = 1 and w = 2
colors = [chart_utils.get_color_from_palette(0), chart_utils.get_color_from_palette(1)]

# Plot MSE for each w value with updated legend labels
for i, w in enumerate(selected_w_values):
    ax.plot(
        w_fine_range,
        mse_latest_creative[w],
        label=f"Pre-train $h$ = {w}" if w == 1 else f"Fine-tune $h$ = {w}",
        color=colors[i],
        linewidth=LINEWIDTH,
    )

ax.set_title("Loss Landscape")
ax.set_xlabel("Learned Model h")
ax.set_yscale(YSCALE)
ax.set_ylim(YMIN, 1000)
ax.legend(framealpha=LEGEND_ALPHA, loc="upper right", bbox_to_anchor=(1, LEGEND_YPOS))
# Save the combined plot
plt.suptitle(
    "Unadaptability in the Scalar Model $f(x, h) = (h^2 - 1)^2hx + (h^2 - 4)x^2 / h^2$",
    y=1.05,
)
chart_utils.save_plot("../charts/scalar_intuition", "scalar_intuition")
