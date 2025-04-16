import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the convergence warnings from the Gaussian Process kernel optimizer
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def expected_improvement(x, model, f_best, xi=0.1):
    """
    Compute the Expected Improvement acquisition function.
    """
    x = np.atleast_2d(x)  # Ensure x is 2D
    mu, sigma = model.predict(x, return_std=True)
    sigma = sigma.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    with np.errstate(divide='warn'):
        Z = (f_best - mu - xi) / sigma
        ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def filter_rows_not_in_list(arr, indices_to_remove):
    """
    Extract rows from the array that are not in the given list of indices.
    """
    indices_to_remove_set = set(indices_to_remove)
    mask = [i not in indices_to_remove_set for i in range(arr.shape[0])]
    filtered_array = arr[mask]
    return filtered_array

# Read the CSV file 
df = pd.read_csv("RSMdataset.csv")

X = df[['B. Temp', 'Target MC', 'CD. Temp']].values
y = df['y'].values

# Gaussian Process (alpha calculated as the square of the standard deviation of center points)
kernel = C(1.0, (1e-3, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1, 200))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=34.3)

def objective(x):
    x = np.array(x).reshape(1, -1)
    return gp.predict(x, return_std=False)[0]

bounds = [(110, 140), (55, 65), (60, 80)]
x0 = [125, 60, 70]

def create_grid(var_fixed, idx_fixed, bounds, steps=100):
    idx_free = [i for i in range(3) if i != idx_fixed]
    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[idx_free[0]][0], bounds[idx_free[0]][1], steps),
        np.linspace(bounds[idx_free[1]][0], bounds[idx_free[1]][1], steps)
    )
    grid_fixed = np.full_like(grid_x, var_fixed)
    return grid_x, grid_y, idx_free, grid_fixed

def evaluate_grid(gp, grid_x, grid_y, idx_free, grid_fixed, idx_fixed):
    Z = []
    for x_val, y_val in zip(grid_x.ravel(), grid_y.ravel()):
        point = [0, 0, 0]
        point[idx_free[0]] = x_val
        point[idx_free[1]] = y_val
        point[idx_fixed] = grid_fixed.ravel()[0]
        Z.append(gp.predict([point], return_std=False)[0])
    return np.array(Z).reshape(grid_x.shape)

# Manually initialize with points from the BBD
initial_indices = [12]  # Replace with the indices of the chosen points
X_iter = X[initial_indices]  # Select the corresponding X values
y_iter = y[initial_indices]  # Select the corresponding y values

iter_list = initial_indices  # Track the indices of the selected points

num_iterations = len(df) - len(initial_indices)
param_names = ["B. Temp", "Target MC", "CD. Temp"]
fixed_values = [125, 60, 70]

min_values = []
final_results_table = []

# Fit the Gaussian Process with the initial points
gp.fit(X_iter, y_iter)

# Iteration 1
print("Iteration 1:")
optimal_fun_val = np.min(y_iter)  # Best observed y value
result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
global_min = result.fun
optimal_params = result.x
std_global_min = gp.predict(optimal_params[np.newaxis], return_std=True)[1]

min_values.append(global_min)
y_pred = gp.predict(X_iter, return_std=False)


print(f"  Global Minimum Value (y): {global_min:.2f}")
print(f"  Global Minimum Value Standard Deviation (y): {std_global_min}")
print("  Optimal Parameters:")
for name, value in zip(param_names, optimal_params):
    print(f"    {name}: {value:.2f}")
print()

observed_best_values = [optimal_fun_val]

# Start the optimization loop
for iteration in range(2, num_iterations + 1):
    n_dimensions = len(bounds)

    x_samples = X.copy()
    x_samples = filter_rows_not_in_list(x_samples, iter_list)

    ei_val = expected_improvement(x=x_samples, model=gp, f_best=optimal_fun_val)
    temp_next_id = np.argmax(ei_val).item()
    optimal_next_x = x_samples[temp_next_id]

    next_id = None
    for i in range(len(X)):
        if np.abs(optimal_next_x - X[i]).mean() < 0.001:
            next_id = i

    iter_list += [next_id]
    next_y = y[next_id]
    if next_y < optimal_fun_val:
        optimal_fun_val = next_y

    observed_best_values.append(optimal_fun_val)

    X_iter = X[iter_list]
    y_iter = y[iter_list]

    print(iter_list)
    print(f'Selected next point: {optimal_next_x}')

    gp.fit(X_iter, y_iter)

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    global_min = result.fun
    optimal_params = result.x
    std_global_min = gp.predict(optimal_params[np.newaxis], return_std=True)[1]

    min_values.append(global_min)
    y_all_mean, y_all_std = gp.predict(X, return_std=True)

    y_pred = gp.predict(X_iter, return_std=False)
    
    if iteration == num_iterations:
        for i in range(len(y_iter)):
            final_results_table.append({
                "B. Temp": X_iter[i, 0],
                "Target MC": X_iter[i, 1],
                "CD. Temp": X_iter[i, 2],
                "Actual y": y_iter[i],
                "Predicted y": y_pred[i],
            })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, ax in enumerate(axes):
        grid_x, grid_y, idx_free, grid_fixed = create_grid(fixed_values[j], j, bounds)
        Z = evaluate_grid(gp, grid_x, grid_y, idx_free, grid_fixed, j)
        cp = ax.contourf(grid_x, grid_y, Z, levels=20, cmap=cm.viridis)
        fig.colorbar(cp, ax=ax)
        ax.scatter(optimal_params[idx_free[0]], optimal_params[idx_free[1]],
                   color='red', label="Global Min", marker='x')
        ax.set_title(f"Fixed {param_names[j]}: {fixed_values[j]:.2f}")
        ax.set_xlabel(param_names[idx_free[0]])
        ax.set_ylabel(param_names[idx_free[1]])

    plt.suptitle(f"Iteration {iteration}: Contour Plots with Fixed Points (125, 60, 70)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    print()
    print(f"Iteration {iteration}:")
    print(f"  Global Minimum Value (y): {global_min:.2f}")
    print(f"  Global Minimum Value Standard Deviation (y): {std_global_min}")
    print("  Optimal Parameters:")
    for name, value in zip(param_names, optimal_params):
        print(f"    {name}: {value:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), min_values, marker='o', linestyle='-', color='b')
plt.title("Predicted Minimum Value vs. Iterations")
plt.xlabel("Iteration")
plt.ylabel("Global Minimum Value (y)")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(observed_best_values) + 1), observed_best_values, marker='o', linestyle='-', color='b')
plt.title("Best Observed Value vs. Iterations")
plt.xlabel("Iteration")
plt.ylabel("Best Observed Value (y)")
plt.grid()
plt.show()

final_results_df = pd.DataFrame(final_results_table)

# Use the fully trained model to make predictions for all data points
y_pred_all, y_std_all = gp.predict(X, return_std=True)

# Create a DataFrame that compares the measured response with the predicted response
df_comparison = df.copy()
df_comparison['Predicted y'] = y_pred_all
df_comparison['Prediction Std'] = y_std_all

print("\nComparison of Predicted and Measured Responses for All Data Points:\n")
print(df_comparison[['B. Temp', 'Target MC', 'CD. Temp', 'y', 'Predicted y', 'Prediction Std']])
