import numpy as np
import matplotlib.pyplot as plt

def polynomial_features(X, degree):
    """
    Construct polynomial features for a univariate input X.
    Features are [x^1, x^2, ..., x^degree]. No constant feature is included;
    the intercept is fit separately (and not penalized) in the notebook.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1)
        Input feature values.
    degree : int
        Maximum power to include (D in the homework).

    Returns
    -------
    ndarray with shape (n_samples, degree) 
        Design matrix.
    """
    features = []
    for point in X:
        for j in point:
            nb = j
            break
        temp = []
        for i in range(degree):
            temp.append(nb**(i+1))
        features.append(temp)
    return features

def fourier_features(X, J, T):
    """
    Construct the Fourier feature map for a univariate input X.

    Features are:
        [cos(2π x / T), sin(2π x / T),
         cos(2π*2 x / T), sin(2π*2 x / T), ...,
         cos(2π*J x / T), sin(2π*J x / T)]
    No constant feature is included; the intercept is fit separately (and not
    penalized) in the notebook.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1)
        Input feature values.
    J : int
        Number of harmonics.
    T : float
        Period; 1/T is the fundamental frequency.

    Returns
    -------
    ndarray with shape (n_samples, 2*J)
        Fourier design matrix (feature dimension D = 2J).
    """
    # TODO: implement
    raise NotImplementedError


def mse(y_true, y_pred):
    """Mean squared error between two arrays of the same length."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def plot_data_and_fit(X, Y, pred_fn,
                      x_min, x_max, num_points = 400, title = None, 
                      save_path = None):
    grid = np.linspace(float(x_min), float(x_max), int(num_points)).reshape(-1, 1)
    pred = np.asarray(pred_fn(grid)).reshape(-1)

    plt.figure()
    
    plt.scatter(X.reshape(-1), Y, s=15, label="training data")
    plt.plot(grid.reshape(-1), pred, c = 'k', lw = 2, label="fit")

    plt.ylim(-1.5, 1.5)
    plt.xlabel("x")
    plt.ylabel("y")
    
    if title is not None:
        plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_extrapolation(X_train, Y_train, X_future, Y_future, pred_fns,
                       x_min = -20, x_max = 20, y_lim = (-2.5, 2.5), 
                       num_points= 800, title = None, save_path = None):
    """Plot training data, future data, and several fitted models on one axis."""
    grid = np.linspace(float(x_min), float(x_max), int(num_points)).reshape(-1, 1)

    plt.figure(figsize=(8, 4))
    
    plt.scatter(X_train.reshape(-1), Y_train, s=12, label="training data")
    plt.scatter(X_future.reshape(-1), Y_future, s=12, color="black", label="future data")
    colors = ["tab:red", "tab:purple", "tab:green", "tab:orange", "tab:brown"]
    
    for i, (name, fn) in enumerate(pred_fns.items()):
        plt.plot(grid.reshape(-1), np.asarray(fn(grid)).reshape(-1),
                 color=colors[i % len(colors)], lw = 2, label=name)
    
    plt.axvline(-10, color="gray", ls=":")
    plt.axvline(10, color="gray", ls=":")
    plt.ylim(*y_lim)
    plt.xlabel("x")
    plt.ylabel("y")
    
    if title is not None:
        plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
