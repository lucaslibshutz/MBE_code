import numpy as np
import matplotlib.pyplot as plt

def plot_covariance_matrix(P, state_labels=None, title="Covariance Matrix",
                           cmap="viridis", show_values=False):
    """
    Visualize a covariance matrix as a heatmap.

    Parameters
    ----------
    P : (n,n) array
        Covariance matrix.
    state_labels : list[str], optional
        Labels for each state variable.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap.
    show_values : bool
        If True, print numerical values inside each cell.
    """

    P = np.asarray(P)
    n = P.shape[0]

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(P, cmap=cmap)

    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Covariance")

    # ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    if state_labels is not None:
        ax.set_xticklabels(state_labels, rotation=45, ha="right")
        ax.set_yticklabels(state_labels)
    else:
        ax.set_xticklabels(np.arange(n))
        ax.set_yticklabels(np.arange(n))

    # optional cell values
    if show_values:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{P[i,j]:.2e}",
                        ha="center", va="center",
                        color="white" if abs(P[i,j]) > np.max(P)/2 else "black")

    ax.set_title(title)
    ax.set_xlabel("State")
    ax.set_ylabel("State")

    plt.tight_layout()
    plt.show()

