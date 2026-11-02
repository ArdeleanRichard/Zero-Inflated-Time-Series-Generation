import json

import numpy as np
import matplotlib.pyplot as plt

def plot_real_vs_generated(
    real,
    generated,
    n_display=None,
    titles=None,
    figsize_per_row=(8, 2.2),
    suptitle=None,
    savepath=None,
    show=True,
    title_fontsize=10,
    label_fontsize=9,
    xtick_step=30,
    overlay=False,
):
    """
    Improved plotting of real vs generated time series with spacing/title fixes.

    Parameters
    ----------
    real, generated : array-like, shape (n_series, timesteps) or (n_series, timesteps, 1)
    n_display : int or None -- how many series to show (from index 0). None -> all.
    titles : list of str or None -- optional per-series title text
    figsize_per_row : tuple(width, height) -- width used for single column; final width = 2*width
    suptitle : str or None
    savepath : str or None -- save figure if provided
    show : bool -- call plt.show()
    title_fontsize, label_fontsize : int
    xtick_step : int or None -- if int, show xticks every xtick_step steps on bottom row only
    overlay : bool -- if True, plot real and generated in same subplot (one column)
    """
    real = np.asarray(real)
    gen = np.asarray(generated)

    # normalize shapes of (n, t, 1) -> (n, t)
    if real.ndim == 3 and real.shape[-1] == 1:
        real = real.reshape(real.shape[0], real.shape[1])
    if gen.ndim == 3 and gen.shape[-1] == 1:
        gen = gen.reshape(gen.shape[0], gen.shape[1])

    if real.shape != gen.shape:
        raise ValueError(f"Shape mismatch: real {real.shape} vs generated {gen.shape}")
    if real.ndim != 2:
        raise ValueError("Expect 2D arrays of shape (n_series, timesteps)")

    n_series, timesteps = real.shape
    if n_display is None:
        n_display = n_series
    n_display = int(min(n_display, n_series))
    if n_display <= 0:
        raise ValueError("n_display must be > 0")

    # layout: rows = n_display, cols = 1 (overlay) or 2 (side-by-side)
    rows = n_display
    cols = 1 if overlay else 2
    figsize = (cols * figsize_per_row[0], rows * figsize_per_row[1])

    # Use constrained_layout to avoid overlaps; will tweak top for suptitle.
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True
    )

    x = np.arange(timesteps)

    for i in range(n_display):
        if overlay:
            ax = axes[i, 0]
            ax.plot(x, real[i], label="Real", linewidth=1)
            ax.plot(x, gen[i], label="Generated", linewidth=1, alpha=0.8)
            ax.set_title(
                f"Series {i}" + (f" — {titles[i]}" if titles and i < len(titles) else ""),
                fontsize=title_fontsize
            )
            if i == rows - 1:  # only bottom row gets x label/ticks to avoid crowding
                if xtick_step:
                    ax.set_xticks(np.arange(0, timesteps, xtick_step))
                ax.set_xlabel("Timestep", fontsize=label_fontsize)
            else:
                ax.set_xticks([])

            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.label_outer()
            if i == 0:
                ax.legend(fontsize=max(title_fontsize-1, 8))
        else:
            # Left = real
            ax_real = axes[i, 0]
            ax_real.plot(x, real[i], linewidth=1)
            ax_real.set_title(
                f"Series {i} — Real" + (f" — {titles[i]}" if titles and i < len(titles) else ""),
                fontsize=title_fontsize
            )
            if i == rows - 1:
                if xtick_step:
                    ax_real.set_xticks(np.arange(0, timesteps, xtick_step))
                ax_real.set_xlabel("Timestep", fontsize=label_fontsize)
            else:
                ax_real.set_xticks([])

            ax_real.grid(True, linestyle=":", linewidth=0.5)
            ax_real.label_outer()

            # Right = generated
            ax_gen = axes[i, 1]
            ax_gen.plot(x, gen[i], linewidth=1)
            ax_gen.set_title(
                f"Series {i} — Generated" + (f" — {titles[i]}" if titles and i < len(titles) else ""),
                fontsize=title_fontsize
            )
            # hide left y tick labels of generated column to reduce clutter
            ax_gen.tick_params(labelleft=False)
            if i == rows - 1:
                if xtick_step:
                    ax_gen.set_xticks(np.arange(0, timesteps, xtick_step))
                ax_gen.set_xlabel("Timestep", fontsize=label_fontsize)
            else:
                ax_gen.set_xticks([])

            ax_gen.grid(True, linestyle=":", linewidth=0.5)
            ax_gen.label_outer()

    # Tidy overall layout: constrained_layout handles most spacing; ensure suptitle space
    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fontsize + 2)
        # When using constrained_layout, use subplots_adjust to allocate room for suptitle
        fig.subplots_adjust(top=0.93)

    # Save/show
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig






if __name__ == "__main__":
    n = 10
    timesteps = 365
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, timesteps)
    real = np.array([ (1 + 0.2 * rng.normal()) * np.sin(t + rng.uniform(0, 2*np.pi)) +
                      0.1 * rng.normal(size=timesteps) for _ in range(n) ])
    generated = real + 0.15 * rng.normal(size=(n, timesteps))

    plot_real_vs_generated(real, generated, n_display=10,
                           suptitle="Real vs Generated (10 series, 365 timesteps)",
                           xtick_step=30, overlay=False)
