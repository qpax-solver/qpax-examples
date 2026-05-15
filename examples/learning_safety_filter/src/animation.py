import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from PIL import Image


def animate_frames(fig, update_fn, n_frames, interval=80):
    """Create a matplotlib FuncAnimation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    update_fn : callable(frame_index) -> list of artists
    n_frames : int
    interval : int, milliseconds between frames

    Returns
    -------
    FuncAnimation
    """
    anim = FuncAnimation(fig, update_fn, frames=n_frames, interval=interval, blit=False)
    return anim


def save_gif(anim_or_frames, filename, fps=12, dpi=100):
    """Save an animation or list of PIL frames as a GIF.

    Parameters
    ----------
    anim_or_frames : FuncAnimation or list[PIL.Image]
        If FuncAnimation, renders to GIF via pillow writer.
        If list of PIL Images, saves directly.
    filename : str
        Output path.
    fps : int
    dpi : int
    """
    if isinstance(anim_or_frames, FuncAnimation):
        anim_or_frames.save(filename, writer="pillow", fps=fps, dpi=dpi)
    else:
        # List of PIL images
        frames = anim_or_frames
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def fig_to_pil(fig, dpi=100):
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()


def display_gif(path):
    """Display a GIF in a notebook using an HTML img tag with base64 data URI.

    This works in Jupyter, VS Code, and GitHub's notebook renderer (which
    does not support the image/gif MIME type from IPython.display.Image).
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    display(HTML(f'<img src="data:image/gif;base64,{b64}">'))
