import numpy as np
import sys

sys.path.append("../")
from utils import plot_stroke

style = np.load(
    "./static/uploads/default_style.npy", allow_pickle=True, encoding="bytes"
).astype(np.float32)
# plot the sequence
plot_stroke(style, "default.png")
