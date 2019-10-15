import numpy as np

PREWITTX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
PREWITTY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

SOBELX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBELY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

