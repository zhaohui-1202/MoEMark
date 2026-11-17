import numpy as np

def components(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]

def resample(coords, count=32):
    a = np.asarray(coords, dtype=np.float64)[:, :2]
    a = a[np.r_[True, np.linalg.norm(np.diff(a, axis=0), axis=1) > 1e-10]]
    if len(a) == 1:
        return np.repeat(a, count, axis=0)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    closed = np.linalg.norm(a[0] - a[-1]) < 1e-9
    t = np.linspace(0, d[-1], count, endpoint=not closed)
    return np.stack([np.interp(t, d, a[:, k]) for k in range(2)], axis=1)
