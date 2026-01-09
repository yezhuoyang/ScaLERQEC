from __future__ import annotations

from typing import Sequence
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]



def r_squared(y_true: Sequence[float], y_pred: Sequence[float], clip: bool = False) -> float:
    """
    Compute the coefficient of determination (R²).

    Parameters
    ----------
    y_true : array-like
        Observed data.
    y_pred : array-like
        Model-predicted data (same length as y_true).
    clip : bool, default False
        If True, negative R² values are clipped to 0 so the
        score lies strictly in the interval [0, 1].
    Returns
    -------
    float
        The R² statistic.
    """
    yt: FloatArray = np.asarray(y_true, dtype=np.float64)
    yp: FloatArray = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    ss_res = np.sum((yt - yp) ** 2)        # residual sum of squares
    ss_tot = np.sum((yt - yt.mean()) ** 2) # total sum of squares

    # Handle the degenerate case where variance is zero
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, r2) if clip else r2

