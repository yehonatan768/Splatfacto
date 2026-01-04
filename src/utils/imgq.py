from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ImgMetrics:
    blur_lap_var: float
    sharp_tenengrad: float
    v_mean: float
    v_std: float


def metrics_bgr(img: np.ndarray) -> ImgMetrics:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur metric 1: Laplacian variance
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Blur metric 2: Tenengrad (Sobel gradient energy)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(gx * gx + gy * gy))

    # Exposure / brightness in HSV-V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    return ImgMetrics(
        blur_lap_var=blur,
        sharp_tenengrad=ten,
        v_mean=float(v.mean()),
        v_std=float(v.std()),
    )


def dct_phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    pHash-like DCT hash (returns bool array of shape [hash_size*hash_size]).
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    size = hash_size * highfreq_factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    small = np.float32(small)
    dct = cv2.dct(small)

    # Top-left low frequencies
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])  # skip DC term for median
    h = (dct_low > med).flatten()
    return h


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))
