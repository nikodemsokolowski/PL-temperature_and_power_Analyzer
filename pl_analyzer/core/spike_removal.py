import numpy as np
from typing import List, Sequence


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        seg = ypad[i:i + window]
        med = np.nanmedian(seg)
        if np.isnan(med):
            # Fallback to nearest finite
            k = i + pad
            left = k
            right = k
            val = np.nan
            while left >= 0:
                if np.isfinite(ypad[left]):
                    val = ypad[left]
                    break
                left -= 1
            if not np.isfinite(val):
                while right < len(ypad):
                    if np.isfinite(ypad[right]):
                        val = ypad[right]
                        break
                    right += 1
            med = val if np.isfinite(val) else 0.0
        out[i] = med
    return out


def median_abs_dev(residual: np.ndarray) -> float:
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    return mad if mad > 0 else float(np.nanstd(residual))


def _segment_indices(mask: np.ndarray, max_width: int) -> List[int]:
    candidates: List[int] = []
    i = 1
    n = mask.size
    while i < n - 1:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            width = j - i + 1
            if width <= max_width:
                candidates.extend(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return candidates


def detect_spikes(
    energy: Sequence[float],
    counts: Sequence[float],
    window: int = 7,
    sigma: float = 6.0,
    max_width: int = 2,
    min_prominence: float = 0.0,
) -> List[int]:
    y = np.asarray(counts, dtype=float)
    if y.size < 3:
        return []
    base = rolling_median(y, max(3, window))
    resid = y - base
    scale = median_abs_dev(resid)
    if scale <= 0:
        s = float(np.nanstd(resid))
        scale = s if s > 0 else 1.0
    thresh = sigma * scale
    above = np.isfinite(resid) & (resid > max(thresh, min_prominence))
    return _segment_indices(above, max_width)


def detect_spikes_adaptive(
    energy: Sequence[float],
    counts: Sequence[float],
    base_window: int = 51,
    mad_window: int = 51,
    sigma: float = 6.0,
    max_width: int = 2,
    min_prominence: float = 0.0,
) -> List[int]:
    y = np.asarray(counts, dtype=float)
    if y.size < 3:
        return []
    base = rolling_median(y, max(3, base_window))
    resid = y - base
    res_med = rolling_median(resid, max(3, mad_window))
    abs_dev = np.abs(resid - res_med)
    loc_scale = rolling_median(abs_dev, max(3, mad_window))
    loc_scale = np.where(loc_scale <= 0, float(np.nanstd(resid)) if np.nanstd(resid) > 0 else 1.0, loc_scale)
    above = (np.isfinite(resid) & np.isfinite(loc_scale) & (resid > (sigma * loc_scale)) & (resid > min_prominence))
    return _segment_indices(above, max_width)


def detect_spikes_hybrid(
    energy: Sequence[float],
    counts: Sequence[float],
    base_window: int = 51,
    mad_window: int = 51,
    sigma: float = 6.0,
    max_width: int = 2,
    min_prominence: float = 0.0,
    broad_window: int = 151,
    avoid_width: int = 9,
    avoid_prominence: float = 0.0,
) -> List[int]:
    y = np.asarray(counts, dtype=float)
    if y.size < 3:
        return []
    base = rolling_median(y, max(3, base_window))
    resid = y - base
    res_med = rolling_median(resid, max(3, mad_window))
    abs_dev = np.abs(resid - res_med)
    loc_scale = rolling_median(abs_dev, max(3, mad_window))
    loc_scale = np.where(loc_scale <= 0, float(np.nanstd(resid)) if np.nanstd(resid) > 0 else 1.0, loc_scale)
    above = (np.isfinite(resid) & np.isfinite(loc_scale) & (resid > (sigma * loc_scale)) & (resid > min_prominence))
    # Broad peak avoidance
    broad = rolling_median(y, max(3, broad_window))
    resid_broad = y - broad
    avoid = resid_broad > avoid_prominence

    # Build candidates from narrow segments; drop if looks like broad peak top
    candidates: List[int] = []
    i = 1
    n = above.size
    while i < n - 1:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            width = j - i + 1
            if width <= max_width:
                region = slice(i, j + 1)
                if width >= avoid_width and np.any(avoid[region]):
                    pass
                else:
                    candidates.extend(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return candidates


def remove_spikes(
    energy: Sequence[float],
    counts: Sequence[float],
    spike_idx: Sequence[int],
    method: str = 'interp',
    neighbor_n: int = 1,
) -> np.ndarray:
    y = np.array(counts, dtype=float, copy=True)
    n = y.size
    if n == 0:
        return y
    mask = np.zeros(n, dtype=bool)
    for idx in spike_idx:
        if 0 <= idx < n:
            mask[idx] = True
    if not mask.any():
        return y

    if method == 'median':
        window = 5
        base = rolling_median(y, window)
        y[mask] = base[mask]
        return y

    if method == 'nan':
        y[mask] = np.nan
        return y

    if method in ('neighbor', 'window_mean'):
        nwin = max(1, int(neighbor_n))
        for idx in np.where(mask)[0]:
            start = max(0, idx - nwin)
            end = min(len(y), idx + nwin + 1)
            win = np.arange(start, end)
            win = win[~mask[win]]
            if win.size > 0:
                y[idx] = float(np.mean(y[win]))
            else:
                left = idx - 1
                while left >= 0 and mask[left]:
                    left -= 1
                right = idx + 1
                while right < len(y) and mask[right]:
                    right += 1
                vals = []
                if left >= 0:
                    vals.append(y[left])
                if right < len(y):
                    vals.append(y[right])
                if vals:
                    y[idx] = float(np.mean(vals))
        return y

    # Interpolate
    keep = ~mask
    if keep.sum() < 2:
        base = rolling_median(y, 5)
        y[mask] = base[mask]
        return y
    x = np.asarray(energy, dtype=float)
    y[mask] = np.interp(x[mask], x[keep], y[keep])
    return y

