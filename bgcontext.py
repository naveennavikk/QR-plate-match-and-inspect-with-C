"""Scale-, light- and stamp-robust comparison of the wall around a QR plate.

Both images are resampled into a shared "QR unit" frame (the QR symbol maps to
the same square in both), so the same physical wall area is compared no matter
how far away each photo was taken. Only pixels visible in BOTH photos are used.
The plate itself and the burned-in ID/GPS/date stamp are excluded, and colours
are white-balanced against the plate's white QR modules so different cameras
and lighting do not change the wall colour.
"""
from pathlib import Path

import cv2
import numpy as np

UNIT = 24                       # pixels per QR width in the shared frame
LEFT, TOP, RIGHT, BOTTOM = 2.5, 2.5, 4.0, 2.5   # context extent in QR units
# The plate extends beyond the QR symbol (mostly to the right: text + logos).
PLATE = (-0.35, -0.35, 2.0, 1.35)               # x1, y1, x2, y2 in QR units
STAMP_FRACTION = 0.20           # bottom part of the photo holding the ID/GPS/date stamp


def _frame(image, box):
    """Warp image into the shared QR-unit frame; return (rgb float, valid mask)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    qw, qh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    sx, sy = UNIT / qw, UNIT / qh
    out_w, out_h = int((LEFT + 1 + RIGHT) * UNIT), int((TOP + 1 + BOTTOM) * UNIT)
    matrix = np.float32([[sx, 0, -(x1 - LEFT * qw) * sx], [0, sy, -(y1 - TOP * qh) * sy]])
    src = image
    if sx < 0.5:  # anti-alias before shrinking a close-up
        sigma = 0.5 / sx
        src = cv2.GaussianBlur(image, (0, 0), sigma)
    warped = cv2.warpAffine(src.astype(np.float32), matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)

    valid_src = np.full(image.shape[:2], 255, np.uint8)
    valid_src[int(image.shape[0] * (1 - STAMP_FRACTION)):, :] = 0
    valid = cv2.warpAffine(valid_src, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0

    px1, py1, px2, py2 = PLATE
    valid[int((TOP + py1) * UNIT):int((TOP + py2) * UNIT), int((LEFT + px1) * UNIT):int((LEFT + px2) * UNIT)] = False
    white = _plate_white(image, box)
    return warped / white, valid


def _plate_white(image, box):
    """Colour of the plate's white QR modules, used as a white-balance reference."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    qr = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].reshape(-1, 3).astype(np.float32)
    if len(qr) < 4:
        return np.ones(3, np.float32) * 255
    lum = qr.sum(axis=1)
    bright = qr[lum >= np.percentile(lum, 85)]
    return np.maximum(bright.mean(axis=0), 1.0)


# ---------------------------------------------------------------------------
# Physical plate fingerprint: nails, damage, dirt, handwriting and the QR
# pattern itself. The shared printed design is removed by subtracting an
# average plate (models/plate_template.npy, see training/build_plate_template.py).
# ---------------------------------------------------------------------------
PLATE_UNIT = 48                               # pixels per QR width in the plate frame
PLATE_EXTENT = (0.45, 0.45, 1.1, 0.45)        # left, top, right, bottom margin in QR units
TEMPLATE_PATH = Path(__file__).resolve().parent / "models" / "plate_template.npy"
_template = None


def plate_frame_gray(image, box):
    """White-balanced luminance of the plate, warped into the plate frame."""
    x1, y1, x2, y2 = [float(v) for v in box]
    qw, qh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    left, top, right, bottom = PLATE_EXTENT
    sx, sy = PLATE_UNIT / qw, PLATE_UNIT / qh
    out_w = int((left + 1 + right) * PLATE_UNIT)
    out_h = int((top + 1 + bottom) * PLATE_UNIT)
    matrix = np.float32([[sx, 0, -(x1 - left * qw) * sx], [0, sy, -(y1 - top * qh) * sy]])
    src = image.astype(np.float32)
    if sx < 0.5:
        src = cv2.GaussianBlur(src, (0, 0), 0.5 / sx)
    warped = cv2.warpAffine(src, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return (warped / _plate_white(image, box)).mean(axis=2).astype(np.float32)


def _degrade(frame, pixels_per_qr):
    """Reduce detail to what a QR of `pixels_per_qr` pixels can show."""
    if pixels_per_qr >= PLATE_UNIT:
        return frame
    scale = max(pixels_per_qr, 4) / PLATE_UNIT
    h, w = frame.shape
    small = cv2.resize(frame, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _best_shift_correlation(a, b, max_shift):
    """Highest normalized correlation of a and b over small alignment shifts."""
    m = max_shift
    core = a[m:-m, m:-m]
    core = core - core.mean()
    best = -1.0
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            win = b[m + dy:b.shape[0] - m + dy, m + dx:b.shape[1] - m + dx]
            win = win - win.mean()
            denom = float(np.sqrt((core * core).sum() * (win * win).sum()))
            if denom > 1e-6:
                best = max(best, float((core * win).sum() / denom))
    return best


def plate_features(ref_rgb, ref_box, prop_rgb, prop_box):
    """Similarity of plate-specific marks (template removed); None without a template."""
    global _template
    if _template is None:
        if not TEMPLATE_PATH.is_file():
            return None
        _template = np.load(TEMPLATE_PATH)
    detail = min(ref_box[2] - ref_box[0], prop_box[2] - prop_box[0])
    ref = _degrade(plate_frame_gray(ref_rgb, ref_box), detail)
    prop = _degrade(plate_frame_gray(prop_rgb, prop_box), detail)
    tmpl = _degrade(_template, detail)
    shift = max(2, round(PLATE_UNIT / max(detail, 1) * 1.5))
    left, top = PLATE_EXTENT[0], PLATE_EXTENT[1]
    q = (slice(int(top * PLATE_UNIT), int((top + 1) * PLATE_UNIT)),
         slice(int(left * PLATE_UNIT), int((left + 1) * PLATE_UNIT)))
    ref_res, prop_res = ref - tmpl, prop - tmpl
    pad = shift
    qr_ref = ref_res[q[0].start - pad:q[0].stop + pad, q[1].start - pad:q[1].stop + pad]
    qr_prop = prop_res[q[0].start - pad:q[0].stop + pad, q[1].start - pad:q[1].stop + pad]
    return {
        # Does the property candidate look like a plate at all (printed design)?
        "plate_like": (_best_shift_correlation(tmpl, prop, shift) + 1) / 2,
        "plate_res": (_best_shift_correlation(ref_res, prop_res, shift) + 1) / 2,
        "qr_res": (_best_shift_correlation(qr_ref, qr_prop, shift) + 1) / 2,
        "detail": float(detail),
    }


def context_features(ref_rgb, ref_box, prop_rgb, prop_box):
    """Return similarity features in [0, 1] (higher = more alike)."""
    plate = plate_features(ref_rgb, ref_box, prop_rgb, prop_box) or {}
    wall = _wall_features(ref_rgb, ref_box, prop_rgb, prop_box)
    return {**wall, **plate}


def _wall_features(ref_rgb, ref_box, prop_rgb, prop_box):
    ref, ref_valid = _frame(ref_rgb, ref_box)
    prop, prop_valid = _frame(prop_rgb, prop_box)
    both = ref_valid & prop_valid
    coverage = float(both.mean())
    if both.sum() < UNIT * UNIT:          # less than one QR-area of shared wall
        return {"color": 0.0, "layout": 0.0, "hist": 0.0, "coverage": coverage}

    # White-balanced brightness and chromaticity.
    def channels(img):
        total = img.sum(axis=2) + 1e-6
        return total / 3.0, img[..., 0] / total, img[..., 1] / total

    rl, rr, rg = channels(ref)
    pl, pr, pg = channels(prop)

    # 1) Median wall colour relative to the plate white.
    def median_colour(l, r, g):
        return np.array([np.median(l[both]), np.median(r[both]) * 3, np.median(g[both]) * 3])

    diff = np.abs(median_colour(rl, rr, rg) - median_colour(pl, pr, pg))
    color = float(np.exp(-(diff[0] / 0.18 + (diff[1] + diff[2]) / 0.06)))

    # 2) Coarse layout: correlation of brightness on a half-QR-unit grid.
    cell = UNIT // 2
    h, w = both.shape
    gh, gw = h // cell, w // cell

    def grid(values):
        v = np.where(both, values, 0)[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).sum(axis=(1, 3))
        n = both[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).sum(axis=(1, 3))
        return v / np.maximum(n, 1), n >= cell * cell * 0.6

    rgv, rok = grid(rl)
    pgv, pok = grid(pl)
    ok = rok & pok
    layout = 0.5
    if ok.sum() >= 8:
        a, b = rgv[ok], pgv[ok]
        if a.std() > 0.02 and b.std() > 0.02:
            layout = float((np.corrcoef(a, b)[0, 1] + 1) / 2)
        else:  # both flat walls: layout agrees if both are flat
            layout = 0.75 if (a.std() <= 0.02 and b.std() <= 0.02) else 0.35

    # 3) Histogram of white-balanced colour over the shared wall area.
    def hist(l, r, g):
        sample = np.stack([np.clip(l[both], 0, 1.5), r[both], g[both]], axis=1).astype(np.float32)
        hist_, _ = np.histogramdd(sample, bins=(8, 8, 8), range=((0, 1.5), (0.2, 0.5), (0.2, 0.5)))
        hist_ = hist_.astype(np.float32).ravel()
        return hist_ / max(hist_.sum(), 1e-6)

    bc = float(np.sum(np.sqrt(hist(rl, rr, rg) * hist(pl, pr, pg))))
    return {"color": color, "layout": layout, "hist": bc, "coverage": coverage}
