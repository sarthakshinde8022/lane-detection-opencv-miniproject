"""Classical lane detection pipeline (OpenCV): grayscale -> blur -> Canny -> ROI -> Hough -> average/extrapolate."""
import cv2
import numpy as np

PARAMS = dict(blur_k=5, canny_lo=50, canny_hi=150, rho=2, theta=np.pi/180,
              hough_thresh=20, min_len=20, max_gap=100, min_abs_slope=0.45)

def region_of_interest(edges):
    h, w = edges.shape
    poly = np.array([[(int(0.08*w), h), (int(0.45*w), int(0.60*h)),
                      (int(0.55*w), int(0.60*h)), (int(0.95*w), h)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask), poly

def preprocess(img, p=PARAMS):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (p["blur_k"], p["blur_k"]), 0)
    edges = cv2.Canny(blur, p["canny_lo"], p["canny_hi"])
    return gray, blur, edges

def hough_lines(masked, p=PARAMS):
    return cv2.HoughLinesP(masked, p["rho"], p["theta"], p["hough_thresh"],
                           minLineLength=p["min_len"], maxLineGap=p["max_gap"])

def fit_lanes(lines, shape, p=PARAMS):
    """Split segments by slope sign, length-weighted average, return ((x1,y1,x2,y2) left, right) or None per side + (slope,intercept)."""
    h, w = shape[:2]
    L, R, Lw, Rw = [], [], [], []
    if lines is None or len(lines) == 0:   # newer OpenCV returns () when nothing is found
        return None, None, None, None
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < p["min_abs_slope"]:
            continue
        b = y1 - m * x1
        length = np.hypot(x2 - x1, y2 - y1)
        (L if m < 0 else R).append((m, b)); (Lw if m < 0 else Rw).append(length)
    def side(params, wts):
        if not params:
            return None, None
        m, b = np.average(np.array(params), axis=0, weights=wts)
        y1, y2 = h, int(0.62*h)
        return (int((y1-b)/m), y1, int((y2-b)/m), y2), (m, b)
    left, lp = side(L, Lw); right, rp = side(R, Rw)
    return left, right, lp, rp

def draw(img, left, right, poly=None):
    overlay = np.zeros_like(img)
    for ln in (left, right):
        if ln is not None:
            cv2.line(overlay, (ln[0], ln[1]), (ln[2], ln[3]), (0, 0, 255), 10)
    if left is not None and right is not None:   # lane area
        pts = np.array([[(left[0], left[1]), (left[2], left[3]), (right[2], right[3]), (right[0], right[1])]], np.int32)
        fill = np.zeros_like(img); cv2.fillPoly(fill, pts, (0, 255, 0))
        overlay = cv2.addWeighted(overlay, 1, fill, 0.35, 0)
    return cv2.addWeighted(img, 1.0, overlay, 1.0, 0)

def detect(img, p=PARAMS, smooth_state=None, alpha=0.2):
    gray, blur, edges = preprocess(img, p)
    masked, poly = region_of_interest(edges)
    lines = hough_lines(masked, p)
    left, right, lp, rp = fit_lanes(lines, img.shape, p)
    if smooth_state is not None:  # exponential smoothing over frames (video)
        for key, cur in (("L", lp), ("R", rp)):
            prev = smooth_state.get(key)
            if cur is not None:
                smooth_state[key] = cur if prev is None else tuple(alpha*np.array(cur)+(1-alpha)*np.array(prev))
        h = img.shape[0]
        def mk(par):
            if par is None: return None
            m, b = par; return (int((h-b)/m), h, int((0.62*h-b)/m), int(0.62*h))
        left, right = mk(smooth_state.get("L")), mk(smooth_state.get("R"))
    out = draw(img, left, right)
    info = dict(n_segments=0 if lines is None else len(lines), left=lp, right=rp,
                left_x=None if left is None else left[0], right_x=None if right is None else right[0],
                both=(lp is not None and rp is not None))
    return out, dict(gray=gray, blur=blur, edges=edges, masked=masked, poly=poly), info
