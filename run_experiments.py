import cv2, glob, os, time, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lane_detection import detect, PARAMS

D = "data/udacity"; R = "results"
os.makedirs(R, exist_ok=True)

# ---- 1. Dataset analysis
rows = []
for f in sorted(glob.glob(f"{D}/test_images/*.jpg")):
    im = cv2.imread(f); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows.append(dict(name=os.path.basename(f), type="image", width=im.shape[1], height=im.shape[0],
                     frames=1, mean_brightness=round(float(g.mean()),1), contrast_std=round(float(g.std()),1)))
for f in sorted(glob.glob(f"{D}/test_videos/*.mp4")):
    c = cv2.VideoCapture(f); n = int(c.get(7)); ok, fr = c.read(); g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    rows.append(dict(name=os.path.basename(f), type="video", width=int(c.get(3)), height=int(c.get(4)),
                     frames=n, mean_brightness=round(float(g.mean()),1), contrast_std=round(float(g.std()),1)))
ds = pd.DataFrame(rows); ds.to_csv(f"{R}/dataset_summary.csv", index=False); print(ds.to_string())

# ---- 2. Images: pipeline stages + overlay
imgs = sorted(glob.glob(f"{D}/test_images/*.jpg")); img_rows = []
for f in imgs:
    im = cv2.imread(f); t = time.time(); out, st, info = detect(im); dt = time.time()-t
    cv2.imwrite(f"{R}/out_{os.path.basename(f)}", out)
    img_rows.append(dict(image=os.path.basename(f), segments=info["n_segments"], left=info["left"] is not None,
                         right=info["right"] is not None, both=info["both"], ms=round(dt*1000,1)))
pd.DataFrame(img_rows).to_csv(f"{R}/image_results.csv", index=False); print(pd.DataFrame(img_rows).to_string())

# stages figure for one image
im = cv2.imread(f"{D}/test_images/solidYellowCurve.jpg"); out, st, info = detect(im)
roi_vis = im.copy(); cv2.polylines(roi_vis, st["poly"], True, (0,255,255), 4)
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, (t, x, cm) in zip(ax.ravel(), [("1. Original + ROI", cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB), None),
        ("2. Grayscale + Gaussian blur", st["blur"], "gray"), ("3. Canny edges", st["edges"], "gray"),
        ("4. ROI-masked edges", st["masked"], "gray"), ("5. Hough segments averaged", None, None),
        ("6. Final overlay", cv2.cvtColor(out, cv2.COLOR_BGR2RGB), None)]):
    if x is None:
        from lane_detection import hough_lines, fit_lanes
        lines = hough_lines(st["masked"]); v = im.copy()
        for l in lines[:,0]: cv2.line(v, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2)
        x = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
    a.imshow(x, cmap=cm); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(f"{R}/pipeline_stages.png", dpi=90); plt.close()

# grid of all outputs
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
for a, f in zip(ax.ravel(), imgs):
    a.imshow(cv2.cvtColor(cv2.imread(f"{R}/out_{os.path.basename(f)}"), cv2.COLOR_BGR2RGB)); a.set_title(os.path.basename(f)); a.axis("off")
plt.tight_layout(); plt.savefig(f"{R}/all_image_outputs.png", dpi=80); plt.close()

# ---- 3. Videos: detection rate, jitter, FPS (raw vs smoothed)
vid_rows = []; series = {}
for f in sorted(glob.glob(f"{D}/test_videos/*.mp4")):
    name = os.path.basename(f)
    for mode in ("raw", "smoothed"):
        c = cv2.VideoCapture(f); W, H = int(c.get(3)), int(c.get(4))
        wr = cv2.VideoWriter(f"{R}/out_{mode}_{name}", cv2.VideoWriter_fourcc(*"mp4v"), c.get(5) or 25, (W, H))
        state = {} if mode == "smoothed" else None; n = both = 0; sl, sr = [], []; t0 = time.time(); saved = 0
        while True:
            ok, fr = c.read()
            if not ok: break
            out, _, info = detect(fr, smooth_state=state); wr.write(out); n += 1
            both += info["both"]
            sl.append(info["left_x"] if info["left_x"] is not None else np.nan); sr.append(info["right_x"] if info["right_x"] is not None else np.nan)
            if mode == "smoothed" and n in (30, 90) : cv2.imwrite(f"{R}/frame_{name[:-4]}_{n}.jpg", out)
        el = time.time()-t0; wr.release()
        sl, sr = np.array(sl), np.array(sr)
        jit = np.nanmean(np.abs(np.diff(np.r_[sl]))) ; jit_r = np.nanmean(np.abs(np.diff(sr)))
        vid_rows.append(dict(video=name, mode=mode, frames=n, both_lanes_pct=round(100*both/n,1),
                             jitter_px=round(float((jit+jit_r)/2),2), fps=round(n/el,1)))
        series[(name, mode)] = (sl, sr)
v = pd.DataFrame(vid_rows); v.to_csv(f"{R}/video_results.csv", index=False); print(v.to_string())

# raw lane-slope series plot (stability)
# use slope via x at the bottom edge isn't fixed; plot slope from stored? -> plot bottom-x proxy of left lane
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, name in zip(ax, ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]):
    for mode in ("raw", "smoothed"):
        a.plot(series[(name, mode)][0], label=mode)
    a.set_title(name); a.set_xlabel("frame"); a.set_ylabel("left lane x at image bottom (px)"); a.legend()
plt.tight_layout(); plt.savefig(f"{R}/stability_plot.png", dpi=90); plt.close()

# dataset plot
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].bar(ds["name"], ds["mean_brightness"]); ax[0].set_title("Mean brightness per sample"); ax[0].tick_params(axis="x", rotation=75)
ax[1].bar(ds["name"], ds["contrast_std"], color="tab:orange"); ax[1].set_title("Contrast (std of gray) per sample"); ax[1].tick_params(axis="x", rotation=75)
plt.tight_layout(); plt.savefig(f"{R}/dataset_analysis.png", dpi=90); plt.close()
