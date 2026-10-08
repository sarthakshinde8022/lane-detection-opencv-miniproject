"""Streamlit demo: Lane Detection (classical OpenCV pipeline).  Run: streamlit run app.py"""
import glob, os, shutil, subprocess, tempfile, time
import cv2, numpy as np, streamlit as st
import lane_detection as ld

st.set_page_config(page_title="Lane Detection", page_icon="🛣️", layout="wide")
st.title("🛣️ Lane Detection – Classical Computer Vision (OpenCV)")
st.caption("AI Mini Project demo · Grayscale → Blur → Canny → ROI → Hough → Lane fit → Overlay")

# ---------- sidebar: parameters ----------
sb = st.sidebar
sb.header("Pipeline parameters")
p = dict(ld.PARAMS)
p["blur_k"] = sb.select_slider("Gaussian blur kernel", [3, 5, 7, 9, 11], value=5)
p["canny_lo"], p["canny_hi"] = sb.slider("Canny thresholds (low, high)", 10, 300, (50, 150))
p["hough_thresh"] = sb.slider("Hough vote threshold", 5, 100, 20)
p["min_len"] = sb.slider("Min line length (px)", 5, 100, 20)
p["max_gap"] = sb.slider("Max line gap (px)", 10, 200, 100)
p["min_abs_slope"] = sb.slider("Min |slope| (drop flat lines)", 0.1, 1.0, 0.45, 0.05)
alpha = sb.slider("Video smoothing α (lower = smoother)", 0.05, 1.0, 0.2, 0.05)
sb.info("Defaults are the values used in the report.")

tab_img, tab_vid, tab_about = st.tabs(["🖼️ Image", "🎞️ Video", "ℹ️ About"])

def load_upload(f):
    arr = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ---------- image tab ----------
with tab_img:
    samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "samples", "*.jpg")))
    c1, c2 = st.columns([1, 1])
    up = c1.file_uploader("Upload a road image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img")
    choice = c2.selectbox("…or pick a sample", [os.path.basename(s) for s in samples] or ["(none)"])
    img = load_upload(up) if up else (cv2.imread(os.path.join(os.path.dirname(__file__), "samples", choice)) if samples else None)
    if img is None:
        st.warning("Upload an image to begin.")
    else:
        t = time.time(); out, st_imgs, info = ld.detect(img, p); ms = (time.time() - t) * 1000
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Left lane", "found" if info["left"] else "not found")
        m2.metric("Right lane", "found" if info["right"] else "not found")
        m3.metric("Hough segments", info["n_segments"])
        m4.metric("Time", f"{ms:.1f} ms")
        a, b = st.columns(2)
        a.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
        b.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Detected lanes", width="stretch")
        with st.expander("Show pipeline stages", expanded=True):
            roi = img.copy(); cv2.polylines(roi, st_imgs["poly"], True, (0, 255, 255), 3)
            s1, s2, s3, s4 = st.columns(4)
            s1.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="1. ROI", width="stretch")
            s2.image(st_imgs["blur"], caption="2. Gray + blur", width="stretch")
            s3.image(st_imgs["edges"], caption="3. Canny edges", width="stretch")
            s4.image(st_imgs["masked"], caption="4. ROI-masked edges", width="stretch")
        ok, buf = cv2.imencode(".png", out)
        st.download_button("⬇️ Download result", buf.tobytes(), "lane_result.png", "image/png")
        st.caption("⚠️ 'Found' means a line was produced, not that it is correct – always check the overlay visually.")

# ---------- video tab ----------
with tab_vid:
    vup = st.file_uploader("Upload a road video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"], key="vid")
    use_sample = st.checkbox("Use sample video (solidWhiteRight.mp4)", value=vup is None)
    smooth = st.toggle("Temporal smoothing", value=True)
    max_frames = st.number_input("Max frames to process", 30, 2000, 300, 30)
    if st.button("▶️ Process video", type="primary"):
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "in.mp4")
        if vup is not None:
            open(src, "wb").write(vup.read())
        elif use_sample:
            src = os.path.join(os.path.dirname(__file__), "samples", "solidWhiteRight.mp4")
        if not os.path.exists(src):
            st.error("No video provided.")
        else:
            cap = cv2.VideoCapture(src); W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or 25
            total = min(int(cap.get(7)) or max_frames, int(max_frames))
            raw_out = os.path.join(tmp, "raw.mp4")
            wr = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            state = {} if smooth else None; n = both = 0; xs = []; prog = st.progress(0.0); t0 = time.time(); keep = {}
            while n < total:
                ok, fr = cap.read()
                if not ok: break
                o, _, inf = ld.detect(fr, p, smooth_state=state, alpha=alpha)
                wr.write(o); n += 1; both += inf["both"]; xs.append(inf["left_x"] if inf["left_x"] is not None else np.nan)
                if n % max(1, total // 6) == 0: keep[n] = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
                if n % 5 == 0: prog.progress(min(n / total, 1.0))
            wr.release(); prog.progress(1.0); el = time.time() - t0
            m1, m2, m3 = st.columns(3)
            m1.metric("Frames", n); m2.metric("Both lanes found", f"{100*both/max(n,1):.1f}%"); m3.metric("Speed", f"{n/el:.1f} FPS")
            jit = np.nanmean(np.abs(np.diff(xs))) if n > 1 else 0
            st.metric("Jitter (px/frame, lower = steadier)", f"{jit:.2f}")
            final = raw_out
            if shutil.which("ffmpeg"):  # re-encode to H.264 so browsers can play it
                h264 = os.path.join(tmp, "out_h264.mp4")
                r = subprocess.run(["ffmpeg", "-y", "-i", raw_out, "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264], capture_output=True)
                if r.returncode == 0: final = h264; st.video(final)
            else:
                st.info("ffmpeg not found, so in-browser playback is unavailable. Download the video or view the sampled frames below.")
            if keep:
                cols = st.columns(len(keep))
                for c, (k, f) in zip(cols, keep.items()): c.image(f, caption=f"frame {k}", width="stretch")
            st.download_button("⬇️ Download processed video", open(final, "rb").read(), "lane_output.mp4", "video/mp4")

# ---------- about ----------
with tab_about:
    st.markdown("""
**Method.** Grayscale + Gaussian blur → Canny edges → trapezoid region of interest → probabilistic Hough transform →
split by slope sign, length-weighted average per side → overlay. For video, an exponential average of each lane line reduces flicker.

**Results (Udacity CarND sample set).** Both lanes found on 6/6 images. Smoothing cut frame-to-frame jitter about 4–5×
(e.g. 4.34 → 0.90 px/frame). ~80 FPS at 960×540 on CPU.

**Limitations.** No ground-truth labels in this set, so no accuracy figure. Known failure on the *challenge* video (left line
pulled onto a shadowed road edge). Straight-line model approximates curves; fixed ROI; not tested at night or in rain.

**Roadmap.** TuSimple labelled evaluation → curved lanes → shadow robustness → U-Net comparison.

*Data: Udacity CarND-LaneLines-P1. This is a classical CV pipeline, not a learned model.*
""")
