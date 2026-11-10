import streamlit as st
import cv2
import numpy as np

from src.fov import compute_fov_mask
from src.vessels import segment_retinal_vessels
from src.exudates import detect_hard_exudates
from src.visualization import build_diagnostic_overlay
from src.metrics import calculate_vessel_density, evaluate_segmentation

st.set_page_config(page_title="Retinal DIP Analyzer", layout="wide")

st.title("🩺 Retinal DIP Analysis & Feature Extraction")
# st.markdown("Computer Science & Engineering — Digital Image Processing Lab")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Algorithm Hyperparameters")
clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.5, 0.5)
gabor_sigma = st.sidebar.slider("Gabor Kernel Sigma (Scale)", 1.0, 4.0, 2.0, 0.5)
gabor_lambda = st.sidebar.slider("Gabor Wavelength (Period)", 3.0, 12.0, 6.0, 1.0)
top_hat_ksize = st.sidebar.slider("Top-Hat Structuring Element Size", 7, 25, 15, 2)
exudate_thresh = st.sidebar.slider("Exudate Intensity Cutoff", 20, 100, 45, 5)

st.sidebar.markdown("---")
st.sidebar.header("📁 File Uploads")
uploaded_file = st.sidebar.file_uploader("Upload Fundus Photo", type=["jpg", "png", "jpeg", "tif"])
uploaded_gt = st.sidebar.file_uploader("(Optional) Ground Truth Vessel Mask", type=["png", "jpg", "tif"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Execute modular pipeline
    with st.spinner("Processing through DIP stages..."):
        fov = compute_fov_mask(image_bgr)
        vessels, v_stages = segment_retinal_vessels(
            image_bgr, fov, 
            clahe_clip=clahe_clip, 
            gabor_sigma=gabor_sigma, 
            gabor_lambda=gabor_lambda
        )
        exudates, candidates, e_stages = detect_hard_exudates(
            image_bgr, fov, 
            top_hat_ksize=top_hat_ksize, 
            intensity_thresh=exudate_thresh
        )
        overlay = build_diagnostic_overlay(image_bgr, vessels, exudates)
        density = calculate_vessel_density(vessels, fov)

    # Top summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Retinal Area (FOV)", f"{cv2.countNonZero(fov):,} px")
    m2.metric("Vessel Density", f"{density:.2f}%")
    m3.metric("Hard Exudate Candidates", len(candidates))

    # Tabbed UI for evaluation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Segmentation Results", 
        "🔬 Intermediate DIP Transforms", 
        "📐 Connected Component Analysis", 
        "📊 Benchmark Accuracy (GT)"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input Color Fundus", use_container_width=True)
        c2.image(vessels, caption="Segmented Vessels (12-Dir Gabor)", use_container_width=True)
        c3.image(overlay, caption="Composite Overlay (Green: Vessels | Yellow: Exudates)", use_container_width=True)

    with tab2:
        st.subheader("Path A: Vessel Extraction Stages")
        cols_v = st.columns(len(v_stages))
        for idx, (title, s_img) in enumerate(v_stages.items()):
            cols_v[idx].image(s_img, caption=title, use_container_width=True)

        st.markdown("---")
        st.subheader("Path B: Exudate Extraction Stages")
        cols_e = st.columns(len(e_stages))
        for idx, (title, s_img) in enumerate(e_stages.items()):
            cols_e[idx].image(s_img, caption=title, use_container_width=True)

    with tab3:
        st.subheader("Candidate Lesion Geometry")
        if candidates:
            table_data = [
                {
                    "Candidate ID": c["id"],
                    "Pixel Area": c["area"],
                    "Centroid (X, Y)": f"({c['centroid_x']}, {c['centroid_y']})",
                    "Bounding Box (x, y, w, h)": str(c["bbox"])
                }
                for c in candidates[:20]
            ]
            st.table(table_data)
        else:
            st.info("No candidates detected with current intensity cutoff.")

    with tab4:
        st.subheader("Quantitative Evaluation against Ground Truth")
        if uploaded_gt is not None:
            gt_bytes = np.asarray(bytearray(uploaded_gt.read()), dtype=np.uint8)
            gt_mask = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
            if gt_mask.shape != vessels.shape:
                gt_mask = cv2.resize(gt_mask, (vessels.shape[1], vessels.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            metrics = evaluate_segmentation(vessels, gt_mask, fov)
            em1, em2, em3, em4 = st.columns(4)
            em1.metric("Dice Coefficient", f"{metrics['Dice / F1']:.4f}")
            em2.metric("IoU (Jaccard)", f"{metrics['IoU (Jaccard)']:.4f}")
            em3.metric("Sensitivity", f"{metrics['Sensitivity']:.4f}")
            em4.metric("Specificity", f"{metrics['Specificity']:.4f}")
        else:
            st.info("Upload an expert ground-truth mask to evaluate Dice and IoU metrics.")
else:
    st.info("👈 Upload a retinal fundus image from the sidebar to inspect results.")
