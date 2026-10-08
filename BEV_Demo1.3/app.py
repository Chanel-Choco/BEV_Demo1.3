"""
Streamlit proof-of-concept: upload an artwork (or pick a sample) -> P(AI-generated) from Model A, Model B
and their ensemble, plus Grad-CAM and SHAP explanations, the thesis results, and the limits.
Run with:   streamlit run app.py
Matches the final models (Version 15, LFAB v4).
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import results_data as RD
from inference import (BAND_LABELS, DEVICE, benchmark_latency, gradcam_overlay, lfab_band_shap,
                       lfab_gain_map, lfab_gains, load_models, pixel_shap, predict_prob, prepare_image)

st.set_page_config(page_title="AI Generated Art Detector (thesis demo)", page_icon="🎨", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(APP_DIR, "weights"))
SAMPLES_DIR = os.path.join(APP_DIR, "samples")
CAM_AI = "The AI generated output (red = evidence for AI)"
CAM_PRED = "The predicted class (as in the thesis notebooks)"


# ---------------------------------------------------------------------------------------------
# Cached work. Streamlit reruns this whole script on every click, so everything slow is cached by
# image content, which means moving a slider or switching a tab does not recompute anything.
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights...")
def get_models(weights_dir):
    return load_models(weights_dir)


@st.cache_resource(show_spinner="Measuring speed on this machine...")
def get_latency(_models, labels):
    return {label: benchmark_latency(m) for label, m in _models.items()}


@st.cache_data(show_spinner=False, max_entries=20)
def cached_prepare(image_bytes, force_jpeg, already_processed):
    return prepare_image(image_bytes, force_jpeg=force_jpeg, already_processed=already_processed)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_predict(_models, labels, tensor_np):
    return {label: predict_prob(m, tensor_np) for label, m in _models.items()}


@st.cache_data(show_spinner=False, max_entries=20)
def cached_cams(_models, labels, tensor_np, targets):
    return {label: gradcam_overlay(_models[label], tensor_np, t) for label, t in zip(labels, targets)}


@st.cache_data(show_spinner=False, max_entries=20)
def cached_band_shap(_models, labels, tensor_np):
    return {label: lfab_band_shap(_models[label], tensor_np) for label in labels}


@st.cache_data(show_spinner=False, max_entries=10)
def cached_pixel_shap(_model, model_label, crop_bytes, max_evals):
    """`_model` is not hashed; model_label + the image bytes + max_evals form the cache key."""
    crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(224, 224, 3)
    return pixel_shap(_model, crop, max_evals=max_evals)


def load_samples():
    """Images in samples/ show up as one-click examples. Optional samples/samples.json adds a title,
    the known label ("AI" or "Human"), a note, and preprocessed=true for files copied straight out of
    Notebook 1's resized_dataset folder (made by make_samples.py)."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    meta = {}
    meta_path = os.path.join(SAMPLES_DIR, "samples.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = {}
    out = []
    for fn in sorted(os.listdir(SAMPLES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            m = meta.get(fn, {})
            out.append({"file": fn, "path": os.path.join(SAMPLES_DIR, fn),
                        "title": m.get("title", os.path.splitext(fn)[0]), "label": m.get("label"),
                        "note": m.get("note", ""), "preprocessed": bool(m.get("preprocessed", False))})
    return out


# ---------------------------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------------------------
st.title("AI Generated Art Detector")
st.caption("Thesis Proof of Concept: Enhancing Lightweight CNN Models with Explainable AI for Detecting "
           "AI-Generated Art on Online Platforms")

with st.sidebar:
    st.header("Settings")
    st.subheader("Explanations")
    show_cam = st.checkbox("Grad-CAM heatmaps", value=True)
    show_band_shap = st.checkbox("SHAP: LFAB frequency bands", value=True,
                                 help="Exact Shapley values over LFAB's 6 frequency bands. Fast (64 forward "
                                      "passes per model).")
    show_pix_shap = st.checkbox("SHAP: pixel regions (slow)", value=False,
                                help="Partition explainer with a blur masker, as in Notebook 2. Takes seconds "
                                     "to a minute or more depending on the machine.")
    st.caption(f"Running on: **{DEVICE}**")
    with st.expander("Advanced"):
        threshold = st.slider("Decision threshold (P(AI) at or above this means 'AI generated')",
                              0.30, 0.90, 0.50, 0.01,
                              help="Every metric reported in the thesis uses 0.5. Other values are for "
                                   "exploring only and have no reported accuracy.")
        match_training = st.checkbox("Match training preprocessing (JPEG recompress every upload)", value=True,
                                     help="Notebook 1 saved every training image as a 256x256 JPEG. Leave on "
                                          "unless you are deliberately testing the effect of skipping it.")
        cam_mode = st.radio("Grad-CAM explains", [CAM_AI, CAM_PRED],
                            help="For a prediction of 'human', the first option shows which regions looked "
                                 "most AI like, and the second shows the evidence for 'human'.")
        shap_evals = st.slider("Pixel SHAP evaluations (more is finer but slower)", 100, 800, 250, 50,
                               disabled=not show_pix_shap)

models, missing = get_models(WEIGHTS_DIR)
if missing:
    st.error("Some checkpoints were not found:\n\n" + "\n".join(f"- {m}" for m in missing))
if not models:
    st.info("Copy `mobilenet_lfab_model.pth` (Notebook 2 output) and `efficientnet_lfab_model.pth` "
            "(Notebook 3 output) into the weights folder, then refresh.")
    st.stop()

labels = tuple(models)
latency = get_latency(models, labels)
samples = load_samples()

tab_det, tab_exp, tab_res, tab_about = st.tabs(["Detector", "Explain", "Thesis results", "About and limits"])

# state shared between the Detector and Explain tabs
ready = False
cams, band_shap = {}, {}
crop = None
probs = {}

# ---------------------------------------------------------------------------------------------
# Tab 1: detector
# ---------------------------------------------------------------------------------------------
with tab_det:
    up = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        up_id = getattr(up, "file_id", None) or f"{up.name}-{up.size}"
        if up_id != st.session_state.get("last_upload_id"):
            st.session_state["last_upload_id"] = up_id
            st.session_state["source"] = "upload"

    if samples:
        st.markdown("**Or try a sample from the thesis test set**")
        cols = st.columns(min(len(samples), 4))
        for i, s in enumerate(samples):
            with cols[i % len(cols)]:
                st.image(s["path"], width=140)
                if st.button(s["title"], key=f"sample_{s['file']}"):
                    st.session_state["source"] = "sample"
                    st.session_state["sample_file"] = s["file"]

    image_bytes, meta = None, None
    if st.session_state.get("source") == "upload" and up is not None:
        image_bytes = up.getvalue()
    elif st.session_state.get("source") == "sample":
        meta = next((s for s in samples if s["file"] == st.session_state.get("sample_file")), None)
        if meta is not None:
            with open(meta["path"], "rb") as fh:
                image_bytes = fh.read()

    if image_bytes is None:
        st.info("Upload an artwork or pick a sample to start.")
    else:
        already = bool(meta and meta["preprocessed"])
        with st.status("Analysing image...", expanded=True) as status:
            st.write("Preprocessing (RGB, 256x256, center crop 224, normalise), same as training")
            prep = cached_prepare(image_bytes, match_training, already)
            crop = prep["crop"]

            st.write("Running the models")
            probs = cached_predict(models, labels, prep["tensor"])
            ensemble = sum(probs.values()) / len(probs)

            if show_cam:
                st.write("Grad-CAM: where did each model look?")
                targets = tuple(1 if cam_mode == CAM_AI else int(probs[lb] >= 0.5) for lb in labels)
                cams = cached_cams(models, labels, prep["tensor"], targets)

            if show_band_shap:
                st.write("SHAP over LFAB's frequency bands (exact, 64 coalitions per model)")
                band_shap = cached_band_shap(models, labels, prep["tensor"])
            status.update(label="Done", state="complete", expanded=False)
        ready = True

        verdict_ai = ensemble >= threshold
        left, right = st.columns([1, 1.4])
        with left:
            st.image(crop, caption="What the models see (224x224 center crop)")
        with right:
            st.subheader("🤖 Likely AI generated" if verdict_ai else "🖌️ Likely human made")
            st.progress(min(max(ensemble, 0.0), 1.0),
                        text=f"Ensemble P(AI) = {ensemble:.1%}   (threshold {threshold:.0%})")
            for label, p in probs.items():
                st.write(f"**{label}**: P(AI) = {p:.1%}  ·  {latency[label]:.1f} ms per image")
            st.caption(f"Speed is measured on this machine ({DEVICE.type.upper()}, average of 30 runs, model "
                       "only). The thesis values of 6.53 ms and 9.16 ms were measured on a Tesla T4 GPU, see "
                       "the Thesis results tab. P(AI) is an uncalibrated model score, not a guarantee.")
            above = [p >= threshold for p in probs.values()]
            if len(above) > 1 and any(above) and not all(above):
                st.warning("The two models disagree on this image.")
            if abs(ensemble - threshold) < 0.10:
                st.warning("Close to the threshold, so treat this as uncertain. "
                           "(The 10 point margin is a demo rule of thumb, not a thesis result.)")
            if meta and meta.get("label"):
                known = "AI generated" if str(meta["label"]).lower().startswith("ai") else "human made"
                st.info(f"Known label in the test set: **{known}**. {meta.get('note', '')}")
            w, h = prep["orig_size"]
            if abs(w - h) > 0.05 * max(w, h):
                st.caption(f"Your image is {w}x{h}, not square. It is squashed to a square before analysis, "
                           "the same way the training images were, so fine details such as stroke "
                           "thickness can change.")

# ---------------------------------------------------------------------------------------------
# Tab 2: explanations
# ---------------------------------------------------------------------------------------------
with tab_exp:
    if not ready:
        st.info("Run an image on the Detector tab first.")
    else:
        if cams:
            st.subheader("Grad-CAM")
            st.caption("Heatmap of the regions that influenced each model's output. "
                       + ("Red marks regions pushing the score toward 'AI generated'."
                          if cam_mode == CAM_AI else
                          "It explains the class each model predicted."))
            cols = st.columns(len(cams))
            for col, (label, overlay) in zip(cols, cams.items()):
                col.image(overlay, caption=label)
            st.caption("Grad-CAM shows where the model looked, not why those regions led to the score. The map "
                       "is coarse because it comes from the last backbone block (7x7). Model B tends to look "
                       "along the borders, with a border to center ratio above 1.15 in the thesis "
                       "(Section 4.7.1), so check whether red sits on the edges before trusting it.")

        if band_shap:
            st.subheader("SHAP: how much does each LFAB frequency band move the score?")
            cols = st.columns(len(band_shap))
            for col, (label, (phi, p_all, p_none)) in zip(cols, band_shap.items()):
                fig, ax = plt.subplots(figsize=(4.6, 3.0))
                ax.barh(range(len(phi)), phi * 100, color=["#d62728" if v > 0 else "#1f77b4" for v in phi])
                ax.set_yticks(range(len(phi)))
                ax.set_yticklabels([b.split(" (")[0] for b in BAND_LABELS], fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="grey", lw=0.8)
                ax.set_xlabel("Shapley value (percentage points of P(AI))", fontsize=8)
                ax.set_title(label.split(" + ")[0], fontsize=9)
                fig.tight_layout()
                col.pyplot(fig)
                plt.close(fig)
                gains = lfab_gains(models[label])
                col.caption(f"Effect of setting all frequency gains to zero: {(p_all - p_none) * 100:+.2f} "
                            f"points (all bands kept vs. all dropped). Mean learned gain per band, low to "
                            f"high frequency (1.00 is neutral): {', '.join(f'{g:.2f}' for g in gains)}.")
            ref = RD.BAND_SHAP_REFERENCE
            st.caption("This shows how much the prediction depends on each band. It does not show that any "
                       "frequency proves an image is AI generated. Red pushes toward 'AI generated', blue "
                       "toward 'human'. Band 0 is the lowest radial frequency band and Band 5 the highest. "
                       "Bars near zero mean LFAB had little influence on this image, which fits the thesis "
                       "finding that LFAB adds little accuracy and learned no preference for specific "
                       f"frequencies. For scale, the thesis mean absolute SHAP per band at the final "
                       f"diagnosis was about {ref['Model A']:.4f} for Model A and {ref['Model B']:.4f} for "
                       f"Model B in probability units, which is about {ref['Model A'] * 100:.1f} to "
                       f"{ref['Model B'] * 100:.1f} points on this chart.")

            with st.expander("Learned LFAB gain maps (frequency filter, averaged over channels)"):
                cols = st.columns(len(band_shap))
                for col, label in zip(cols, band_shap):
                    gm = lfab_gain_map(models[label])
                    fig, ax = plt.subplots(figsize=(3.6, 3.2))
                    im = ax.imshow(gm, cmap="coolwarm", vmin=0.0, vmax=2.0)
                    ax.set_title(label.split(" + ")[0], fontsize=9)
                    ax.set_xlabel("horizontal frequency", fontsize=8)
                    ax.set_ylabel("vertical frequency", fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    col.pyplot(fig)
                    plt.close(fig)
                st.caption("Gain 1.0 (white) means the frequency is left unchanged. Below 1 damps it and above 1 "
                           "amplifies it (range 0 to 2). A map that stays near white means the filter barely "
                           "moved from neutral, as reported in the thesis (Figure 9).")

        if show_pix_shap:
            st.subheader("SHAP: which image regions moved the score?")
            shap_label = st.radio("Model to explain", list(models), horizontal=True)
            with st.spinner("Computing pixel SHAP... this can take a while on CPU"):
                t0 = time.perf_counter()
                sv = cached_pixel_shap(models[shap_label], shap_label, crop.tobytes(), int(shap_evals))
                elapsed = time.perf_counter() - t0
            lim = max(float(np.abs(sv).max()), 1e-9)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(crop)
            axes[0].axis("off")
            axes[0].set_title("Input", fontsize=9)
            axes[1].imshow(crop.mean(axis=-1), cmap="gray", alpha=0.6)
            im = axes[1].imshow(sv, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.7)
            axes[1].axis("off")
            axes[1].set_title("SHAP (red toward AI, blue toward human)", fontsize=9)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"{shap_label} · {int(shap_evals)} evaluations · {elapsed:.1f} s. SHAP values sum to the "
                       "change in P(AI) between the blurred baseline and this image. Fewer evaluations give "
                       "coarser, noisier maps.")

        if not (cams or band_shap or show_pix_shap):
            st.info("Turn on at least one explanation in the sidebar.")

# ---------------------------------------------------------------------------------------------
# Tab 3: thesis results (static, copied from the manuscript)
# ---------------------------------------------------------------------------------------------
with tab_res:
    st.caption("All numbers on this tab are copied from the final manuscript (Version 15). Nothing here is "
               "computed live. Each configuration was trained once, and a single accuracy figure on the "
               "1,500 image test set has a 95 percent interval of about plus or minus 1.5 points.")

    st.subheader("Main test set (Table 7)")
    st.table(RD.main_test_set())
    st.caption("The ensemble is the average of the two models' probabilities. It was evaluated on this test "
               "set only, not on the distortion or unseen generator tests.")

    st.subheader("Errors by data source (Table 8)")
    st.table(RD.source_errors())

    st.subheader("Effect of LFAB, with and without (Table 9)")
    st.table(RD.ablation())
    st.caption("LFAB gave small and mixed changes. Accuracy rose about 0.6 points for both models, recall "
               "fell, and the differences are within the uncertainty of the test set.")

    st.subheader("Controlled distortions (Table 11)")
    st.caption("Noise with standard deviation 20, JPEG quality 20, and a 2x downscale then upscale.")
    st.bar_chart(RD.distortion_accuracy_chart())
    st.table(RD.distortions())

    st.subheader("Unseen generators (Table 13)")
    st.table(RD.unseen_generators())
    st.caption("750 AI images from AI Pastiche and 750 from benjaminStreltzin, plus 750 human made images. "
               "Accuracy is left out because the set has twice as many AI images as human ones. With LFAB, "
               "Model A did worse than without it, and Model B was about the same.")

    st.subheader("Different image type, CIFAKE (Table 12)")
    st.table(RD.cifake())
    st.caption("CIFAKE is 32x32 photographs, a different kind of image from artwork. This is a cross domain "
               "test, not an unseen generator test, and both models were weak on it.")

    st.subheader("Computational efficiency (Table 14)")
    st.table(RD.efficiency())
    st.caption("Measured on a Tesla T4 GPU. The thesis did not measure CPU latency, and low resource "
               "conditions were simulated, so these numbers do not prove performance on phones or edge "
               "devices. LFAB itself adds about 14,000 parameters to Model A and 21,000 to Model B, under "
               "1.3 percent of either model.")

    st.subheader("What LFAB learned (Table 10)")
    st.table(RD.lfab_learned())

    st.subheader("Comparison with the published SCADET benchmark (Table 15)")
    st.table(RD.benchmark())
    st.caption("The SCADET paper used different datasets, splits and test setups, so this is a reference "
               "comparison and not proof that these models are better.")

# ---------------------------------------------------------------------------------------------
# Tab 4: about and limits
# ---------------------------------------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style=rounded, fontsize=11];
      A [label="Artwork"]; B [label="Resize 256, JPEG,\\ncenter crop 224"];
      C [label="Lightweight CNN\\nMobileNetV3-Small or\\nEfficientNet-B0"];
      D [label="LFAB inside the backbone\\n(frequency filter, FFT)"];
      E [label="Score P(AI)"]; F [label="Grad-CAM and SHAP"];
      A -> B -> C -> D -> E; E -> F;
    }""")
    st.markdown(
        "The task is only **AI generated versus human made**. The system does not say which AI tool made an "
        "image. LFAB sits early in the backbone, after block 1 for Model A and block 2 for Model B, on a 56x56 "
        "feature map. It is part of the models and is not a setting the user controls.")

    st.subheader("What the models were trained on")
    st.markdown(
        "- 10,000 images from ArtBench human, WikiArt, ArtBench AI, MidJourney and Stable Diffusion, split "
        "70/15/15, as 224x224 crops of 256x256 JPEGs.\n"
        "- Human made digital art and photographs were not in the training data. Do not feed it photographs.")

    st.subheader("Limits to keep in mind")
    st.markdown(
        "- **Main test set.** Accuracy was 91.1% (Model A), 92.3% (Model B) and 93.3% (ensemble), with an "
        "interval of about plus or minus 1.5 points.\n"
        "- **Unseen generators.** AI recall dropped to about 74% (Model A) and 80% (Model B), with ROC AUC "
        "of 0.91 and 0.92. Accuracy is not compared here because that set is not balanced.\n"
        "- **Other kinds of image.** On CIFAKE photographs accuracy was about 64%.\n"
        "- **Distortions.** Noise, heavy JPEG compression and resizing each lowered accuracy by 4 to 8 "
        "points. Deliberate attempts to fool the detector were not tested.\n"
        "- **LFAB.** It added about 0.6 accuracy points on the main test set for both models. On unseen "
        "generators Model A was 4.3 points worse with LFAB, and Model B was about the same. The learned "
        "filters stayed close to neutral.\n"
        "- **Explanations.** Grad-CAM and SHAP show where and how much, not why. Their faithfulness was not "
        "formally evaluated, and no user study was done with artists or moderators.\n"
        "- **Speed.** Reported latency is from a GPU. CPU latency was not part of the thesis.\n"
        "- **The score.** It is an uncalibrated model probability, not proof. Treat results near the "
        "threshold as uncertain.")
