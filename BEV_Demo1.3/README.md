# AI Generated Art Detector (thesis demo)

Streamlit proof of concept for the capstone "Enhancing Lightweight CNN Models with Explainable AI for
Detecting AI-Generated Art on Online Platforms". It matches the final models (Version 15, LFAB v4).

## Run it
1. Put `mobilenet_lfab_model.pth` and `efficientnet_lfab_model.pth` in `weights/` (see `weights/README.txt`).
2. `pip install -r requirements.txt`
3. Optional, builds the sample gallery from your test split (see the top of `make_samples.py`):
   `python make_samples.py --splits test_split.csv --images resized_dataset`
4. `streamlit run app.py`

## Files
- `app.py` the interface (tabs: Detector, Explain, Thesis results)
- `inference.py` preprocessing, prediction, Grad-CAM, SHAP, speed test
- `model_def.py` copied from Notebook 4. If you change LFAB or ProposedAIDetector in the notebooks, re-copy it.
- `results_data.py` the thesis tables, copied from the manuscript
- `make_samples.py` builds `samples/` and `samples/samples.json`

Set the environment variable `WEIGHTS_DIR` to load the checkpoints from another folder.
