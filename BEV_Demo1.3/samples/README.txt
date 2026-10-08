Optional. Any .jpg, .jpeg, .png or .webp file placed here becomes a one click sample in the Detector tab.
Run make_samples.py to fill this folder from your test split, or add your own images.
samples.json (optional) can hold {"file.jpg": {"title": "...", "label": "AI" or "Human", "note": "...",
"preprocessed": true}}. Use preprocessed=true only for files copied straight out of resized_dataset.
