"""
results_data.py -- static numbers copied from the final manuscript (Version 15), so the demo's Results tab
matches the thesis. Table numbers refer to the manuscript. Nothing here is computed live.
"""
import pandas as pd


def _df(rows, columns, index):
    return pd.DataFrame(rows, columns=columns).set_index(index)


def main_test_set():  # Table 7
    return _df([
        ["Accuracy", 0.9113, 0.9227, 0.9333],
        ["Precision", 0.9141, 0.9043, 0.9210],
        ["Recall", 0.9080, 0.9453, 0.9480],
        ["F1 score", 0.9110, 0.9244, 0.9343],
        ["ROC AUC", 0.9735, 0.9787, 0.9850],
        ["Misclassified (of 1,500)", 133, 116, 100],
    ], ["Metric", "Model A", "Model B", "Ensemble (A + B)"], "Metric")


def source_errors():  # Table 8
    return _df([
        ["ArtBench Human", 375, "33 (8.8%)", "41 (10.9%)"],
        ["WikiArt", 375, "31 (8.3%)", "34 (9.1%)"],
        ["ArtBench AI", 300, "38 (12.7%)", "21 (7.0%)"],
        ["MidJourney", 225, "13 (5.8%)", "9 (4.0%)"],
        ["Stable Diffusion", 225, "18 (8.0%)", "11 (4.9%)"],
    ], ["Data source", "Images", "Model A errors", "Model B errors"], "Data source")


def ablation():  # Table 9
    return _df([
        ["Accuracy", 0.9113, 0.9053, 0.9227, 0.9167],
        ["Precision", 0.9141, 0.8781, 0.9043, 0.8911],
        ["Recall", 0.9080, 0.9413, 0.9453, 0.9493],
        ["F1 score", 0.9110, 0.9086, 0.9244, 0.9193],
        ["ROC AUC", 0.9735, 0.9731, 0.9787, 0.9753],
    ], ["Metric", "A with LFAB", "A without LFAB", "B with LFAB", "B without LFAB"], "Metric")


def distortions():  # Table 11
    return pd.DataFrame([
        ["A", "Clean", 0.9120, 0.9153, 0.9080, 0.9116, 0.9735],
        ["A", "Noise", 0.8353, 0.9299, 0.7253, 0.8150, 0.9389],
        ["A", "JPEG compression", 0.8427, 0.9119, 0.7587, 0.8282, 0.9345],
        ["A", "Resize", 0.8700, 0.9039, 0.8280, 0.8643, 0.9515],
        ["B", "Clean", 0.9227, 0.9043, 0.9453, 0.9244, 0.9787],
        ["B", "Noise", 0.8493, 0.8333, 0.8733, 0.8529, 0.9281],
        ["B", "JPEG compression", 0.8513, 0.9230, 0.7667, 0.8376, 0.9455],
        ["B", "Resize", 0.8687, 0.8900, 0.8413, 0.8650, 0.9528],
    ], columns=["Model", "Condition", "Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"]
    ).set_index(["Model", "Condition"])


def distortion_accuracy_chart():
    d = distortions().reset_index()
    out = d.pivot(index="Condition", columns="Model", values="Accuracy")
    out = out.loc[["Clean", "Noise", "JPEG compression", "Resize"]]
    out.columns = [f"Model {c}" for c in out.columns]
    return out


def cifake():  # Table 12
    return _df([
        ["Accuracy", 0.6440, 0.6447],
        ["Precision", 0.7222, 0.7174],
        ["Recall", 0.4680, 0.4773],
        ["F1 score", 0.5680, 0.5733],
        ["ROC AUC", 0.7130, 0.7154],
    ], ["Metric", "Model A", "Model B"], "Metric")


def unseen_generators():  # Table 13 (accuracy left out, see the manuscript: the set is not balanced)
    return _df([
        ["Model A with LFAB", 0.7373, 0.9124, 0.6480, 0.8267, 0.9160],
        ["Model A without LFAB", 0.8247, 0.9206, 0.7480, 0.9013, 0.8693],
        ["Model B with LFAB", 0.7980, 0.9187, 0.7493, 0.8467, 0.9000],
        ["Model B without LFAB", 0.8100, 0.9175, 0.7520, 0.8680, 0.8840],
    ], ["Model", "AI recall", "ROC AUC", "Detected AI Pastiche", "Detected benjaminStreltzin",
        "Human specificity"], "Model")


def efficiency():  # Table 14
    return _df([
        ["Parameters", "1,121,746", "4,192,710"],
        ["Model size, 32 bit parameters (MB)", "4.28", "15.99"],
        ["Saved checkpoint (MB)", "4.46", "16.39"],
        ["After INT8 dynamic quantization (MB)", "3.95", "15.93"],
        ["FLOPs (giga, 224 x 224 input)", "0.124", "0.832"],
        ["Latency, Tesla T4 GPU, batch size 1 (ms)", "6.53", "9.16"],
        ["Peak GPU memory (MB)", "15.00", "34.89"],
    ], ["Measure", "Model A", "Model B"], "Measure")


def benchmark():  # Table 15
    return _df([
        ["Zhang et al. SCADET (published)", 0.8100, 0.9620],
        ["Model A", 0.9113, 0.9735],
        ["Model B", 0.9227, 0.9787],
    ], ["Model", "Accuracy", "ROC AUC"], "Model")


def lfab_learned():  # Table 10 (selected rows)
    return _df([
        ["Final alpha (started at 0.1)", "0.0808", "0.3412"],
        ["Gain map, min to max (1.0 is neutral)", "0.948 to 1.049", "0.903 to 1.148"],
        ["Mean change LFAB makes to features", "7.00%", "4.67%"],
    ], ["Measure", "Model A", "Model B"], "Measure")


# Mean absolute SHAP per band at the final diagnosis (Table 6), in probability units.
BAND_SHAP_REFERENCE = {"Model A": 0.00942, "Model B": 0.01135}
