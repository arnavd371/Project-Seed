from __future__ import annotations

import numpy as np

TOP_N = 8


def _importance_fallback(model, features, feature_names, top_n=TOP_N):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"method": "none", "contributions": [], "top_positive": [], "top_negative": []}

    x = np.asarray(features, dtype=float).ravel()
    raw = importances * np.abs(x)
    if raw.sum() == 0:
        raw = importances.copy()
    order = np.argsort(raw)[::-1][:top_n]
    contributions = [
        {
            "feature": feature_names[i],
            "value": round(float(x[i]), 4),
            "contribution": round(float(raw[i]), 4),
            "direction": "positive" if x[i] >= 0 else "negative",
        }
        for i in order
    ]
    return {
        "method": "feature_importance_fallback",
        "contributions": contributions,
        "top_positive": [c for c in contributions if c["direction"] == "positive"][:3],
        "top_negative": [c for c in contributions if c["direction"] == "negative"][:3],
    }


def explain_prediction(model, features, feature_names, top_n=TOP_N):
    x = np.asarray(features, dtype=float).reshape(1, -1)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        # Multi-class: use predicted class SHAP vector
        if isinstance(shap_values, list):
            pred_class = int(model.predict(x)[0])
            sv = np.asarray(shap_values[pred_class]).ravel()
        else:
            sv = np.asarray(shap_values).ravel()
            if sv.ndim > 1:
                pred_class = int(model.predict(x)[0])
                sv = sv[pred_class] if isinstance(shap_values, list) else sv[0]

        pairs = [
            {
                "feature": feature_names[i],
                "value": round(float(x[0, i]), 4),
                "contribution": round(float(sv[i]), 4),
                "direction": "positive" if sv[i] >= 0 else "negative",
            }
            for i in range(len(feature_names))
        ]
        pairs.sort(key=lambda p: abs(p["contribution"]), reverse=True)
        top = pairs[:top_n]
        return {
            "method": "shap_tree",
            "contributions": top,
            "top_positive": [p for p in sorted(pairs, key=lambda p: p["contribution"], reverse=True) if p["contribution"] > 0][:3],
            "top_negative": [p for p in sorted(pairs, key=lambda p: p["contribution"]) if p["contribution"] < 0][:3],
        }
    except Exception:
        return _importance_fallback(model, x, feature_names, top_n=top_n)
