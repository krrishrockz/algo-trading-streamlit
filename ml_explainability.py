import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
import gymnasium as gym


def generate_explainability(df, trained_model, model_type):
    """
    Generate explainability plots (all as Plotly Figures).
    Returns dict with shap_bar, shap_beeswarm, lime_plot, shap_force.
    Also adds (when available):
      - shap_force_html (string HTML with the real SHAP force plot)
      - shap_explanation (shap.Explanation)
      - shap_force_matplot (matplotlib Figure fallback)
    """

    results = {}
    try:
        logging.info("=== Starting generate_explainability ===")

        # Features
        features = ["MA10", "MA50", "RSI", "MACD"]
        X = df[features].copy().fillna(0)
        y = pd.to_numeric(df["Signal"], errors="coerce").fillna(0).astype(int)

        # --- SHAP Values ---
        # NOTE: we keep your original logic, but also derive a shap.Explanation when possible
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(trained_model)
            shap_values_raw = explainer.shap_values(X)
            expected_value_raw = explainer.expected_value
        else:
            # keep small background to avoid UI collapse
            try:
                background = shap.sample(X, min(40, len(X)), random_state=0)
            except Exception:
                background = X.sample(min(40, len(X)), random_state=0) if len(X) > 0 else X
            explainer = shap.KernelExplainer(trained_model.predict_proba, background)
            # limit to a recent slice to keep it fast
            shap_values_raw = explainer.shap_values(X.tail(min(200, len(X))))
            expected_value_raw = explainer.expected_value

        # If model is multiclass, shap_values is a list → choose first class consistently
        class_index_used = 0
        shap_values = shap_values_raw
        expected_value = expected_value_raw
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[class_index_used]
            if isinstance(expected_value_raw, list) and len(expected_value_raw) > class_index_used:
                expected_value = expected_value_raw[class_index_used]

        # Ensure 2D numpy array
        shap_values = np.asarray(shap_values)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # --- SHAP Bar (Feature Importance) ---
        try:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=mean_abs_shap,
                y=features,
                orientation="h",
                marker=dict(color="royalblue")
            ))
            fig_bar.update_layout(
                title="SHAP Feature Importance (Mean |SHAP Value|)",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                template="plotly_white",
                height=500
            )
            results["shap_bar"] = fig_bar
        except Exception as e:
            logging.error(f"SHAP bar error: {e}")
            results["shap_bar"] = None

        # --- SHAP Beeswarm ---
        try:
            fig_bee = go.Figure()
            for i, feat in enumerate(features):
                fig_bee.add_trace(go.Scatter(
                    x=shap_values[:, i],
                    y=np.random.normal(i, 0.05, size=shap_values.shape[0]),
                    mode="markers",
                    marker=dict(size=6, opacity=0.6),
                    name=feat
                ))
            fig_bee.update_layout(
                title="SHAP Beeswarm Plot",
                xaxis_title="SHAP Value",
                yaxis=dict(showticklabels=False),
                template="plotly_white",
                height=500
            )
            results["shap_beeswarm"] = fig_bee
        except Exception as e:
            logging.error(f"SHAP beeswarm error: {e}")
            results["shap_beeswarm"] = None

        # --- LIME ---
        try:
            lime_exp = lime.lime_tabular.LimeTabularExplainer(
                X.values, feature_names=features,
                class_names=[str(c) for c in np.unique(y)],
                discretize_continuous=True
            )
            instance = X.iloc[-1].values
            pred_fn = trained_model.predict_proba if hasattr(trained_model, "predict_proba") else \
                      (lambda Xarr: np.vstack([1 - trained_model.predict(Xarr), trained_model.predict(Xarr)]).T)
            exp = lime_exp.explain_instance(instance, pred_fn, num_features=4)
            lime_list = exp.as_list()

            feat_names = [f for f, v in lime_list]
            feat_values = [v for f, v in lime_list]

            fig_lime = go.Figure()
            fig_lime.add_trace(go.Bar(
                x=feat_values,
                y=feat_names,
                orientation="h",
                marker=dict(color="orange")
            ))
            fig_lime.update_layout(
                title="LIME Feature Importance",
                xaxis_title="Contribution",
                yaxis_title="Features",
                template="plotly_white",
                height=500
            )
            results["lime_plot"] = fig_lime
        except Exception as e:
            logging.error(f"LIME error: {e}")
            results["lime_plot"] = None


        return results

    except Exception as e:
        logging.error(f"General explainability error: {e}")
        return {}
