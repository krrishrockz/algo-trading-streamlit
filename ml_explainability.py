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

        # ---------------------------
        # Pick a compact sample
        # ---------------------------
        if len(X) == 0:
            logging.warning("generate_explainability: empty X")
            return {}

        # We’ll explain the *last* row shown to the user
        explain_slice = X.tail(min(300, len(X))).copy()
        instance_idx = len(explain_slice) - 1  # last row in the slice
        instance_row = explain_slice.iloc[instance_idx:instance_idx + 1]

        # ---------------------------
        # Predict probabilities (for class selection)
        # ---------------------------
        pred_proba = None
        try:
            if hasattr(trained_model, "predict_proba"):
                pred_proba = trained_model.predict_proba(instance_row)
            else:
                # Build a pseudo-proba if model lacks predict_proba
                pred = trained_model.predict(instance_row.values)
                pred_proba = np.vstack([1 - pred, pred]).T
        except Exception as e:
            logging.warning(f"predict_proba failed for class selection: {e}")

        # Default to class index 0; if we have probs, pick argmax
        class_index_used = 0
        if isinstance(pred_proba, np.ndarray) and pred_proba.ndim == 2 and pred_proba.shape[0] > 0:
            class_index_used = int(np.argmax(pred_proba[0]))

        # ---------------------------
        # Choose the best SHAP explainer for the model
        # ---------------------------
        shap_values_raw = None
        expected_value_raw = None

        try:
            if model_type in ("XGBoost", "Random Forest"):
                # Tree-based: fast & stable
                explainer = shap.TreeExplainer(trained_model, feature_perturbation="interventional")
                shap_values_raw = explainer.shap_values(explain_slice)
                expected_value_raw = explainer.expected_value

            elif model_type == "Logistic Regression":
                # Linear: exact, stable
                # Use a small background to avoid heavy compute
                bg = X.sample(min(500, len(X)), random_state=0) if len(X) > 0 else X
                explainer = shap.LinearExplainer(trained_model, bg, feature_dependence="independent")
                shap_values_raw = explainer.shap_values(explain_slice)
                expected_value_raw = explainer.expected_value

            else:
                # Fallback: Kernel (kept from your original)
                try:
                    bg = shap.sample(X, min(40, len(X)), random_state=0)
                except Exception:
                    bg = X.sample(min(40, len(X)), random_state=0) if len(X) > 0 else X
                explainer = shap.KernelExplainer(
                    trained_model.predict_proba if hasattr(trained_model, "predict_proba") else trained_model.predict,
                    bg
                )
                shap_values_raw = explainer.shap_values(explain_slice)
                expected_value_raw = explainer.expected_value
        except Exception as e:
            logging.error(f"SHAP explainer failed: {e}")

        if shap_values_raw is None:
            logging.warning("No SHAP values computed; returning empty results.")
            return results

        # If multiclass, pick the predicted class’ attribution
        shap_values = shap_values_raw
        expected_value = expected_value_raw
        if isinstance(shap_values_raw, list):
            # Guard against out-of-range indices
            if class_index_used >= len(shap_values_raw):
                class_index_used = 0
            shap_values = shap_values_raw[class_index_used]

            if isinstance(expected_value_raw, list) and len(expected_value_raw) > class_index_used:
                expected_value = expected_value_raw[class_index_used]

        shap_values = np.asarray(shap_values)  # (n_rows, n_features)
        if shap_values.ndim != 2:
            shap_values = np.atleast_2d(shap_values)

        # ---------------------------
        # SHAP Bar (feature importance)
        # ---------------------------
        try:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
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

        # ---------------------------
        # SHAP Beeswarm (scatter per feature)
        # ---------------------------
        try:
            fig_bee = go.Figure()
            for i, feat in enumerate(features):
                vals = shap_values[:, i] if shap_values.shape[1] > i else np.zeros(shap_values.shape[0])
                fig_bee.add_trace(go.Scatter(
                    x=vals,
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

        # ---------------------------
        # LIME
        # ---------------------------
        try:
            lime_exp = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=features,
                class_names=[str(c) for c in np.unique(y)],
                discretize_continuous=True
            )
            inst_vec = instance_row.values[0]
            pred_fn = trained_model.predict_proba if hasattr(trained_model, "predict_proba") else \
                      (lambda Xarr: np.vstack([1 - trained_model.predict(Xarr), trained_model.predict(Xarr)]).T)
            exp = lime_exp.explain_instance(inst_vec, pred_fn, num_features=len(features))
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

        # ---------------------------
        # SHAP “Force” – reliable Plotly bar + (try) real SHAP force
        # ---------------------------
        try:
            # Use the attributions for the *same* instance_row we picked
            if instance_idx >= shap_values.shape[0]:
                instance_idx = shap_values.shape[0] - 1
            force_vals = shap_values[instance_idx] if instance_idx >= 0 else np.zeros(len(features))

            # Plotly-bar fallback (always works)
            fig_force = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=force_vals,
                    marker=dict(color=["green" if v > 0 else "red" for v in force_vals])
                )
            ])
            fig_force.update_layout(
                title="SHAP Force Plot (Single Instance)",
                xaxis_title="Feature",
                yaxis_title="SHAP Value",
                template="plotly_white",
                height=500
            )
            results["shap_force"] = fig_force

            # Try to create a real shap.Explanation → HTML/matplotlib
            shap_force_html = None
            shap_explanation = None
            shap_force_matplot = None

            try:
                # expected value can be scalar or array
                if isinstance(expected_value, (list, np.ndarray)):
                    base_val = np.asarray(expected_value).squeeze()
                    base_val = float(base_val[0]) if getattr(base_val, "ndim", 0) != 0 else float(base_val)
                else:
                    base_val = float(expected_value)

                data_row = instance_row.values[0] if len(instance_row) else np.zeros(len(features))
                shap_explanation = shap.Explanation(
                    values=np.asarray(force_vals),
                    base_values=base_val,
                    data=np.asarray(data_row),
                    feature_names=features
                )

                try:
                    force_obj = shap.plots.force(shap_explanation, show=False)
                    if hasattr(force_obj, "to_html"):
                        shap_force_html = force_obj.to_html()
                    elif hasattr(force_obj, "html"):
                        shap_force_html = force_obj.html()
                    elif isinstance(force_obj, str):
                        shap_force_html = force_obj
                except Exception as e:
                    logging.info(f"SHAP HTML force not available: {e}")

                try:
                    import matplotlib.pyplot as plt
                    shap.plots.force(shap_explanation, matplotlib=True, show=False)
                    shap_force_matplot = plt.gcf()
                except Exception:
                    shap_force_matplot = None

            except Exception as inner_e:
                logging.warning(f"Could not construct shap.Explanation: {inner_e}")

            if shap_force_html:
                results["shap_force_html"] = shap_force_html
            if shap_explanation is not None:
                results["shap_explanation"] = shap_explanation
            if shap_force_matplot is not None:
                results["shap_force_matplot"] = shap_force_matplot

        except Exception as e:
            logging.error(f"SHAP force error: {e}")
            results["shap_force"] = results.get("shap_force", None)

        return results

    except Exception as e:
        logging.error(f"General explainability error: {e}")
        return {}
