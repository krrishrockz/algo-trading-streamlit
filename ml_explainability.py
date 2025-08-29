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

        # --- SHAP Force (single instance)
        # Keep your existing Plotly-bar "force" as a fallback,
        # but ALSO try to produce a real SHAP force plot (HTML) + a matplotlib fallback.
        try:
            # === Original fallback figure (kept) ===
            instance_idx = 0
            # guard for short slices
            if shap_values.ndim == 2 and shap_values.shape[0] > 0:
                if instance_idx >= shap_values.shape[0]:
                    instance_idx = shap_values.shape[0]-1
                force_vals = shap_values[instance_idx]
            else:
                force_vals = np.zeros(len(features))

            fig_force = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=force_vals,
                    marker=dict(
                        color=["green" if v > 0 else "red" for v in force_vals]
                    )
                )
            ])
            fig_force.update_layout(
                title="SHAP Force Plot (Single Instance)",
                xaxis_title="Feature",
                yaxis_title="SHAP Value",
                template="plotly_white",
                height=500
            )
            results["shap_force"] = fig_force  # <-- reliable Plotly bar

            # === New: try to build a real SHAP force HTML ===
            shap_force_html = None
            shap_explanation = None
            shap_force_matplot = None

            try:
                # Build a shap.Explanation if possible
                # expected_value can be scalar or array; make it scalar for the selected instance
                if isinstance(expected_value, (list, np.ndarray)):
                    base_val = np.asarray(expected_value).squeeze()
                    base_val = float(base_val[0]) if getattr(base_val, "ndim", 0) != 0 else float(base_val)
                else:
                    base_val = float(expected_value)

                data_row = np.asarray(X.iloc[instance_idx].values) if len(X) > instance_idx else np.zeros(len(features))
                shap_explanation = shap.Explanation(
                    values=np.asarray(force_vals),
                    base_values=base_val,
                    data=data_row,
                    feature_names=features
                )

                # Try html force
                force_obj = shap.plots.force(shap_explanation, show=False)
                if hasattr(force_obj, "to_html"):
                    shap_force_html = force_obj.to_html()
                elif hasattr(force_obj, "html"):
                    shap_force_html = force_obj.html()
                elif isinstance(force_obj, str):
                    shap_force_html = force_obj

                # Matplotlib fallback (static)
                try:
                    import matplotlib.pyplot as plt
                    shap.plots.force(shap_explanation, matplotlib=True, show=False)
                    shap_force_matplot = plt.gcf()
                except Exception:
                    shap_force_matplot = None

            except Exception as inner_e:
                logging.warning(f"Could not create SHAP Explanation/HTML force: {inner_e}")

            # Store extras (non-breaking for your UI)
            if shap_force_html:
                results["shap_force_html"] = shap_force_html
            if shap_explanation is not None:
                results["shap_explanation"] = shap_explanation
            if shap_force_matplot is not None:
                results["shap_force_matplot"] = shap_force_matplot

        except Exception as e:
            logging.error(f"SHAP force error: {e}")
            # keep the key to avoid downstream KeyError
            results["shap_force"] = results.get("shap_force", None)

        return results

    except Exception as e:
        logging.error(f"General explainability error: {e}")
        return {}
