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
    """

    results = {}
    try:
        logging.info("=== Starting generate_explainability ===")

        # Features
        features = ["MA10", "MA50", "RSI", "MACD"]
        X = df[features].copy().fillna(0)
        y = pd.to_numeric(df["Signal"], errors="coerce").fillna(0).astype(int)

        # --- SHAP Values ---
        if model_type == "XGBoost":
            explainer = shap.TreeExplainer(trained_model)
            shap_values = explainer.shap_values(X)
        else:
            background = shap.sample(X, 50, random_state=0)
            explainer = shap.KernelExplainer(trained_model.predict_proba, background)
            shap_values = explainer.shap_values(X.iloc[:50])

        # If model is multiclass, shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

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
            exp = lime_exp.explain_instance(instance, trained_model.predict_proba, num_features=4)
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

        # --- SHAP Force (single instance) ---
        try:
            instance = X.iloc[0:1]
            force_vals = shap_values[0]

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
            results["shap_force"] = fig_force
        except Exception as e:
            logging.error(f"SHAP force error: {e}")
            results["shap_force"] = None

        return results

    except Exception as e:
        logging.error(f"General explainability error: {e}")
        return {}
