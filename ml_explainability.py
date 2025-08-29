import shap
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
import gymnasium as gym


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

        # --- Features (keep same order you trained with) ---
        features = ["MA10", "MA50", "RSI", "MACD"]
        X_full = df[features].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        y = pd.to_numeric(df.get("Signal", 0), errors="coerce").fillna(0).astype(int)

        if X_full.empty:
            logging.warning("generate_explainability: empty feature matrix.")
            return {}

        # === Sample to keep SHAP light & avoid tab collapsing ===
        #   - background: tiny set for Kernel/Linear explainers
        #   - sample: subset used to compute/show SHAP values
        bg_size = min(50, len(X_full))
        smp_size = min(300, len(X_full))
        X_bg = X_full.sample(bg_size, random_state=42) if len(X_full) > bg_size else X_full.copy()
        X_smp = X_full.tail(smp_size)  # prefer most-recent rows for trading

        # --- Model kind detection (so we choose a right SHAP explainer) ---
        mname = getattr(trained_model, "__class__", type(trained_model)).__name__.lower()
        is_tree_like = (
            "xgb" in mname or
            "randomforest" in mname or
            "gradientboost" in mname or
            hasattr(trained_model, "feature_importances_")
        )
        is_linear_like = ("logistic" in mname or "linear" in mname)

        # --- Pick the explainer safely ---
        import shap
        shap_values = None
        try:
            if is_tree_like:
                # very fast for tree models (XGBoost/RandomForest/etc)
                explainer = shap.TreeExplainer(trained_model)
                sv = explainer.shap_values(X_smp)
            elif is_linear_like:
                # try LinearExplainer; back off to Kernel if it fails
                try:
                    explainer = shap.LinearExplainer(trained_model, X_bg, feature_dependence="independent")
                    sv = explainer.shap_values(X_smp)
                except Exception:
                    explainer = shap.KernelExplainer(trained_model.predict_proba, X_bg)
                    sv = explainer.shap_values(X_smp, nsamples=100)
            else:
                # generic fallback (kept small to prevent UI freeze)
                explainer = shap.KernelExplainer(trained_model.predict_proba, X_bg)
                sv = explainer.shap_values(X_smp, nsamples=100)
        except Exception as e:
            logging.error(f"SHAP explainer failed: {e}")
            sv = None

        # Convert multi-class -> take one class (e.g., "Buy" class = 1) if needed
        if isinstance(sv, list):
            # pick the last class or class index 1 if 3 classes (-1,0,1)
            class_idx = 1 if (len(sv) == 3) else -1
            shap_mat = np.array(sv[class_idx])
        else:
            shap_mat = np.array(sv) if sv is not None else None

        # If nothing came back, bail gracefully
        if shap_mat is None or shap_mat.size == 0:
            logging.warning("No SHAP values computed; returning empty plots.")
            return {"shap_bar": None, "shap_beeswarm": None, "lime_plot": None, "shap_force": None}

        # Make sure shape is (n_samples, n_features)
        if shap_mat.ndim == 1:
            shap_mat = shap_mat.reshape(1, -1)

        # --- SHAP bar ---
        try:
            mean_abs_shap = np.abs(shap_mat).mean(axis=0)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=mean_abs_shap,
                y=features,
                orientation="h"
            ))
            fig_bar.update_layout(
                title="SHAP Feature Importance (Mean |SHAP Value|)",
                xaxis_title="Mean |SHAP|",
                yaxis_title="Features",
                template="plotly_white",
                height=420
            )
            results["shap_bar"] = fig_bar
        except Exception as e:
            logging.error(f"SHAP bar error: {e}")
            results["shap_bar"] = None

        # --- SHAP beeswarm (plotly scatter per feature) ---
        try:
            fig_bee = go.Figure()
            n = shap_mat.shape[0]
            for i, feat in enumerate(features):
                fig_bee.add_trace(go.Scatter(
                    x=shap_mat[:, i],
                    y=np.random.normal(i, 0.06, size=n),
                    mode="markers",
                    name=feat,
                    marker=dict(size=5, opacity=0.6)
                ))
            fig_bee.update_layout(
                title="SHAP Beeswarm (approx.)",
                xaxis_title="SHAP Value",
                yaxis=dict(showticklabels=False),
                template="plotly_white",
                height=420
            )
            results["shap_beeswarm"] = fig_bee
        except Exception as e:
            logging.error(f"SHAP beeswarm error: {e}")
            results["shap_beeswarm"] = None

        # --- LIME (single instance, lightweight) ---
        try:
            import lime
            import lime.lime_tabular
            lime_exp = lime.lime_tabular.LimeTabularExplainer(
                X_bg.values,
                feature_names=features,
                class_names=[str(c) for c in sorted(np.unique(y))],
                discretize_continuous=True
            )
            instance = X_smp.iloc[-1].values
            # handle models without predict_proba
            if hasattr(trained_model, "predict_proba"):
                pred_fn = trained_model.predict_proba
            else:
                pred_fn = lambda X: np.vstack([1 - trained_model.predict(X), trained_model.predict(X)]).T
            exp = lime_exp.explain_instance(instance, pred_fn, num_features=len(features))
            lime_list = exp.as_list()
            feat_names = [f for f, _ in lime_list]
            feat_values = [v for _, v in lime_list]

            fig_lime = go.Figure()
            fig_lime.add_trace(go.Bar(
                x=feat_values, y=feat_names, orientation="h"
            ))
            fig_lime.update_layout(
                title="LIME Feature Contributions (latest row)",
                xaxis_title="Contribution",
                yaxis_title="Features",
                template="plotly_white",
                height=420
            )
            results["lime_plot"] = fig_lime
        except Exception as e:
            logging.error(f"LIME error: {e}")
            results["lime_plot"] = None

        # --- SHAP "force" (as a compact bar for a single instance) ---
        try:
            row0 = shap_mat[-1, :]  # take most recent row
            fig_force = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=row0.tolist(),
                    marker=dict(color=["#22c55e" if v > 0 else "#ef4444" for v in row0])
                )
            ])
            fig_force.update_layout(
                title="SHAP Force (single-row bar)",
                xaxis_title="Feature",
                yaxis_title="SHAP Value",
                template="plotly_white",
                height=420
            )
            results["shap_force"] = fig_force
        except Exception as e:
            logging.error(f"SHAP force error: {e}")
            results["shap_force"] = None

        return results

    except Exception as e:
        logging.error(f"General explainability error: {e}")
        return {}

