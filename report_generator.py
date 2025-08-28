from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
import os

def generate_html_report(
    stock_name,
    model_name,
    acc,
    shap_path,
    lime_path,
    chart_path,
    benchmark_df,
    report_df,
    output_path="final_report.pdf"
):
    """
    Generate a strategy comparison report PDF.
    """

    # Ensure template directory is correct
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )

    # Load template
    try:
        template = env.get_template("report_template.html")
    except Exception as e:
        raise FileNotFoundError(f"❌ Could not load report_template.html from {template_dir}: {e}")

    # Replace missing paths with None so template won’t break
    shap_path = shap_path if shap_path and os.path.exists(shap_path) else None
    lime_path = lime_path if lime_path and os.path.exists(lime_path) else None
    chart_path = chart_path if chart_path and os.path.exists(chart_path) else None

    # Render HTML
    html_content = template.render(
        stock=stock_name,
        model=model_name,
        accuracy=f"{acc:.2%}" if acc else "N/A",
        shap_path=shap_path,
        lime_path=lime_path,
        chart_path=chart_path,
        benchmark_table=benchmark_df.to_html(index=False, classes="styled-table"),
        metrics_table=report_df.to_html(classes="compact styled-table", float_format="{:.2f}".format, index=False)
    )

    # Write and convert to PDF
    HTML(string=html_content, base_url=template_dir).write_pdf(output_path)
    return output_path
