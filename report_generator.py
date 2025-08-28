# report_generator.py
# Safe report generator that doesn't hard-require WeasyPrint (works on Streamlit Cloud).
from __future__ import annotations
import datetime as _dt
from typing import Any, Dict

# Try optional PDF engine
try:
    from weasyprint import HTML  # type: ignore
    _HAS_WEASYPRINT = True
except Exception:
    HTML = None  # type: ignore
    _HAS_WEASYPRINT = False


def _html_shell(title: str, body_html: str) -> str:
    """Wrap a body section in a minimal, print-friendly HTML shell."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }}
  h1,h2,h3 {{ margin: 0 0 12px; }}
  .muted {{ color: #666; font-size: 12px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
  th {{ background: #f5f7fa; text-align: left; }}
  .kpi {{ display: inline-block; min-width: 140px; padding: 8px 10px; margin: 6px 10px 6px 0; background:#f6f9fc; border-radius: 8px; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def generate_html_report(context: Dict[str, Any]) -> str:
    """
    Build an HTML summary string for download/preview.
    context can include:
      - title: str
      - symbol: str
      - period: str
      - run_time: datetime or str
      - kpis: dict[str, Any]
      - notes: str
      - tables: dict[name -> HTML string or simple dict/list]
    """
    title = context.get("title") or "Algorithmic Trading Report"
    symbol = context.get("symbol") or ""
    period = context.get("period") or ""
    run_time = context.get("run_time") or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # KPIs (render if provided)
    kpi_html = ""
    kpis = context.get("kpis") or {}
    if isinstance(kpis, dict) and kpis:
        chips = []
        for k, v in kpis.items():
            chips.append(f'<div class="kpi"><b>{k}</b><br>{v}</div>')
        kpi_html = "<div>" + "".join(chips) + "</div>"

    # Tables (accept raw HTML strings, or simple lists/dicts)
    tables_html = ""
    tables = context.get("tables") or {}
    if isinstance(tables, dict) and tables:
        for name, tbl in tables.items():
            tables_html += f"<h3>{name}</h3>"
            if isinstance(tbl, str) and "<table" in tbl.lower():
                tables_html += tbl
            else:
                # build a simple table from sequences/dicts
                import pandas as _pd
                try:
                    df = _pd.DataFrame(tbl)
                except Exception:
                    df = _pd.DataFrame({"value": [tbl]})
                tables_html += df.to_html(index=False, border=0)

    notes = context.get("notes") or ""

    body = f"""
<h1>{title}</h1>
<p class="muted">Symbol: <b>{symbol}</b> &nbsp;|&nbsp; Period: <b>{period}</b> &nbsp;|&nbsp; Generated: {run_time}</p>
{kpi_html}
{tables_html}
{('<h3>Notes</h3><p>' + notes + '</p>') if notes else ''}
"""
    return _html_shell(title, body)


def html_to_pdf_bytes(html_str: str) -> bytes | None:
    """
    Try to render HTML to PDF. Returns bytes on success, or None if
    WeasyPrint is not available (e.g., Streamlit Community Cloud).
    """
    if _HAS_WEASYPRINT and HTML is not None:
        return HTML(string=html_str).write_pdf()
    return None
