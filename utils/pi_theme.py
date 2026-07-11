PI_CREAM = "#F5F4EF"
PI_BLACK = "#000000"
PI_WHITE = "#FFFFFF"
PI_MUTED = "#686868"
PI_BORDER = "#D1D5DB"
PI_DIVIDER = "#E5E4DC"

PI_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500;600&family=Source+Sans+3:wght@400;500;600&display=swap');

    .stApp {{
        background-color: {PI_CREAM};
        color: {PI_BLACK};
        font-family: 'Source Sans 3', system-ui, sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {PI_CREAM};
        border-right: 1px solid {PI_BORDER};
    }}

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        color: {PI_MUTED};
    }}

    h1, h2, h3, h4 {{
        font-family: 'IBM Plex Serif', Georgia, serif !important;
        color: {PI_BLACK} !important;
        font-weight: 500 !important;
    }}

    p, li, span, label, .stCaption {{
        color: {PI_BLACK};
    }}

    .pi-box {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.55;
    }}

    .pi-box-title {{
        font-weight: 600;
        color: {PI_BLACK};
        margin-bottom: 0.35rem;
    }}

    .pi-box-meta {{
        color: {PI_MUTED};
        font-size: 0.75rem;
        margin-bottom: 0.35rem;
    }}

    .pi-hero-title {{
        font-family: 'IBM Plex Serif', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 500;
        color: {PI_BLACK};
        margin-bottom: 0.25rem;
    }}

    .pi-hero-sub {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.95rem;
        color: {PI_MUTED};
        margin-bottom: 1rem;
    }}

    .pi-intro {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        padding: 1.25rem 1.5rem;
        max-width: 860px;
        font-family: 'Source Sans 3', system-ui, sans-serif;
        font-size: 1rem;
        line-height: 1.65;
        color: {PI_BLACK};
        margin: 0 auto 1.5rem;
    }}

    .pi-stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1.25rem 0;
    }}

    @media (max-width: 768px) {{
        .pi-stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    .pi-stat {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        padding: 0.9rem;
        text-align: center;
    }}

    .pi-stat-num {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: {PI_BLACK};
    }}

    .pi-stat-label {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.72rem;
        color: {PI_MUTED};
        margin-top: 0.25rem;
    }}

    .pi-urgency-0 {{ border-left: 4px solid #4a7c59; }}
    .pi-urgency-1 {{ border-left: 4px solid #9a7b2e; }}
    .pi-urgency-2 {{ border-left: 4px solid #a84848; }}
    .pi-urgency-3 {{ border-left: 4px solid {PI_BLACK}; background: #f0ece4; }}

    .pi-caveat {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        padding: 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        line-height: 1.55;
        color: {PI_BLACK};
    }}

    div[data-testid="stPlotlyChart"] {{
        background: transparent !important;
    }}

    div[data-testid="stMetric"] {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        padding: 0.75rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: {PI_MUTED};
    }}

    .stTabs [aria-selected="true"] {{
        color: {PI_BLACK} !important;
        text-decoration: underline;
        text-underline-offset: 4px;
    }}

    div[data-testid="stExpander"] {{
        background: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
    }}

    .stButton > button[kind="primary"] {{
        background: {PI_BLACK};
        color: {PI_WHITE};
        border: 1px solid {PI_BLACK};
        box-shadow: 3px 3px 0 0 {PI_BLACK};
        border-radius: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {PI_BLACK};
        color: {PI_WHITE};
        transform: translate(1px, 1px);
        box-shadow: 2px 2px 0 0 {PI_BLACK};
    }}

    .large-text .pi-box,
    .large-text .pi-intro,
    .large-text .pi-stat-label,
    .large-text .pi-hero-sub {{
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }}

    .large-text .pi-stat-num {{
        font-size: 2rem !important;
    }}

    .rubric-check {{
        display: flex; align-items: flex-start; gap: 0.5rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.78rem; margin-bottom: 0.35rem;
    }}

    .rubric-check-yes::before {{
        content: "✓"; color: {PI_BLACK}; font-weight: 700;
    }}
</style>
"""


def pi_box(title: str, body: str, meta: str = "") -> str:
    meta_html = f'<div class="pi-box-meta">{meta}</div>' if meta else ""
    return (
        f'<div class="pi-box">'
        f'<div class="pi-box-title">{title}</div>'
        f"{meta_html}"
        f"<div>{body}</div>"
        f"</div>"
    )
