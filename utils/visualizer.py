import plotly.graph_objects as go

from utils.pi_theme import PI_BLACK, PI_CREAM, PI_MUTED, PI_WHITE

TEAL = "#00b894"
AMBER = "#e0a94a"
RED = "#e05a5a"
BLUE = "#5da9e9"


def risk_gauge_chart(risk_score, title="Combined India Risk Score"):
    if risk_score < 0.3:
        bar_color = TEAL
    elif risk_score < 0.6:
        bar_color = AMBER
    else:
        bar_color = RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"valueformat": ".2f", "font": {"color": PI_BLACK, "size": 44}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"color": PI_MUTED, "size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": PI_MUTED},
            "bar": {"color": bar_color, "thickness": 0.35},
            "bgcolor": PI_WHITE,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.3], "color": "rgba(0, 184, 148, 0.15)"},
                {"range": [0.3, 0.6], "color": "rgba(224, 169, 74, 0.15)"},
                {"range": [0.6, 1.0], "color": "rgba(224, 90, 90, 0.15)"},
            ],
            "threshold": {
                "line": {"color": PI_BLACK, "width": 3},
                "thickness": 0.85,
                "value": risk_score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=PI_CREAM,
        plot_bgcolor=PI_WHITE,
        font={"color": PI_BLACK},
        height=280,
        margin={"l": 30, "r": 30, "t": 60, "b": 10},
    )
    return fig


def population_comparison_chart(gene, diplotype, india_freq, europe_freq, east_asia_freq=None):
    populations = ["South Asia (SAS)", "Europe (EUR)"]
    values = [india_freq, europe_freq]
    colors = [TEAL, PI_MUTED]

    if east_asia_freq is not None:
        populations.append("East Asia (EAS)")
        values.append(east_asia_freq)
        colors.append(BLUE)

    fig = go.Figure(go.Bar(
        x=populations,
        y=values,
        marker_color=colors,
        text=[f"{v * 100:.1f}%" for v in values],
        textposition="outside",
        textfont={"color": PI_BLACK},
    ))
    fig.update_layout(
        title={
            "text": "How common is this genotype in Indian patients?",
            "font": {"color": PI_BLACK, "size": 18},
        },
        xaxis={"color": PI_BLACK, "showgrid": False},
        yaxis={
            "color": PI_BLACK,
            "title": "Diplotype frequency",
            "tickformat": ".0%",
            "gridcolor": "#E5E4DC",
        },
        paper_bgcolor=PI_CREAM,
        plot_bgcolor=PI_WHITE,
        font={"color": PI_BLACK},
        height=380,
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        showlegend=False,
    )
    fig.add_annotation(
        text=f"{gene} {diplotype}",
        xref="paper", yref="paper", x=0.0, y=1.18,
        showarrow=False, font={"color": PI_MUTED, "size": 13},
    )
    return fig
