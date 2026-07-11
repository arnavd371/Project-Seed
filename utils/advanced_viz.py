from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.pi_theme import PI_BLACK, PI_CREAM, PI_MUTED, PI_WHITE

PI_ACCENT = "#4a7c59"
PI_WARN = "#b8860b"
PI_DANGER = "#c45c5c"


def _pi_layout(fig, title="", height=420):
    fig.update_layout(
        title={"text": title, "font": {"family": "IBM Plex Serif", "size": 16, "color": PI_BLACK}},
        paper_bgcolor=PI_CREAM,
        plot_bgcolor=PI_WHITE,
        font={"family": "Source Sans 3", "color": PI_BLACK, "size": 12},
        height=height,
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
    )
    fig.update_xaxes(showgrid=False, linecolor=PI_BLACK, tickfont={"color": PI_BLACK})
    fig.update_yaxes(gridcolor="#E5E4DC", linecolor=PI_BLACK, tickfont={"color": PI_BLACK})
    return fig


def animated_frequency_shift(simulation: dict, gene: str, genotype: str) -> go.Figure:
    sim = simulation.get("genotypes", {}).get(genotype, simulation)
    sas = sim.get("sas_freqs", [])
    eur = sim.get("eur_freqs", [])
    if len(sas) == 0:
        sas = np.array([sim.get("sas_mean", 0)])
        eur = np.array([sim.get("eur_mean", 0)])

    n_frames = min(len(sas), 60)
    step = max(1, len(sas) // n_frames)
    indices = list(range(0, len(sas), step))[:n_frames]

    fig = go.Figure(
        data=[
            go.Bar(name="South Asian (SAS)", x=["SAS"], y=[sas[0]], marker_color=PI_ACCENT),
            go.Bar(name="European (EUR)", x=["EUR"], y=[eur[0]], marker_color=PI_MUTED),
        ]
    )
    frames = []
    for i in indices:
        frames.append(
            go.Frame(
                data=[
                    go.Bar(x=["SAS"], y=[sas[i]], marker_color=PI_ACCENT),
                    go.Bar(x=["EUR"], y=[eur[i]], marker_color=PI_MUTED),
                ],
                name=str(i),
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.1, "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
        }],
        barmode="group",
    )
    return _pi_layout(fig, title=f"{gene} {genotype}: Bootstrap SAS vs EUR Frequency Shift", height=400)


def gene_level_heatmap(gene: str, genotype_results: dict) -> go.Figure:
    genotypes = list(genotype_results.keys())
    sas_vals = [genotype_results[g].get("sas_mean", 0) for g in genotypes]
    eur_vals = [genotype_results[g].get("eur_mean", 0) for g in genotypes]
    z = [sas_vals, eur_vals]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=genotypes,
        y=["SAS", "EUR"],
        colorscale=[[0, "#f5f4ef"], [0.5, "#a8c5b0"], [1, "#4a7c59"]],
        text=[[f"{v:.1%}" if v < 1 else f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar={"title": "Freq"},
    ))
    return _pi_layout(fig, title=f"{gene}: Genotype Frequency Heatmap (SAS vs EUR)", height=360)


def gnomad_comparison_chart(
    gene: str,
    allele: str,
    gnomad_sas: float,
    gnomad_nfe: float,
    observed_sas: float | None = None,
    observed_eur: float | None = None,
) -> go.Figure:
    labels = ["gnomAD SAS", "gnomAD NFE (EUR)"]
    values = [gnomad_sas, gnomad_nfe]
    colors = [PI_ACCENT, PI_MUTED]
    if observed_sas is not None:
        labels.append("1000G SAS (observed)")
        values.append(observed_sas)
        colors.append(PI_WARN)
    if observed_eur is not None:
        labels.append("1000G EUR (observed)")
        values.append(observed_eur)
        colors.append(PI_DANGER)

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v * 100:.2f}%" for v in values],
        textposition="outside",
    ))
    fig.update_yaxes(tickformat=".1%", title="Allele frequency")
    return _pi_layout(fig, title=f"{gene} {allele}: gnomAD South Asian vs European", height=400)


def shap_contribution_chart(contributions: list[dict]) -> go.Figure:
    if not contributions:
        fig = go.Figure()
        return _pi_layout(fig, title="Feature contributions (unavailable)", height=200)

    names = [c["feature"] for c in contributions][::-1]
    vals = [c["contribution"] for c in contributions][::-1]
    colors = [PI_ACCENT if v >= 0 else PI_DANGER for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_xaxes(title="Contribution")
    method = "SHAP" if any(abs(v) > 0 for v in vals) else ""
    return _pi_layout(fig, title=f"Why this urgency? Feature contributions ({method})", height=max(280, 40 * len(names)))
