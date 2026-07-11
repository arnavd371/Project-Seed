from __future__ import annotations

import plotly.graph_objects as go

from utils.pi_theme import PI_ACCENT, PI_BORDER, PI_CREAM, PI_MUTED, PI_WHITE

# Soft palette only (no pure black fills in charts)
PI_SOFT = "#6b9080"
PI_LIGHT = "#e8ede8"


def parse_star_alleles(genotype: str) -> tuple[str, str]:
    if "/" in genotype:
        parts = genotype.split("/", 1)
    elif "|" in genotype:
        parts = genotype.split("|", 1)
    else:
        parts = [genotype, genotype]
    return parts[0].strip(), parts[1].strip()


def _layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": "#333"}},
        paper_bgcolor=PI_CREAM,
        plot_bgcolor=PI_WHITE,
        height=height,
        margin={"l": 24, "r": 24, "t": 48, "b": 24},
        font={"size": 12, "color": "#333"},
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6])
    return fig


def diplotype_structure_diagram(
    gene: str,
    genotype: str,
    phenotype: str = "",
    drug: str = "",
) -> go.Figure:
    a1, a2 = parse_star_alleles(genotype)
    fig = go.Figure()

    for y, allele, label in [(4.2, a1, "Allele 1"), (2.2, a2, "Allele 2")]:
        fig.add_shape(type="rect", x0=1, y0=y - 0.35, x1=9, y1=y + 0.35,
                      line={"color": PI_BORDER, "width": 1}, fillcolor=PI_LIGHT)
        fig.add_annotation(x=0.6, y=y, text=label, showarrow=False, xanchor="right", font={"size": 11})
        fig.add_annotation(x=5, y=y, text=allele, showarrow=False, font={"size": 13, "color": PI_ACCENT})

    fig.add_annotation(
        x=5, y=5.3,
        text=f"<b>{gene}</b> diplotype: {a1} / {a2}",
        showarrow=False, font={"size": 14},
    )
    if phenotype:
        fig.add_annotation(x=5, y=0.8, text=f"Phenotype: {phenotype}", showarrow=False, font={"size": 12})
    if drug:
        fig.add_annotation(x=5, y=0.2, text=f"Example drug: {drug}", showarrow=False, font={"size": 11, "color": PI_MUTED})

    return _layout(fig, f"How to read {gene} {genotype}")


def gene_locus_overview(gene: str, alleles: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.5, y0=2.6, x1=9.5, y1=3.4,
                  line={"color": PI_BORDER, "width": 1.5}, fillcolor=PI_LIGHT)
    fig.add_annotation(x=5, y=3.0, text=f"{gene} gene (pharmacogene locus)", showarrow=False, font={"size": 12})

    n = len(alleles)
    for i, al in enumerate(alleles):
        x = 1.2 + (7.6 * i / max(n - 1, 1))
        fig.add_shape(type="circle", x0=x - 0.22, y0=1.55, x1=x + 0.22, y1=1.99,
                      line={"color": PI_ACCENT, "width": 1.5}, fillcolor=PI_WHITE)
        name = al.get("name", "?")
        fig.add_annotation(x=x, y=1.77, text=name, showarrow=False, font={"size": 10})
        fig.add_annotation(
            x=x, y=1.05,
            text=f"SAS {al.get('sas_pct', 0):.0f}%<br>EUR {al.get('eur_pct', 0):.0f}%",
            showarrow=False, font={"size": 9, "color": PI_MUTED},
        )
        role = al.get("role", "")
        if role:
            fig.add_annotation(x=x, y=0.45, text=role, showarrow=False, font={"size": 8, "color": PI_SOFT})

    return _layout(fig, f"{gene}: common star alleles at this locus", height=320)


def inheritance_flow(genotype: str, enzyme_activity: str = "normal") -> go.Figure:
    a1, a2 = parse_star_alleles(genotype)
    fig = go.Figure()
    boxes = [
        (1.5, 4.5, f"Allele A\n{a1}"),
        (5.5, 4.5, f"Allele B\n{a2}"),
        (3.5, 2.8, f"Diplotype\n{a1}/{a2}"),
        (3.5, 1.0, f"Activity\n{enzyme_activity}"),
    ]
    for x, y, text in boxes:
        fig.add_shape(type="rect", x0=x - 1.1, y0=y - 0.55, x1=x + 1.1, y1=y + 0.55,
                      line={"color": PI_BORDER, "width": 1}, fillcolor=PI_WHITE)
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font={"size": 11})

    for x0, y0, x1, y1 in [(1.5, 3.95, 3.5, 3.35), (5.5, 3.95, 3.5, 3.35), (3.5, 2.25, 3.5, 1.55)]:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line={"color": PI_MUTED, "width": 1, "dash": "dot"})

    return _layout(fig, "From two alleles to clinical phenotype", height=360)


GENE_ALLELE_MAP = {
    "CYP2C19": [
        {"name": "*1", "sas_pct": 35, "eur_pct": 55, "role": "normal function"},
        {"name": "*2", "sas_pct": 28, "eur_pct": 12, "role": "loss of function"},
        {"name": "*17", "sas_pct": 18, "eur_pct": 20, "role": "increased function"},
    ],
    "DPYD": [
        {"name": "Reference", "sas_pct": 85, "eur_pct": 80, "role": "normal"},
        {"name": "*2A", "sas_pct": 0.1, "eur_pct": 1.2, "role": "no function"},
        {"name": "*9A", "sas_pct": 10, "eur_pct": 12, "role": "decreased"},
    ],
    "CYP2D6": [
        {"name": "*1", "sas_pct": 40, "eur_pct": 45, "role": "normal"},
        {"name": "*10", "sas_pct": 38, "eur_pct": 0.2, "role": "decreased (common in SAS)"},
        {"name": "*4", "sas_pct": 2, "eur_pct": 20, "role": "no function"},
    ],
    "NUDT15": [
        {"name": "*1", "sas_pct": 95, "eur_pct": 98, "role": "normal"},
        {"name": "*3", "sas_pct": 2, "eur_pct": 0.2, "role": "no function (SAS enriched)"},
    ],
}

ACTIVITY_BY_PHENOTYPE = {
    "Intermediate Metabolizer": "reduced",
    "Poor Metabolizer": "very low / absent",
    "Normal Metabolizer": "normal",
    "Rapid Metabolizer": "increased",
    "Ultrarapid Metabolizer": "high",
}
