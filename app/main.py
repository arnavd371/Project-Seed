"""Seed FastAPI entrypoint for Vercel."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT))

from seed_scorer import SAS_POPULATION_NAMES, SAS_POPULATIONS, SeedScorer  # noqa: E402
from utils.hindi_templates import hindi_summary  # noqa: E402
from utils.rubric_content import (  # noqa: E402
    ACCESSIBILITY_FEATURES,
    CITATIONS,
    ETHICS_PRIVACY,
    GTM_STRATEGY,
    SDG_MAPPING,
    TARGET_AUDIENCE,
    TECH_STACK,
    WHY_AI_NOT_RULES,
)

app = FastAPI(title="Seed | बीज", version="1.0.0")

CSS = """
:root { --cream:#F5F4EF; --ink:#1a1a1a; --muted:#666; --card:#fff; --border:#000; }
* { box-sizing:border-box; }
body { margin:0; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
  background:linear-gradient(180deg,#F5F4EF 0%,#ebe8df 100%); color:var(--ink); }
a { color:inherit; }
.wrap { max-width:980px; margin:0 auto; padding:2rem 1.25rem 4rem; }
.hero { text-align:center; padding:2rem 0 1rem; }
.hero h1 { font-size:clamp(2.4rem,6vw,3.6rem); margin:0; letter-spacing:-.03em; }
.hero .sub { color:var(--muted); margin-top:.4rem; font-size:1.15rem; }
.hero .tag { margin-top:.85rem; font-size:1.05rem; line-height:1.5; }
.nav { display:flex; flex-wrap:wrap; gap:.5rem; justify-content:center; margin:1.5rem 0 2rem; }
.nav a { text-decoration:none; border:2px solid var(--border); background:var(--card);
  padding:.45rem .85rem; font-size:.9rem; box-shadow:3px 3px 0 var(--border); }
.nav a:hover { transform:translate(1px,1px); box-shadow:2px 2px 0 var(--border); }
.card { background:var(--card); border:2px solid var(--border); box-shadow:4px 4px 0 var(--border);
  padding:1.15rem 1.25rem; margin:1rem 0; }
.card h2,.card h3 { margin:0 0 .65rem; }
.muted { color:var(--muted); }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.75rem; margin:1.25rem 0; }
.stat { background:var(--card); border:2px solid var(--border); padding:1rem; text-align:center;
  box-shadow:3px 3px 0 var(--border); }
.stat .n { font-size:1.7rem; font-weight:700; }
.stat .l { font-size:.8rem; color:var(--muted); margin-top:.25rem; }
label { display:block; font-weight:600; margin:.75rem 0 .35rem; }
select,button { font:inherit; }
select { width:100%; padding:.65rem .7rem; border:2px solid var(--border); background:#fff; }
button,.btn { display:inline-block; border:2px solid var(--border); background:#000; color:#fff;
  padding:.7rem 1.1rem; cursor:pointer; text-decoration:none; box-shadow:3px 3px 0 #444; }
button:hover,.btn:hover { transform:translate(1px,1px); }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
@media (max-width:720px){ .grid2{grid-template-columns:1fr;} }
.urgency-0{border-left:8px solid #6b7280;} .urgency-1{border-left:8px solid #ca8a04;}
.urgency-2{border-left:8px solid #ea580c;} .urgency-3{border-left:8px solid #dc2626;}
table { width:100%; border-collapse:collapse; font-size:.92rem; }
th,td { border-bottom:1px solid #ddd; padding:.45rem .3rem; text-align:left; vertical-align:top; }
.footer { margin-top:2.5rem; color:var(--muted); font-size:.85rem; text-align:center; }
pre { white-space:pre-wrap; font-size:.9rem; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>🌱 Seed</h1>
      <div class="sub">बीज</div>
      <div class="tag">India-calibrated pharmacogenomics.<br>Because Indian genes are not European genes.</div>
    </div>
    <nav class="nav">
      <a href="/">Home</a>
      <a href="/analyse">Analyse</a>
      <a href="/performance">Model Performance</a>
      <a href="/impact">Impact</a>
      <a href="/ai">AI Innovation</a>
      <a href="/technical">Technical</a>
    </nav>
    {body}
    <div class="footer">Decision support only · not a diagnostic device · Open source · Offline-capable</div>
  </div>
</body>
</html>"""


@lru_cache(maxsize=1)
def get_scorer() -> SeedScorer:
    return SeedScorer()


def _load_comparison() -> dict[str, Any]:
    path = ROOT / "model" / "model_artifacts" / "comparison_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def home():
    body = """
    <div class="card">
      <p>Standard pharmacogenomic tools are calibrated on European allele frequencies.
      Seed is trained on real 1000 Genomes Project South Asian genomes, runs without LLM/API keys,
      and reports honest grouped cross-validation metrics.</p>
    </div>
    <div class="stats">
      <div class="stat"><div class="n">6,035</div><div class="l">Real SAS individual rows</div></div>
      <div class="stat"><div class="n">489</div><div class="l">South Asian genomes</div></div>
      <div class="stat"><div class="n">168</div><div class="l">Unique gene-genotype groups</div></div>
      <div class="stat"><div class="n">0</div><div class="l">Synthetic / SMOTE rows</div></div>
    </div>
    <p style="text-align:center;margin-top:1.5rem;"><a class="btn" href="/analyse">Enter Seed → Analyse</a></p>
    """
    return HTMLResponse(_page("Seed | बीज", body))


@app.get("/analyse", response_class=HTMLResponse)
def analyse_form(gene: str | None = None):
    scorer = get_scorer()
    genes = scorer.genes_available
    selected = gene if gene in genes else (genes[0] if genes else "")
    genotypes = scorer.genotypes_for_gene(selected) if selected else []
    gene_opts = "".join(
        f'<option value="{g}" {"selected" if g == selected else ""}>{g}</option>'
        for g in genes
    )
    gt_opts = "".join(
        f'<option value="{gt}">{gt} · {n} real individuals</option>'
        for gt, n in genotypes
    )
    body = f"""
    <div class="card">
      <h2>Analyse genotype</h2>
      <p class="muted">Only genotypes observed in real 1000 Genomes SAS data are listed.</p>
      <form method="get" action="/analyse">
        <label for="gene">Gene</label>
        <select id="gene" name="gene" onchange="this.form.submit()">{gene_opts}</select>
      </form>
      <form method="post" action="/analyse">
        <input type="hidden" name="gene" value="{selected}"/>
        <label for="genotype">Genotype</label>
        <select id="genotype" name="genotype" required>{gt_opts}</select>
        <p style="margin-top:1rem;"><button type="submit">Analyse Genotype</button></p>
      </form>
    </div>
    """
    return HTMLResponse(_page("Analyse · Seed", body))


@app.post("/analyse", response_class=HTMLResponse)
def analyse_submit(gene: str = Form(...), genotype: str = Form(...)):
    scorer = get_scorer()
    try:
        result = scorer.score(gene, genotype)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    cls = result["observed_clinical_significance"]
    label = result["observed_clinical_significance_label"]
    urgency_text = {
        0: "No immediate pharmacogenomic action required for this genotype.",
        1: "Moderate significance. Consider this genotype when prescribing affected drugs.",
        2: "Significant. Dose or drug-choice adjustment should be considered.",
        3: "URGENT. This genotype may cause serious harm with standard dosing of affected drugs.",
    }[cls]

    breakdown_rows = "".join(
        f"<tr><td>{SAS_POPULATION_NAMES[p]}</td><td>{p}</td>"
        f"<td>{(str(result['population_breakdown'][p]) + '%') if result['population_breakdown'][p] is not None else 'N/A'}</td></tr>"
        for p in SAS_POPULATIONS
    )
    drugs = result.get("drugs") or []
    drug_rows = "".join(
        f"<tr><td>{d}</td><td>{label}</td></tr>" for d in drugs
    ) or "<tr><td colspan='2'>No CPIC drug associations on file for this gene.</td></tr>"

    hindi = hindi_summary(
        result["gene"],
        result["genotype"],
        result["phenotype"],
        round(result["india_diplotype_freq"] * 100, 1),
        result["observed_clinical_significance"],
        result["drugs"],
    )

    pop_only = (
        f"{result['population_only_predicted_label']} ({result['population_only_confidence'] * 100:.1f}%)"
        if result.get("population_only_predicted_label") is not None
        else "N/A"
    )

    body = f"""
    <div class="card urgency-{cls}">
      <h2>{urgency_text}</h2>
      <p class="muted">{result["gene"]} {result["genotype"]} → {result["phenotype"]} ({label})</p>
    </div>
    <div class="grid2">
      <div class="card">
        <h3>Calibration ML</h3>
        <p><strong>{result["india_predicted_label"]}</strong> ({result["india_confidence"] * 100:.1f}%)</p>
      </div>
      <div class="card">
        <h3>Global baseline</h3>
        <p><strong>{result["baseline_predicted_label"]}</strong> ({result["baseline_confidence"] * 100:.1f}%)</p>
      </div>
      <div class="card">
        <h3>Population-only ML</h3>
        <p><strong>{pop_only}</strong></p>
      </div>
      <div class="card">
        <h3>Frequencies</h3>
        <p>SAS: {result["india_diplotype_freq"]:.4f}<br>
           EUR: {result["european_diplotype_freq"]:.4f}<br>
           SAS/EUR: {result["sas_vs_eur_ratio"]:.2f}<br>
           n = {result["n_real_individuals"]} real SAS individuals</p>
      </div>
    </div>
    <div class="card">
      <h3>SAS subpopulation breakdown</h3>
      <table><thead><tr><th>Subpopulation</th><th>Code</th><th>% carrying</th></tr></thead>
      <tbody>{breakdown_rows}</tbody></table>
    </div>
    <div class="card">
      <h3>Clinical interpretation</h3>
      <p><strong>{result["gene"]}</strong> ({result["full_name"]}) - CPIC evidence level
      <strong>{result["cpic_level"]}</strong>. Phenotype: <strong>{result["phenotype"]}</strong>.</p>
      <table><thead><tr><th>Affected drug</th><th>Significance</th></tr></thead>
      <tbody>{drug_rows}</tbody></table>
    </div>
    <div class="card">
      <h3>Hindi summary</h3>
      <pre>{hindi}</pre>
    </div>
    <p><a class="btn" href="/analyse?gene={result["gene"]}">Analyse another</a></p>
    """
    return HTMLResponse(_page(f"{gene} {genotype} · Seed", body))


@app.get("/api/score")
def api_score(gene: str, genotype: str):
    scorer = get_scorer()
    try:
        return JSONResponse(scorer.score(gene, genotype))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/genes")
def api_genes():
    scorer = get_scorer()
    return {"genes": scorer.genes_available}


@app.get("/api/genotypes")
def api_genotypes(gene: str):
    scorer = get_scorer()
    return {"gene": gene, "genotypes": scorer.genotypes_for_gene(gene)}


@app.get("/performance", response_class=HTMLResponse)
def performance():
    comp = _load_comparison()
    if not comp:
        body = '<div class="card"><p>comparison_results.json not found.</p></div>'
        return HTMLResponse(_page("Performance · Seed", body))

    def row(name: str, key: str) -> str:
        block = comp.get(key, {})
        acc = block.get("accuracy", "-")
        f1 = block.get("macro_f1", "-")
        return f"<tr><td>{name}</td><td>{acc}</td><td>{f1}</td></tr>"

    rows = "".join([
        row("Rule baseline", "rule_baseline"),
        row("Population-only ML", "population_only"),
        row("Calibration ML", "india_calibrated"),
        row("Global baseline ML", "baseline"),
    ])
    note = comp.get("notes") or comp.get("disclosure") or ""
    if isinstance(note, list):
        note = " ".join(str(x) for x in note)
    body = f"""
    <div class="card">
      <h2>Model performance</h2>
      <p class="muted">Group-level metrics - each genotype counts once.</p>
      <table>
        <thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="card"><h3>Notes</h3><pre>{note}</pre></div>
    """
    return HTMLResponse(_page("Performance · Seed", body))


@app.get("/impact", response_class=HTMLResponse)
def impact():
    cites = "".join(
        f'<div class="card"><strong>[{c["id"]}]</strong> {c["text"]}<br><span class="muted">{c["source"]}</span></div>'
        for c in CITATIONS
    )
    sdg = "".join(
        f"<tr><td>{s['sdg']}</td><td>{s['mapping']}</td></tr>" for s in SDG_MAPPING
    )
    audience = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in TARGET_AUDIENCE
    )
    access = "".join(f"<li>{a}</li>" for a in ACCESSIBILITY_FEATURES)
    body = f"""
    <div class="card">
      <h2>Impact &amp; Inclusion</h2>
      <p>Genomic tools trained on European cohorts can mis-estimate prescribing risk for ~1.4B South Asians.
      Seed recalibrates CPIC context with real SAS allele-frequency structure.</p>
    </div>
    <h3>Evidence</h3>{cites}
    <div class="card"><h3>Target audience</h3>
      <table><tbody>{audience}</tbody></table></div>
    <div class="card"><h3>SDG mapping</h3>
      <table><tbody>{sdg}</tbody></table></div>
    <div class="card"><h3>Accessibility</h3><ul>{access}</ul></div>
    """
    return HTMLResponse(_page("Impact · Seed", body))


@app.get("/ai", response_class=HTMLResponse)
def ai_innovation():
    gtm = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in GTM_STRATEGY)
    ethics = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ETHICS_PRIVACY)
    body = f"""
    <div class="card"><h2>Why AI (not only rules)</h2><pre>{WHY_AI_NOT_RULES}</pre></div>
    <div class="card"><h3>Go-to-market</h3><table><tbody>{gtm}</tbody></table></div>
    <div class="card"><h3>Ethics &amp; privacy</h3><table><tbody>{ethics}</tbody></table></div>
    """
    return HTMLResponse(_page("AI Innovation · Seed", body))


@app.get("/technical", response_class=HTMLResponse)
def technical():
    stack_bits = []
    for k, v in TECH_STACK.items():
        if isinstance(v, list):
            stack_bits.append(f"<tr><td>{k}</td><td>{', '.join(v)}</td></tr>")
        else:
            stack_bits.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    body = f"""
    <div class="card">
      <h2>Technical</h2>
      <p>Dual XGBoost classifiers + risk regressor on real SAS WGS-derived genotype groups.
      Grouped CV by gene|genotype prevents duplicate-row leakage.</p>
      <table><tbody>{''.join(stack_bits)}</tbody></table>
    </div>
    <div class="card">
      <h3>API</h3>
      <p><code>GET /api/genes</code> · <code>GET /api/genotypes?gene=CYP2C19</code> ·
      <code>GET /api/score?gene=CYP2C19&amp;genotype=*1/*2</code></p>
    </div>
    """
    return HTMLResponse(_page("Technical · Seed", body))


@app.get("/health")
def health():
    return {"ok": True, "root": str(ROOT), "has_data": (ROOT / "data" / "real_training_data.json").exists()}
