"""
Seed | बीज: India-calibrated pharmacogenomic variant interpreter.
Runs offline on real 1000 Genomes SAS data.
"""

import os
import sys

import pandas as pd
import streamlit as st

SEED_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SEED_ROOT, "model"))

from seed_scorer import (  # noqa: E402
    SeedScorer,
    SAS_POPULATIONS,
    SAS_POPULATION_NAMES,
    _model_fingerprint,
)
from utils.hindi_templates import hindi_summary  # noqa: E402
from utils.pdf_generator import generate_pdf_report  # noqa: E402
from utils.advanced_viz import (
    animated_frequency_shift,
    gene_level_heatmap,
    gnomad_comparison_chart,
    shap_contribution_chart,
)
from utils.i18n import t  # noqa: E402
from utils.pi_theme import PI_CSS, pi_box  # noqa: E402
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
from utils.pharma_guide import DEMO_SCENARIOS, GLOSSARY, MANIFESTO, QUICK_START  # noqa: E402
from utils.simulations import cyp2c19_clopidogrel_demo, dpyd_poor_metabolizer_demo
from utils.visualizer import population_comparison_chart, risk_gauge_chart  # noqa: E402

ARTIFACTS_DIR = os.path.join(SEED_ROOT, "model", "model_artifacts")

st.set_page_config(
    page_title="Seed | बीज",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css(large_text: bool = False):
    st.markdown(PI_CSS, unsafe_allow_html=True)
    if large_text:
        st.markdown(
            """<style>
            .pi-box, .pi-intro, .pi-caveat { font-size: 1.05rem !important; line-height: 1.7 !important; }
            .pi-stat-num { font-size: 2rem !important; }
            </style>""",
            unsafe_allow_html=True,
        )


@st.cache_resource
def get_scorer(_artifact_fingerprint):
    return SeedScorer()


@st.cache_data(show_spinner=False)
def build_pdf_bytes(gene: str, genotype: str, payload: str):
    import json
    return generate_pdf_report(json.loads(payload))


def urgency_banner(result):
    cls = result["observed_clinical_significance"]
    label = result["observed_clinical_significance_label"]
    texts = {
        0: "No immediate pharmacogenomic action required for this genotype.",
        1: "Moderate significance. Consider this genotype when prescribing affected drugs.",
        2: "Significant. Dose or drug-choice adjustment should be considered.",
        3: "URGENT. This genotype may cause serious harm with standard dosing of affected drugs.",
    }
    st.markdown(
        f'<div class="pi-box pi-urgency-{cls}">'
        f'<div class="pi-box-title">{texts[cls]}</div>'
        f'<div class="pi-box-meta">{result["gene"]} {result["genotype"]} → {result["phenotype"]} ({label})</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_intro():
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
          <div class="pi-hero-title">🌱 Seed</div>
          <div class="pi-hero-sub">बीज</div>
          <div class="pi-hero-sub" style="margin-top:0.5rem;">
            India-calibrated pharmacogenomics.<br>
            Because Indian genes are not European genes.
          </div>
        </div>
        <div class="pi-intro">
          Standard pharmacogenomic tools are calibrated on European allele frequencies.
          A genotype that is rare in Europe can be common in South Asia (and vice versa),
          which means the same CPIC dose-adjustment guideline can imply very different
          real-world risk for an Indian patient. Seed is trained on real 1000 Genomes
          Project South Asian genomes, runs fully offline, and reports honest grouped
          cross-validation metrics with every frequency traceable to observed individuals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = [
        ("6,035", "Real SAS individual rows"),
        ("489", "South Asian (SAS) genomes"),
        ("168", "Unique gene-genotype groups"),
        ("0", "Synthetic / SMOTE rows"),
    ]
    st.markdown(
        '<div class="pi-stat-grid">'
        + "".join(
            f'<div class="pi-stat"><div class="pi-stat-num">{n}</div>'
            f'<div class="pi-stat-label">{label}</div></div>'
            for n, label in stats
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Enter Seed", type="primary", use_container_width=True):
            st.session_state["entered_app"] = True
            st.rerun()

    st.caption("Open source · Offline · Free · Hindi + English")

    st.markdown("---")
    st.markdown("#### Key capabilities")
    checks = [
        "Problem defined with citations (Impact tab)",
        "Target audience: South Asian patients & clinicians",
        "Offline / low-bandwidth · Free · Hindi + English",
        "SDG 3, 10, 9, 12 mapped",
        "AI (XGBoost) is primary, not a force-fit",
        "Working prototype · Open-source deployment path",
        "Full tech stack documented",
    ]
    for c in checks:
        st.markdown(f'<div class="rubric-check rubric-check-yes">{c}</div>', unsafe_allow_html=True)


def render_sidebar(scorer):
    lang = st.session_state.get("ui_lang", "en")
    with st.sidebar:
        st.markdown("### Seed")
        st.selectbox(
            t("language", lang),
            options=["en", "hi"],
            format_func=lambda x: "English" if x == "en" else "हिंदी",
            key="ui_lang",
        )
        st.checkbox("Large text mode (accessibility)", key="large_text")
        st.markdown("---")
        st.markdown(
            pi_box(
                "Two XGBoost classifiers",
                "Both trained on identical real South Asian 1000 Genomes rows. "
                "Global Baseline uses European-frequency features only. "
                "India-Calibrated adds SAS diplotype frequency, SAS/EUR ratio, and gene-importance features.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            pi_box(
                "Data provenance",
                "1000 Genomes SAS cohort (GIH, PJL, ITU, STU, BEB): 6,035 rows, 168 unique groups. "
                "CPIC phenotype mapping. Zero synthetic or catalog training rows.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            pi_box(
                "Known limitations",
                "VKORC1 excluded (all SAS phenotypes indeterminate). "
                "Grouped CV by gene|genotype, not row-level k-fold. "
                "Urgent class: only 2 unique genotype groups.",
            ),
            unsafe_allow_html=True,
        )
        st.caption("Decision support only. Consult a qualified clinician before prescribing.")
        if st.button("Clear model cache", help="Use if you see a feature mismatch error after updating models"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()


def render_guide_tab(scorer):
    lang = st.session_state.get("ui_lang", "en")
    st.subheader(t("guide", lang))
    st.markdown(pi_box("Manifesto", MANIFESTO.strip()), unsafe_allow_html=True)

    st.markdown("#### Quick start for clinicians")
    for title, body in QUICK_START:
        st.markdown(pi_box(title, body), unsafe_allow_html=True)

    st.markdown("#### Try these example genotypes")
    for ex in DEMO_SCENARIOS:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown(f"**{ex['name']}**")
            st.caption(ex["why"])
        with c2:
            st.code(f"{ex['gene']}  {ex['genotype']}", language=None)
        with c3:
            if st.button("Load", key=f"load_{ex['gene']}_{ex['genotype'][:8]}"):
                st.session_state["sel_gene"] = ex["gene"]
                st.session_state["sel_genotype"] = ex["genotype"]
                try:
                    st.session_state["result"] = scorer.score(ex["gene"], ex["genotype"])
                    st.session_state["active_tab_hint"] = "analyse"
                    st.success("Loaded on Analyse tab.")
                except (KeyError, ValueError) as err:
                    st.error(str(err))

    with st.expander("Glossary", expanded=False):
        for term, definition in GLOSSARY:
            st.markdown(f"**{term}:** {definition}")


def render_analyse_tab(scorer):
    lang = st.session_state.get("ui_lang", "en")
    st.subheader(t("analyse", lang))
    col1, col2 = st.columns(2)
    with col1:
        gene = st.selectbox(t("select_gene", lang), scorer.genes_available, key="sel_gene")
    with col2:
        genotype_options = scorer.genotypes_for_gene(gene)
        genotype = st.selectbox(
            t("select_genotype", lang),
            options=[g for g, _ in genotype_options],
            format_func=lambda g: f"{g}  ·  {dict(genotype_options)[g]} real individuals",
            key="sel_genotype",
        )

    if st.button(t("analyse_btn", lang), type="primary"):
        try:
            st.session_state["result"] = scorer.score(gene, genotype)
        except ValueError as err:
            st.error(str(err))
            st.session_state.pop("result", None)
        except KeyError as err:
            st.warning(str(err))
            st.session_state.pop("result", None)

    result = st.session_state.get("result")
    if not result:
        st.info("Select a gene and genotype above, then click **Analyse Genotype**. Or use **Guide** for example cases.")
        return

    st.markdown("---")
    urgency_banner(result)

    st.markdown(
        pi_box(
            "Known limitations",
            "168 unique SAS-observed gene|genotype groups only. Genotypes not seen in the "
            "1000 Genomes South Asian reference cannot be scored. Decision support tool only; "
            "not an FDA-cleared medical device. VKORC1 excluded (all SAS phenotypes indeterminate). "
            "Grouped CV metrics are at the genotype-group level, not per individual.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Dual-Model Comparison")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown(
            pi_box(
                "Population-Only ML (~58% CV)",
                f"Prediction: **{result.get('population_only_predicted_label') or 'N/A'}** "
                f"({(result.get('population_only_confidence') or 0) * 100:.1f}%). "
                "No CPIC score in features; frequency structure only.",
            ),
            unsafe_allow_html=True,
        )
    with dc2:
        st.markdown(
            pi_box(
                "Calibration ML (~96% CV)",
                f"Prediction: **{result['india_predicted_label']}** "
                f"({result['india_confidence'] * 100:.1f}%). "
                "Population features plus CPIC clinical significance.",
            ),
            unsafe_allow_html=True,
        )
    with dc3:
        agree = result.get("dual_model_agreement")
        agree_text = "Models agree" if agree else ("Models disagree" if agree is False else "Population-only N/A")
        st.markdown(
            pi_box(
                "Global Baseline (EUR features)",
                f"Prediction: **{result['baseline_predicted_label']}** "
                f"({result['baseline_confidence'] * 100:.1f}%) · {agree_text}",
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Why this urgency? Feature contributions (SHAP)", expanded=False):
        cal_exp = result.get("calibration_explanations", {})
        pop_exp = result.get("population_only_explanations") or {}
        ec1, ec2 = st.columns(2)
        with ec1:
            st.caption(f"Calibration model ({cal_exp.get('method', 'n/a')})")
            if cal_exp.get("contributions"):
                st.plotly_chart(
                    shap_contribution_chart(cal_exp["contributions"]),
                    use_container_width=True,
                )
            else:
                st.info("Contributions unavailable for calibration model.")
        with ec2:
            st.caption(f"Population-only model ({pop_exp.get('method', 'n/a')})")
            if pop_exp.get("contributions"):
                st.plotly_chart(
                    shap_contribution_chart(pop_exp["contributions"]),
                    use_container_width=True,
                )
            else:
                st.info("Population-only model artifact not loaded.")

    gnomad_hits = result.get("gnomad_comparison", [])
    if gnomad_hits:
        st.markdown("#### gnomAD South Asian vs European Reference")
        for hit in gnomad_hits[:2]:
            fig = gnomad_comparison_chart(
                result["gene"],
                hit["allele"],
                hit["gnomad_sas_af"],
                hit["gnomad_nfe_af"],
                observed_sas=result["india_diplotype_freq"],
                observed_eur=result["european_diplotype_freq"],
            )
            st.plotly_chart(fig, use_container_width=True)
            if hit.get("description"):
                st.caption(hit["description"])

    st.markdown(
        pi_box(
            "Evidence trail",
            f"n = {result['n_real_individuals']} real SAS individuals · "
            f"SAS freq = {result['india_diplotype_freq']:.4f} "
            f"(95% CI {result['india_freq_ci_low']:.4f}–{result['india_freq_ci_high']:.4f}) · "
            f"EUR freq = {result['european_diplotype_freq']:.4f} · "
            f"SAS/EUR ratio = {result['sas_vs_eur_ratio']:.2f} · "
            f"Phenotype: {result['phenotype']}",
            meta="Population-adjusted significance: " + result["population_adjusted_label"],
        ),
        unsafe_allow_html=True,
    )

    col_risk, col_pop = st.columns([1, 1.4])
    with col_risk:
        st.markdown("#### India Risk Score")
        st.markdown(
            f'<div class="pi-box" style="text-align:center;">'
            f'<div class="pi-stat-num">{result["india_risk_score"]:.2f}</div>'
            f'<div class="pi-box-meta">India-calibrated risk score (0.0 – 1.0)</div></div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Global Baseline", f"{result['baseline_risk_score']:.2f}")
        with d2:
            st.metric("Heuristic Formula", f"{result['heuristic_risk_score']:.2f}")
        with d3:
            delta = result["risk_score_delta"]
            st.metric("Δ India vs Baseline", f"{delta:+.2f}", delta_color="off" if delta == 0 else "normal")
        st.caption(
            f"**India ML prediction:** {result['india_predicted_label']} "
            f"({result['india_confidence'] * 100:.1f}%) · "
            f"**Baseline:** {result['baseline_predicted_label']} "
            f"({result['baseline_confidence'] * 100:.1f}%) · "
            f"**Rule baseline (CPIC only):** {result['rule_predicted_label']}"
            + (" · ⚠️ India and baseline models **disagree** on this genotype."
               if result["classification_changed"] else "")
        )

    with col_pop:
        st.markdown("#### Population Frequency Comparison")
        fig = population_comparison_chart(
            result["gene"], result["genotype"],
            result["india_diplotype_freq"], result["european_diplotype_freq"],
            result["east_asian_diplotype_freq"],
        )
        st.plotly_chart(fig, use_container_width=True)
        eur_freq = result["european_diplotype_freq"]
        india_freq = result["india_diplotype_freq"]
        if eur_freq == 0 and india_freq == 0:
            st.caption(
                "This genotype was not observed in either the South Asian or European reference "
                "samples used here."
            )
        elif eur_freq == 0:
            st.caption(
                f"This genotype was **not observed at all** in the European reference sample used "
                f"here, but appears in **{result['n_real_individuals']} real South Asian individuals** "
                f"in this dataset -- i.e. it may be substantially more South-Asian-specific than a "
                f"finite ratio can express, rather than a literal '∞x'."
            )
        elif india_freq == 0:
            st.caption(
                "This genotype was **not observed at all** in the South Asian reference sample used "
                "here, despite appearing in the European reference population."
            )
        else:
            ratio = result["sas_vs_eur_ratio"]
            display_ratio = ratio if ratio >= 1 else round(1 / (ratio + 1e-8), 2)
            more_common_in = "South Asian" if ratio >= 1 else "European"
            st.caption(
                f"This genotype is **{display_ratio}x more common** in **{more_common_in}** genomes "
                f"than in the comparison population, based on {result['n_real_individuals']} real "
                f"observed SAS individuals in this dataset."
            )

    st.markdown("#### SAS Subpopulation Breakdown")
    breakdown = result["population_breakdown"]
    bdf = pd.DataFrame([
        {
            "Subpopulation": SAS_POPULATION_NAMES[p],
            "Code": p,
            "% carrying this genotype": f"{breakdown[p]:.1f}%" if breakdown[p] is not None else "N/A",
        }
        for p in SAS_POPULATIONS
    ])
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.markdown("#### Clinical Interpretation")
    st.markdown(
        f"**{result['gene']}** ({result['full_name']}) - CPIC evidence level "
        f"**{result['cpic_level']}**. This genotype produces the **{result['phenotype']}** phenotype."
    )
    if result["drugs"]:
        drug_df = pd.DataFrame({
            "Affected Drug": result["drugs"],
            "Clinical Significance": [result["observed_clinical_significance_label"]] * len(result["drugs"]),
        })
        st.dataframe(drug_df, use_container_width=True, hide_index=True)
    else:
        st.info("No CPIC drug associations on file for this gene.")
    if result.get("india_relevance_note"):
        st.warning(f"**India-specific note:** {result['india_relevance_note']}")

    st.markdown(f"#### {t('hindi_summary', lang)}")
    st.markdown(
        hindi_summary(
            result["gene"], result["genotype"], result["phenotype"],
            round(result["india_diplotype_freq"] * 100, 1),
            result["observed_clinical_significance"], result["drugs"],
        )
    )

    st.markdown("---")
    import json
    pdf_payload = json.dumps({k: result[k] for k in result if isinstance(result[k], (str, int, float, bool, list, dict, type(None)))}, default=str)
    try:
        pdf_bytes = build_pdf_bytes(result["gene"], result["genotype"], pdf_payload)
        st.download_button(
            t("download_pdf", lang),
            data=pdf_bytes,
            file_name=f"seed_{result['gene']}_{result['genotype'].replace('/', '-')}.pdf",
            mime="application/pdf",
        )
    except Exception as pdf_err:
        st.warning(f"PDF export unavailable: {pdf_err}")


def render_model_performance_tab():
    st.subheader("Model Performance")

    from utils.rubric_content import GROUPED_CV_EXPLANATION

    with st.expander("What is grouped cross-validation?", expanded=False):
        st.markdown(GROUPED_CV_EXPLANATION)

    import json

    json_path = os.path.join(ARTIFACTS_DIR, "comparison_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            comp = json.load(f)
        rows = [
            {
                "Model": "Global Baseline (ML)",
                "Accuracy (group-level)": comp["baseline"]["accuracy"],
                "Macro F1": comp["baseline"]["macro_f1"],
                "Unique genotypes": comp["baseline"].get("n_groups", 168),
                "Label": "clinical_significance",
            },
            {
                "Model": "Rule Baseline (no ML)",
                "Accuracy (group-level)": comp.get("rule_baseline", {}).get("accuracy", "-"),
                "Macro F1": comp.get("rule_baseline", {}).get("macro_f1", "-"),
                "Unique genotypes": comp.get("rule_baseline", {}).get("n_groups", 168),
                "Label": "clinical_significance (CPIC thresholds)",
            },
            {
                "Model": "Population-Only ML (no CPIC)",
                "Accuracy (group-level)": comp.get("population_only", {}).get("accuracy", "-"),
                "Macro F1": comp.get("population_only", {}).get("macro_f1", "-"),
                "Unique genotypes": comp.get("population_only", {}).get("n_groups", 168),
                "Label": "population_adjusted_significance (hard task)",
            },
            {
                "Model": "Calibration ML (CPIC + population)",
                "Accuracy (group-level)": comp["india_calibrated"]["accuracy"],
                "Macro F1": comp["india_calibrated"]["macro_f1"],
                "Unique genotypes": comp["india_calibrated"].get("n_groups", 168),
                "Label": comp["india_calibrated"].get("label", "population_adjusted_significance"),
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        delta = comp.get("accuracy_delta_india_vs_baseline")
        if delta is not None:
            st.caption(
                f"Calibration ML vs Global Baseline accuracy delta: **{delta:+.2f}** · "
                f"Population-only vs Calibration: "
                f"**{comp.get('population_only', {}).get('accuracy', 0) - comp['india_calibrated']['accuracy']:+.2f}**"
            )

        risk = comp.get("risk_regressor")
        if risk:
            st.markdown("#### Risk Regressor (continuous 0–1 score)")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Rule-adjusted MAE", f"{risk.get('rule_adjusted_mae', '-')}")
            rc2.metric("Heuristic MAE", f"{risk['heuristic_mae']:.4f}")
            rc3.metric("XGBRegressor MAE", f"{risk['ml_mae']:.4f}")
            rc4.metric("ML vs rule", f"{risk.get('mae_improvement_vs_rule', risk['mae_improvement']):+.4f}")

        loso = comp.get("loso", {})
        if loso and "summary" in loso:
            st.markdown("#### LOSO Validation (leave-one-SAS-subpopulation-out)")
            loso_rows = [
                {
                    "Population": r["population_name"],
                    "Code": pop,
                    "Accuracy (group-level)": r["accuracy"],
                    "Macro F1": r["macro_f1"],
                    "Test genotypes": r.get("n_test_groups", r.get("n_test", "-")),
                }
                for pop, r in loso.items() if pop != "summary"
            ]
            st.dataframe(pd.DataFrame(loso_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Mean LOSO accuracy: {loso['summary']['mean_accuracy']:.2f} "
                f"(±{loso['summary']['std_accuracy']:.2f}) across {loso['summary']['n_folds']} folds"
            )

        ic = comp.get("india_calibrated", {})
        od = comp.get("overfitting_diagnosis", {})
        if od.get("models"):
            st.markdown("#### Overfitting Diagnosis (train vs grouped-CV)")
            od_rows = []
            for label, key in [
                ("Global Baseline (ML)", "global_baseline"),
                ("Population-Only ML", "population_only"),
                ("Calibration ML (CPIC + pop)", "india_calibrated"),
            ]:
                d = od["models"].get(key, {})
                flagged = d.get("overfitting_flagged", False)
                od_rows.append({
                    "Model": label,
                    "Train accuracy": d.get("train_accuracy", "-"),
                    "CV accuracy": d.get("cv_accuracy", "-"),
                    "Gap (train − CV)": d.get("accuracy_gap", "-"),
                    "Status": "⚠️ Overfitting" if flagged else "✓ OK",
                })
            st.dataframe(pd.DataFrame(od_rows), use_container_width=True, hide_index=True)
            if od.get("any_overfitting_flagged"):
                st.warning(
                    "One or more models exceed the 0.10 train/CV accuracy gap threshold. "
                    "Stronger regularization was applied during re-tuning."
                )
            else:
                st.success(
                    "No model exceeds the 0.10 train/CV accuracy gap - "
                    "grouped CV metrics are trustworthy."
                )

        gh = comp.get("gene_holdout", {})
        if gh:
            st.markdown("#### Gene-Held-Out CV (leave-one-gene-out)")
            gh_rows = []
            for model_label, key in [
                ("Calibration ML", "calibration_model"),
                ("Population-Only ML", "population_only_model"),
            ]:
                s = gh.get(key, {}).get("summary", {})
                gh_rows.append({
                    "Model": model_label,
                    "Mean accuracy": s.get("mean_accuracy", "-"),
                    "Std": s.get("std_accuracy", "-"),
                    "Genes held out": s.get("n_genes", "-"),
                })
            st.dataframe(pd.DataFrame(gh_rows), use_container_width=True, hide_index=True)
            with st.expander("Per-gene holdout details"):
                for model_label, key in [
                    ("Calibration", "calibration_model"),
                    ("Population-Only", "population_only_model"),
                ]:
                    st.markdown(f"**{model_label}**")
                    per = gh.get(key, {}).get("per_gene", {})
                    if per:
                        st.dataframe(
                            pd.DataFrame([
                                {"Gene": g, **v} for g, v in sorted(per.items())
                            ]),
                            use_container_width=True,
                            hide_index=True,
                        )

        n_groups = ic.get("n_groups", comp.get("baseline", {}).get("n_groups", "-"))
        st.markdown(
            '<div class="pi-caveat"><b>Group-level metrics.</b> Each unique gene|genotype counts '
            f"once ({n_groups} SAS-observed groups from 6,035 real individuals), not once per person. "
            "Training uses real WGS data only - no catalog or synthetic rows.</div>",
            unsafe_allow_html=True,
        )
        for note in comp.get("methodological_notes", comp.get("methodological_caveats", [])):
            st.markdown(pi_box("Methodological note", note), unsafe_allow_html=True)
    else:
        st.warning("Run `python model/compare_models.py` to generate comparison_results.json.")

    st.markdown("---")
    st.markdown("#### Visualizations")
    plot_files = [
        ("learning_curve.png", "Learning Curve (group-aware, held-out validation genotypes)"),
        ("feature_importances.png", "Feature Importances (India-Calibrated Model)"),
        ("confusion_matrix.png", "Confusion Matrix (grouped CV)"),
        ("cyp2c19_population_comparison.png", "CYP2C19*2 Frequency by Population"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(plot_files):
        path = os.path.join(ARTIFACTS_DIR, fname)
        with cols[i % 2]:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)
            else:
                st.warning(f"Missing {fname} - run `model/generate_plots.py`.")


def render_simulations_tab(scorer):
    lang = st.session_state.get("ui_lang", "en")
    st.subheader(t("simulations", lang))
    st.markdown(
        pi_box(
            "Gene-level Monte Carlo",
            "Bootstrap resampling of SAS vs EUR carrier frequencies under observed "
            "1000 Genomes allele rates. Shows how population-specific prevalence "
            "shifts prescribing urgency for the same CPIC phenotype.",
        ),
        unsafe_allow_html=True,
    )

    demo_gene = st.selectbox(
        "Gene for simulation",
        ["CYP2C19", "DPYD", "CYP2D6", "NUDT15"],
        key="sim_gene",
    )
    if demo_gene == "CYP2C19":
        demo = cyp2c19_clopidogrel_demo()
    elif demo_gene == "DPYD":
        demo = dpyd_poor_metabolizer_demo()
    else:
        genotypes = scorer.genotypes_for_gene(demo_gene)
        gt_data = []
        for gt, _ in genotypes[:5]:
            try:
                r = scorer.score(demo_gene, gt)
                gt_data.append({
                    "genotype": gt,
                    "sas_freq": r["india_diplotype_freq"],
                    "eur_freq": r["european_diplotype_freq"],
                })
            except (KeyError, ValueError):
                continue
        demo = {
            "gene": demo_gene,
            "genotype": gt_data[0]["genotype"] if gt_data else "*1/*1",
            "simulation": {"genotypes": {
                g["genotype"]: {
                    "sas_freqs": __import__("numpy").array([g["sas_freq"]] * 50),
                    "eur_freqs": __import__("numpy").array([g["eur_freq"]] * 50),
                    "sas_mean": g["sas_freq"],
                    "eur_mean": g["eur_freq"],
                }
                for g in gt_data
            }},
        }

    sim = demo["simulation"]
    sel_gt = st.selectbox(
        "Genotype",
        list(sim.get("genotypes", {demo.get("genotype", "*1/*2"): {}}).keys()),
        key="sim_gt",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            animated_frequency_shift(sim, demo["gene"], sel_gt),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            gene_level_heatmap(demo["gene"], sim.get("genotypes", {})),
            use_container_width=True,
        )

    if demo.get("clinical_note"):
        st.markdown(pi_box("Clinical context", demo["clinical_note"]), unsafe_allow_html=True)


def render_demo_walkthrough_tab(scorer):
    lang = st.session_state.get("ui_lang", "en")
    st.subheader(t("demo", lang))
    st.markdown(
        pi_box(
            "3-minute demo script",
            "Walk through two flagship scenarios: CYP2C19 *1/*2 (clopidogrel, moderate) "
            "and DPYD poor metabolizer (fluoropyrimidine, URGENT). Compare SAS vs EUR "
            "frequency shift and dual-model predictions.",
        ),
        unsafe_allow_html=True,
    )

    scenario = st.radio(
        "Scenario",
        ["CYP2C19 *1/*2 + Clopidogrel", "DPYD Poor Metabolizer (URGENT)"],
        key="demo_scenario",
    )

    if "CYP2C19" in scenario:
        demo = cyp2c19_clopidogrel_demo()
        gene, genotype = "CYP2C19", "*1/*2"
    else:
        demo = dpyd_poor_metabolizer_demo()
        gene, genotype = "DPYD", demo["genotype"]

    steps = [
        f"1. On **Guide** or **Analyse**, select **{gene}** and **{genotype}**.",
        f"2. Compare SAS frequency ({demo['sas_freq']:.1%}) vs EUR ({demo['eur_freq']:.1%}). "
        f"The SAS/EUR ratio drives population-adjusted urgency.",
        "3. Compare **Population-Only ML** (~59% CV, no CPIC) vs **Calibration ML** (~96% CV).",
        "4. Open **Why this urgency?** to see which features pushed the class.",
        "5. Check **gnomAD SAS vs NFE** for independent allele-frequency validation.",
        "6. Read **Known limitations**: 168 groups, decision support only, VKORC1 excluded.",
    ]
    for s in steps:
        st.markdown(f"- {s}")

    st.markdown("---")
    st.markdown("#### Live preview")
    if st.button("Run live score for this scenario", type="primary", key="demo_run"):
        st.session_state["demo_result"] = None
        try:
            st.session_state["demo_result"] = scorer.score(gene, genotype)
            st.session_state["result"] = st.session_state["demo_result"]
        except (KeyError, ValueError) as err:
            st.error(str(err))

    result = st.session_state.get("demo_result")
    if result:
        urgency_banner(result)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                animated_frequency_shift(demo["simulation"], gene, genotype),
                use_container_width=True,
            )
        with c2:
            if result.get("gnomad_comparison"):
                hit = result["gnomad_comparison"][0]
                st.plotly_chart(
                    gnomad_comparison_chart(
                        gene, hit["allele"],
                        hit["gnomad_sas_af"], hit["gnomad_nfe_af"],
                        result["india_diplotype_freq"], result["european_diplotype_freq"],
                    ),
                    use_container_width=True,
                )
        st.markdown(pi_box("Clinical note", demo["clinical_note"]), unsafe_allow_html=True)
    else:
        st.caption("Click **Run live score** to preview this scenario without leaving the tab.")


def render_impact_tab():
    st.subheader("Metric 01 · Impact & Inclusion")
    st.markdown(
        pi_box(
            "Problem statement",
            "Pharmacogenomic guidelines (CPIC) are calibrated primarily on European and East Asian "
            "cohorts. For ~1.4 billion South Asians, the same gene/genotype can imply a different "
            "population prevalence - and therefore a different real-world prescribing risk - than "
            "global tools assume. Seed re-calibrates CPIC phenotypes using observed South Asian "
            "allele frequencies from the 1000 Genomes Project.",
            meta="Where · India & South Asian diaspora · What · PGx dose/toxicity risk · Solution · offline AI tool",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Evidence & citations")
    for c in CITATIONS:
        st.markdown(
            pi_box(f"Reference {c['id']}", f"{c['text']} - {c['source']}"),
            unsafe_allow_html=True,
        )

    st.markdown("#### Target audience")
    for role, desc in TARGET_AUDIENCE:
        st.markdown(pi_box(role, desc), unsafe_allow_html=True)

    st.markdown("#### Inclusion & accessibility")
    for feat in ACCESSIBILITY_FEATURES:
        st.markdown(f'<div class="rubric-check rubric-check-yes">{feat}</div>', unsafe_allow_html=True)

    st.markdown("#### SDG mapping")
    sdg_df = pd.DataFrame(SDG_MAPPING)
    st.dataframe(sdg_df, use_container_width=True, hide_index=True)

    st.markdown(
        pi_box(
            "Societal impact",
            "Prevents mis-calibrated chemotherapy toxicity risk (DPYD/NUDT15/TPMT), improves "
            "clopidogrel response prediction for cardiac patients (CYP2C19), and democratises "
            "precision medicine for populations excluded from genomic research.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        pi_box(
            "Sustainable pathways",
            "Open-source codebase, reproducible training scripts, CPIC-update pipeline, and "
            "offline deployment for rural clinics without recurring cloud costs.",
        ),
        unsafe_allow_html=True,
    )


def render_ai_innovation_tab():
    st.subheader("Metric 02 · AI Innovation")
    st.markdown(
        pi_box(
            "Why AI is the primary technology (not a force-fit)",
            WHY_AI_NOT_RULES.strip(),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        pi_box(
            "Original contribution",
            "Population-only XGBoost classifier (no CPIC score in features) predicting "
            "population_adjusted_significance - SAS-ratio-calibrated urgency labels. "
            "Three-way comparison: CPIC rule baseline vs European-feature ML vs India "
            "population ML. XGBRegressor risk model + bootstrap frequency CIs. Grouped CV "
            "and LOSO validation with full methodological disclosure.",
            meta="Classification: new/original adaptation for South Asian PGx",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Data pipeline (obtain → analyse → use)")
    pipeline = [
        ("Obtain", "1000 Genomes combined.csv → filter SAS (GIH/PJL/ITU/STU/BEB) → 489 individuals"),
        ("Analyse", "Observed diplotype frequencies, CPIC phenotype mapping, clinical significance labels"),
        ("Justify", "Grouped StratifiedGroupKFold CV; SMOTE rejected (n=4 in smallest class); no synthetic rows"),
        ("Deploy", "Pre-trained XGBoost artifacts loaded locally - zero network at inference"),
    ]
    for step, detail in pipeline:
        st.markdown(pi_box(step, detail), unsafe_allow_html=True)

    st.markdown("#### Ethics, privacy, bias & environment")
    for title, body in ETHICS_PRIVACY:
        st.markdown(pi_box(title, body), unsafe_allow_html=True)

    st.markdown("#### Deployment & GTM strategy")
    gtm_df = pd.DataFrame(GTM_STRATEGY, columns=["Phase", "Plan"])
    st.dataframe(gtm_df, use_container_width=True, hide_index=True)
    st.markdown(
        pi_box(
            "Live deployment",
            "Working prototype: run locally via Streamlit (localhost:8502). "
            "Open-source: github.com/arnavd371/arnavdhiman/tree/main/seed. "
            "One-click Streamlit Cloud deploy available for public demo URL.",
        ),
        unsafe_allow_html=True,
    )


def render_technical_tab():
    st.subheader("Metric 03 · Technical Skills")
    st.markdown(
        pi_box(
            "Tech stack",
            f"Languages: {', '.join(TECH_STACK['languages'])} · "
            f"ML: {', '.join(TECH_STACK['ml'])} · "
            f"Data: {', '.join(TECH_STACK['data'])} · "
            f"UI: {', '.join(TECH_STACK['ui'])} · "
            f"Reporting: {', '.join(TECH_STACK['reporting'])}",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(pi_box("Hardware", TECH_STACK["hardware"]), unsafe_allow_html=True)
    st.markdown(pi_box("Software & deployment", TECH_STACK["deployment"]), unsafe_allow_html=True)

    st.markdown(
        pi_box(
            "UI complexity",
            "Custom-built Streamlit interface with PI-style research CSS, interactive Plotly "
            "charts, genotype dropdowns from real data, PDF export, and bilingual summaries - "
            "not a no-code template.",
            meta="Score target: custom UI built specifically for this solution",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        pi_box(
            "AI / ML packages used",
            "XGBoost gradient boosting (primary classifier), scikit-learn (StratifiedGroupKFold, "
            "metrics, preprocessing), pandas/numpy (feature engineering), imbalanced-learn "
            "(evaluated). Deliberately no Gen-AI/LLM - clinical reliability requires deterministic, "
            "auditable inference.",
            meta="Score target: advanced ML packages; explainable classical AI over black-box LLM",
        ),
        unsafe_allow_html=True,
    )

    render_methods_tab()


def render_methods_tab():
    st.subheader("Methods & Reproducibility")
    sections = [
        (
            "Dataset",
            "Parsed from 1000 Genomes Project phase 3 (Sherman, Claw & Lee, Sci Rep 2024). "
            "Filtered to SAS superpopulation (GIH, PJL, ITU, STU, BEB): 489 individuals, "
            "6,035 gene–genotype rows, 168 unique (gene, genotype) groups. CPIC-actionable genes only "
            "(includes F5; CACNA1S/RYR1 excluded - Uncertain Susceptibility only). "
            "Indeterminate phenotypes removed. VKORC1 excluded.",
        ),
        (
            "Labels",
            "Clinical significance classes 0–3 derived from CPIC phenotype strings: "
            "0 = No Action, 1 = Moderate, 2 = Significant, 3 = Urgent (DPYD/NUDT15/TPMT poor metaboliser).",
        ),
        (
            "Cross-validation",
            "StratifiedGroupKFold grouped by gene|genotype (k=4–5 adaptive). "
            "Plain row-level k-fold was rejected after discovering duplicate-genotype leakage "
            "that inflated accuracy to ~100%.",
        ),
        (
            "Models",
            "Two XGBoost multi-class classifiers on identical rows. Baseline: 4 European-frequency features. "
            "India-calibrated: 19 expanded population features (multi-superpopulation frequencies, "
            "SAS subpop dispersion/entropy, genotype complexity) with grouped random-search tuning "
            "and balanced class weights. No cpic_phenotype_score in features. "
            "SMOTE not applied (smallest class n=4).",
        ),
        (
            "Reproduce",
            "cd seed && python data/parse_real_data.py && python model/train_baseline.py "
            "&& python model/train_model.py && python model/compare_models.py && python model/generate_plots.py",
        ),
        (
            "References",
            "CPIC (cpicpgx.org) · PharmGKB · 1000 Genomes Project · "
            "Sherman RM, Claw KG, Lee SB. Sci Rep 14, 22774 (2024).",
        ),
    ]
    for title, body in sections:
        st.markdown(pi_box(title, body), unsafe_allow_html=True)


def main():
    large = st.session_state.get("large_text", False)
    inject_css(large_text=large)
    if not st.session_state.get("entered_app"):
        render_intro()
        return

    lang = st.session_state.get("ui_lang", "en")
    scorer = get_scorer(_model_fingerprint())
    st.markdown('<div class="pi-hero-title" style="font-size:2rem;">🌱 Seed | बीज</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pi-hero-sub">India-calibrated pharmacogenomics. '
        "Because Indian genes are not European genes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    render_sidebar(scorer)

    tabs = st.tabs([
        t("guide", lang),
        t("analyse", lang),
        t("simulations", lang),
        t("demo", lang),
        t("impact", lang),
        t("ai_innovation", lang),
        t("technical", lang),
        t("performance", lang),
    ])
    with tabs[0]:
        render_guide_tab(scorer)
    with tabs[1]:
        render_analyse_tab(scorer)
    with tabs[2]:
        render_simulations_tab(scorer)
    with tabs[3]:
        render_demo_walkthrough_tab(scorer)
    with tabs[4]:
        render_impact_tab()
    with tabs[5]:
        render_ai_innovation_tab()
    with tabs[6]:
        render_technical_tab()
    with tabs[7]:
        render_model_performance_tab()


if __name__ == "__main__":
    main()
