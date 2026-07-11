import io

from fpdf import FPDF

TEAL = (0, 180, 154)
DARK = (20, 24, 38)
GREY = (90, 98, 115)


class SeedPDF(FPDF):
    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, self.w, 22, style="F")
        # NOTE: fpdf2's built-in core fonts (Helvetica) only support Latin-1,
        # so Devanagari text ("बीज") cannot be rendered here without
        # bundling a full Unicode TTF font. The PDF header therefore uses
        # the ASCII-only "Seed" -- the Hindi summary itself is shown
        # in the Streamlit app, not duplicated in this PDF.
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*TEAL)
        self.set_xy(10, 6)
        self.cell(0, 10, "Seed", ln=1)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 15)
        self.cell(0, 6, "India-Calibrated Pharmacogenomic Report", ln=1)
        self.set_y(26)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.cell(0, 10, f"Page {self.page_no()} - Clinical decision support only. Not a diagnostic device.",
                  align="C")


def _section_title(pdf, text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 120, 100)
    pdf.cell(0, 8, text, ln=1)
    pdf.set_text_color(30, 30, 30)


def _kv_row(pdf, key, value):
    # multi_cell() does not always reset x back to the left margin after a
    # single-line render, so each row must explicitly re-anchor there --
    # otherwise x drifts right across successive rows until there is no
    # width left to render text (a real bug caught during verification).
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 6, key, border=0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, str(value))


def generate_pdf_report(result, chart_image_path=None):
    pdf = SeedPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _section_title(pdf, "Genotype Summary")
    _kv_row(pdf, "Gene:", f"{result['gene']} ({result.get('full_name', result['gene'])})")
    _kv_row(pdf, "Genotype:", result["genotype"])
    _kv_row(pdf, "Phenotype:", result["phenotype"])
    _kv_row(pdf, "Observed clinical significance:", result["observed_clinical_significance_label"])
    pdf.ln(3)

    _section_title(pdf, "India Risk Score")
    _kv_row(pdf, "India-calibrated risk score:", f"{result['india_risk_score']:.3f} / 1.0")
    _kv_row(pdf, "Global baseline risk score:", f"{result['baseline_risk_score']:.3f} / 1.0")
    delta = result["risk_score_delta"]
    _kv_row(pdf, "Delta from India calibration:", f"{'+' if delta >= 0 else ''}{delta:.3f}")
    _kv_row(pdf, "Model predicted class (India):", result["india_predicted_label"])
    _kv_row(pdf, "Model confidence:", f"{result['india_confidence'] * 100:.1f}%")
    pdf.ln(3)

    _section_title(pdf, "Population Frequency Comparison")
    _kv_row(pdf, "South Asian (SAS) diplotype freq.:", f"{result['india_diplotype_freq'] * 100:.2f}%")
    _kv_row(pdf, "European (EUR) diplotype freq.:", f"{result['european_diplotype_freq'] * 100:.2f}%")
    _kv_row(pdf, "SAS vs EUR ratio:", f"{result['sas_vs_eur_ratio']:.2f}x")
    _kv_row(pdf, "Real individuals observed (SAS):", str(result["n_real_individuals"]))
    pdf.ln(2)

    if chart_image_path:
        try:
            pdf.image(chart_image_path, x=15, w=pdf.w - 30)
            pdf.ln(3)
        except Exception:
            pass

    _section_title(pdf, "Affected Medications")
    drugs = result.get("drugs", [])
    if drugs:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 245, 242)
        pdf.cell(70, 7, "Drug", border=1, fill=True)
        pdf.cell(50, 7, "CPIC Level", border=1, fill=True)
        pdf.cell(0, 7, "Phenotype relevance", border=1, fill=True, ln=1)
        pdf.set_font("Helvetica", "", 9)
        for drug in drugs:
            pdf.cell(70, 7, drug, border=1)
            pdf.cell(50, 7, str(result.get("cpic_level", "N/A")), border=1)
            pdf.cell(0, 7, result["observed_clinical_significance_label"], border=1, ln=1)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No CPIC drug associations on file for this gene.", ln=1)
    pdf.ln(3)

    if result.get("india_relevance_note"):
        _section_title(pdf, "India-Specific Clinical Note")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5.5, result["india_relevance_note"])
        pdf.ln(2)

    _section_title(pdf, "Data Provenance")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5.5,
                    f"Source: {result.get('data_provenance', 'N/A')}\n"
                    f"Reference: {result.get('reference', 'N/A')}\n"
                    f"Zero synthetic training records were used to build this model.")
    pdf.ln(3)

    _section_title(pdf, "Disclaimer")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        0, 5.5,
        "Seed is a decision-support research prototype, not a diagnostic device. "
        "It has not been clinically validated in Indian hospitals. All outputs require "
        "review and interpretation by a qualified healthcare professional before any "
        "prescribing decision is made or changed."
    )

    return bytes(pdf.output())
