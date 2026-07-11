PHENOTYPE_HINDI = {
    "Normal Metabolizer": "सामान्य मेटाबोलाइज़र (Normal Metabolizer)",
    "Normal Function": "सामान्य कार्य (Normal Function)",
    "Favorable Response": "अनुकूल प्रतिक्रिया (Favorable Response)",
    "Increased Function": "बढ़ा हुआ कार्य (Increased Function)",
    "Intermediate Metabolizer": "मध्यवर्ती मेटाबोलाइज़र (Intermediate Metabolizer)",
    "Intermediate Function": "मध्यवर्ती कार्य (Intermediate Function)",
    "Decreased Function": "घटा हुआ कार्य (Decreased Function)",
    "Rapid Metabolizer": "तेज़ मेटाबोलाइज़र (Rapid Metabolizer)",
    "Ultrarapid Metabolizer": "अत्यंत तेज़ मेटाबोलाइज़र (Ultrarapid Metabolizer)",
    "Poor Metabolizer": "कमज़ोर मेटाबोलाइज़र (Poor Metabolizer)",
    "Poor Function": "कमज़ोर कार्य (Poor Function)",
    "Unfavorable Response": "प्रतिकूल प्रतिक्रिया (Unfavorable Response)",
    "Possible Increased Function": "संभावित रूप से बढ़ा हुआ कार्य",
}

SIGNIFICANCE_HINDI = {
    0: "कोई तत्काल कार्रवाई आवश्यक नहीं (No Action Required)",
    1: "मध्यम नैदानिक महत्व - दवा लेते समय जीनोटाइप पर विचार करें (Moderate)",
    2: "महत्वपूर्ण - खुराक में बदलाव पर विचार करें (Significant)",
    3: "अत्यावश्यक - उपचार न किए जाने पर जीवन के लिए जोखिम हो सकता है (URGENT)",
}


def hindi_summary(gene, genotype, phenotype, india_pct, significance_class, drugs):
    phenotype_hindi = PHENOTYPE_HINDI.get(phenotype, phenotype)
    significance_hindi = SIGNIFICANCE_HINDI.get(significance_class, "अज्ञात")
    drugs_str = "، ".join(drugs) if drugs else "कोई नहीं"

    lines = [
        f"जीन **{gene}** के लिए आपका जीनोटाइप **{genotype}** है।",
        f"फेनोटाइप: {phenotype_hindi}।",
        f"भारतीय (दक्षिण एशियाई) जनसंख्या में यह लगभग **{india_pct}%** लोगों में पाया जाता है।",
        f"नैदानिक महत्व: {significance_hindi}।",
        f"प्रभावित दवाएं: {drugs_str}।",
        "",
        "**अस्वीकरण:** यह रिपोर्ट केवल नैदानिक सहायता के लिए है, निदान उपकरण नहीं है। "
        "किसी भी दवा से संबंधित निर्णय लेने से पहले कृपया एक योग्य चिकित्सक से सलाह लें।",
    ]
    return "\n\n".join(lines)
