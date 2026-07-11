UI_STRINGS = {
    "en": {
        "guide": "Guide",
        "analyse": "Analyse",
        "impact": "Impact & Inclusion",
        "ai_innovation": "AI Innovation",
        "technical": "Technical Skills",
        "performance": "Model Performance",
        "simulations": "Simulations",
        "demo": "Demo Walkthrough",
        "select_gene": "Gene",
        "select_genotype": "Genotype (observed in real SAS data)",
        "analyse_btn": "Analyse Genotype",
        "download_pdf": "Download PDF Clinical Summary",
        "hindi_summary": "Hindi Summary",
        "language": "Interface language",
    },
    "hi": {
        "guide": "मार्गदर्शिका",
        "analyse": "विश्लेषण",
        "impact": "प्रभाव और समावेशन",
        "ai_innovation": "AI नवाचार",
        "technical": "तकनीकी कौशल",
        "performance": "मॉडल प्रदर्शन",
        "simulations": "सिमुलेशन",
        "demo": "डेमो वॉकथ्रू",
        "select_gene": "जीन",
        "select_genotype": "जीनोटाइप (वास्तविक SAS डेटा)",
        "analyse_btn": "जीनोटाइप का विश्लेषण करें",
        "download_pdf": "PDF रिपोर्ट डाउनलोड करें",
        "hindi_summary": "हिंदी सारांश",
        "language": "इंटरफ़ेस भाषा",
    },
}


def t(key: str, lang: str = "en") -> str:
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, key)
