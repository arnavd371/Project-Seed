"""
Simplified demo star-allele caller - NOT clinical-grade.

Matches defining rsIDs from gene_drug_map.json against uploaded VCF variants.
Real callers (PharmCAT/Stargazer) need full PharmVar haplotype tables, phasing,
and structural variant handling.
"""

import json
import os

SEED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GENE_DRUG_MAP_PATH = os.path.join(SEED_ROOT, "data", "gene_drug_map.json")


def load_gene_drug_map(path=DEFAULT_GENE_DRUG_MAP_PATH):
    with open(path) as f:
        return json.load(f)


def _rsids_for_gene(gene_info):
    mapping = {}
    for allele, info in gene_info.get("star_alleles", {}).items():
        rsid = info.get("rsid", "")
        # Some entries combine multiple rsIDs (e.g. TPMT*3A); only the first
        # is usable for this simplified single-rsID lookup.
        first_rsid = rsid.split("+")[0].strip()
        if first_rsid.startswith("rs"):
            mapping[first_rsid] = allele
    return mapping


def call_diplotypes_from_variants(vcf_variants, gene_drug_map=None):
    if gene_drug_map is None:
        gene_drug_map = load_gene_drug_map()

    variants_by_id = {v.variant_id: v for v in vcf_variants if v.variant_id}

    results = {}
    for gene, gene_info in gene_drug_map.get("genes", {}).items():
        rsid_to_allele = _rsids_for_gene(gene_info)
        matches = []
        for rsid, allele in rsid_to_allele.items():
            if rsid in variants_by_id:
                matches.append((allele, variants_by_id[rsid]))

        if not matches:
            results[gene] = {
                "diplotype": "*1/*1",
                "matched_variants": [],
                "note": "No known defining variants for this gene were found in the uploaded VCF; called as wild-type (*1/*1) by default.",
            }
            continue

        # Simplification: take the match with the most severe zygosity signal.
        allele, variant = matches[0]
        if variant.genotype_zygosity == "hom_alt":
            diplotype = f"{allele}/{allele}"
        else:
            diplotype = f"*1/{allele}"

        results[gene] = {
            "diplotype": diplotype,
            "matched_variants": [{"rsid": variant.variant_id, "allele": allele, "zygosity": variant.genotype_zygosity} for allele, variant in matches],
            "note": "Called by simplified single-rsID demo lookup, not a clinical-grade caller.",
        }

    return results
