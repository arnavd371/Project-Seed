from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VCFVariant:
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    genotype: Optional[str] = None
    genotype_zygosity: Optional[str] = None  # "het", "hom_alt", "hom_ref"


@dataclass
class VCFParseResult:
    variants: List[VCFVariant] = field(default_factory=list)
    sample_name: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _classify_genotype(gt_field):
    if not gt_field or gt_field in (".", "./.", ".|."):
        return None, None

    alleles = gt_field.replace("|", "/").split("/")
    if len(alleles) < 2:
        return gt_field, None

    a, b = alleles[0], alleles[1]
    if a == "0" and b == "0":
        return gt_field, "hom_ref"
    if a == b and a != "0":
        return gt_field, "hom_alt"
    if a != b:
        return gt_field, "het"
    return gt_field, None


def parse_vcf(file_obj_or_path):
    if hasattr(file_obj_or_path, "read"):
        raw = file_obj_or_path.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        lines = text.splitlines()
    else:
        with open(file_obj_or_path, "r") as f:
            lines = f.read().splitlines()

    result = VCFParseResult()

    for line in lines:
        if not line.strip():
            continue

        if line.startswith("##"):
            continue

        if line.startswith("#CHROM"):
            header_cols = line.lstrip("#").split("\t")
            if len(header_cols) > 9:
                result.sample_name = header_cols[9]
            continue

        cols = line.split("\t")
        if len(cols) < 5:
            result.warnings.append(f"Skipped malformed line (fewer than 5 columns): {line[:60]}")
            continue

        chrom, pos, variant_id, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]

        genotype, zygosity = None, None
        if len(cols) >= 10:
            fmt_fields = cols[8].split(":")
            sample_fields = cols[9].split(":")
            if "GT" in fmt_fields:
                gt_index = fmt_fields.index("GT")
                if gt_index < len(sample_fields):
                    genotype, zygosity = _classify_genotype(sample_fields[gt_index])

        try:
            pos_int = int(pos)
        except ValueError:
            result.warnings.append(f"Skipped line with non-numeric POS: {line[:60]}")
            continue

        result.variants.append(VCFVariant(
            chrom=chrom,
            pos=pos_int,
            variant_id=variant_id if variant_id != "." else "",
            ref=ref,
            alt=alt,
            genotype=genotype,
            genotype_zygosity=zygosity,
        ))

    if not result.variants:
        result.warnings.append("No variant records found in this VCF.")

    return result


def variants_to_rows(parse_result):
    return [
        {
            "CHROM": v.chrom,
            "POS": v.pos,
            "ID": v.variant_id or "-",
            "REF": v.ref,
            "ALT": v.alt,
            "Genotype": v.genotype or "-",
            "Zygosity": v.genotype_zygosity or "-",
        }
        for v in parse_result.variants
    ]
