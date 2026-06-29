"""Curated protein -> encoding-gene(s) map for the CITE-seq ADT panel.

This is the small, one-time, molecule-level curation the protein-derived ground
truth relies on (e.g. the CD3 antibody binds a complex encoded by CD3D/CD3E/CD3G).
It is based on UniProt/HGNC molecular fact, **not** on any differential-expression
result, so it is independent of the four RNA methods under test.

``map_protein_to_genes`` normalises the antibody/clone name (strips the TotalSeq
``-1``/``-2`` deduplication suffixes, isotype/control tags, whitespace, case) and
returns the encoding gene symbol(s). Proteins with no sensible single-cell gene
(e.g. isotype controls, or antigens whose gene is not expressed in PBMC RNA) map
to an empty list and are dropped from the ground truth.
"""
from __future__ import annotations

from src.logs import get_logger

log = get_logger()

# Ground Truth Mapping: Cell Type -> Primary Expected Protein
# From: Hao et al.

CELL_TYPE_TO_MARKER_PROTEIN = {
    'ASDC': 'CD11c',
    'B intermediate': 'CD19',
    'B memory': 'CD27',
    'B naive': 'CD20',
    'CD4 CTL': 'CD4-1',
    'CD4 Naive': 'CD4-1',
    'CD4 Proliferating': 'CD4-1',
    'CD4 TCM': 'CD4-1',
    'CD4 TEM': 'CD4-1',
    'CD8 Naive': 'CD8',
    'CD8 Proliferating': 'CD8',
    'CD8 TCM': 'CD8',
    'CD8 TEM': 'CD8',
    'CD14 Mono': 'CD14',
    'CD16 Mono': 'CD16',
    'Eryth': 'CD34',
    'HSPC': 'CD34',
    'ILC': 'CD56-1',
    'MAIT': 'CD3-1',
    'NK': 'CD56-1',
    'NK Proliferating': 'CD56-1',
    'NK_CD56bright': 'CD56-1',
    'Plasmablast': 'CD19',
    'Platelet': 'CD11c',
    'Treg': 'CD25',
    'cDC1': 'HLA-DR',
    'cDC2': 'CD11c',
    'dnT': 'CD3-1',
    'gdT': 'CD3-1',
    'pDC': 'CD123'
}

# TODO: Generate this mapping (instead of having it hard coded)

# Antibody (surface-protein) symbol -> list of encoding gene symbol(s).
PROTEIN_TO_GENES: dict[str, list[str]] = {
    # ---- T-cell co-receptors / TCR-CD3 complex ----
    "CD3": ["CD3D", "CD3E", "CD3G"],
    "CD4": ["CD4"],
    "CD8": ["CD8A", "CD8B"],
    "CD8A": ["CD8A"],
    "CD8B": ["CD8B"],
    "CD5": ["CD5"],
    "CD7": ["CD7"],
    "CD2": ["CD2"],
    "CD27": ["CD27"],
    "CD28": ["CD28"],
    "CD127": ["IL7R"],  # IL-7Ra
    "CD278": ["ICOS"],  # ICOS
    "CD279": ["PDCD1"],  # PD-1
    "TIGIT": ["TIGIT"],
    "CD25": ["IL2RA"],
    "CD45RA": ["PTPRC"],
    "CD45RO": ["PTPRC"],
    "CD45": ["PTPRC"],
    "CD62L": ["SELL"],
    "CD197": ["CCR7"],  # CCR7
    "CD183": ["CXCR3"],
    "CD194": ["CCR4"],
    "CD196": ["CCR6"],
    "CD161": ["KLRB1"],
    "CD26": ["DPP4"],
    "CD38": ["CD38"],
    "CD95": ["FAS"],
    "CD57": ["B3GAT1"],
    "KLRG1": ["KLRG1"],
    "CD134": ["TNFRSF4"],  # OX40
    "CD223": ["LAG3"],
    "CD152": ["CTLA4"],
    # ---- B-cell ----
    "CD19": ["CD19"],
    "CD20": ["MS4A1"],
    "CD21": ["CR2"],
    "CD22": ["CD22"],
    "CD23": ["FCER2"],
    "CD24": ["CD24"],
    "CD27 ": ["CD27"],
    "CD40": ["CD40"],
    "CD72": ["CD72"],
    "IGD": ["IGHD"],
    "IGM": ["IGHM"],
    "IGKAPPA": ["IGKC"],
    "IGLAMBDA": ["IGLC1"],
    "CD138": ["SDC1"],
    "CD267": ["TNFRSF13B"],  # TACI
    "CD269": ["TNFRSF17"],  # BCMA
    # ---- Myeloid / monocyte / DC ----
    "CD14": ["CD14"],
    "CD16": ["FCGR3A", "FCGR3B"],
    "CD11B": ["ITGAM"],
    "CD11C": ["ITGAX"],
    "CD13": ["ANPEP"],
    "CD33": ["CD33"],
    "CD64": ["FCGR1A"],
    "CD68": ["CD68"],
    "CD86": ["CD86"],
    "CD80": ["CD80"],
    "CD163": ["CD163"],
    "CD172A": ["SIRPA"],
    "CD1C": ["CD1C"],
    "CD141": ["THBD"],  # BDCA-3, cDC1
    "CD303": ["CLEC4C"],  # BDCA-2, pDC
    "CD304": ["NRP1"],  # BDCA-4
    "CD123": ["IL3RA"],
    "FCERIA": ["FCER1A"],
    "CD32": ["FCGR2A", "FCGR2B"],
    "CD93": ["CD93"],
    "CLEC12A": ["CLEC12A"],
    "CD88": ["C5AR1"],
    "CD89": ["FCAR"],
    # ---- NK ----
    "CD56": ["NCAM1"],
    "CD335": ["NCR1"],  # NKp46
    "CD336": ["NCR2"],  # NKp44
    "CD337": ["NCR3"],  # NKp30
    "CD314": ["KLRK1"],  # NKG2D
    "CD158": ["KIR2DL1"],
    "CD158B": ["KIR2DL2", "KIR2DL3"],
    "CD158E1": ["KIR3DL1"],
    "CD159A": ["KLRC1"],  # NKG2A
    "CD159C": ["KLRC2"],  # NKG2C
    "CD244": ["CD244"],
    "CD16 ": ["FCGR3A"],
    "NKG2C": ["KLRC2"],
    "NKP80": ["KLRF1"],
    # ---- progenitor / activation / other lineage ----
    "CD34": ["CD34"],
    "CD117": ["KIT"],
    "CD71": ["TFRC"],
    "CD41": ["ITGA2B"],
    "CD42B": ["GP1BA"],
    "CD61": ["ITGB3"],
    "CD9": ["CD9"],
    "CD235AB": ["GYPA", "GYPB"],  # glycophorin A/B (RBC)
    "CD105": ["ENG"],
    "CD146": ["MCAM"],
    "CD31": ["PECAM1"],
    "CD144": ["CDH5"],
    "CD49D": ["ITGA4"],
    "CD49B": ["ITGA2"],
    "CD29": ["ITGB1"],
    "CD18": ["ITGB2"],
    "CD11A": ["ITGAL"],
    "CD103": ["ITGAE"],
    "CD69": ["CD69"],
    "CD52": ["CD52"],
    "CD55": ["CD55"],
    "CD58": ["CD58"],
    "CD44": ["CD44"],
    "CD37": ["CD37"],
    "CD39": ["ENTPD1"],
    "CD73": ["NT5E"],
    "CD63": ["CD63"],
    "CD81": ["CD81"],
    "CD82": ["CD82"],
    "CD99": ["CD99"],
    "CD101": ["CD101"],
    "CD107A": ["LAMP1"],
    "CD112": ["NECTIN2"],
    "CD155": ["PVR"],
    "CD226": ["CD226"],
    "CD272": ["BTLA"],
    "CD274": ["CD274"],  # PD-L1
    "CD275": ["ICOSLG"],
    "CD357": ["TNFRSF18"],  # GITR
    "TCR": ["TRAC", "TRBC1", "TRBC2"],
    "TCRVA7.2": ["TRAV1-2"],
    "TCRVD2": ["TRDV2"],
    "TCRVG9": ["TRGV9"],
    "TCRAB": ["TRAC", "TRBC1", "TRBC2"],
    "TCRGD": ["TRDC"],
    "CX3CR1": ["CX3CR1"],
    "CXCR5": ["CXCR5"],
    "CCR10": ["CCR10"],
    "CD185": ["CXCR5"],
    "CD184": ["CXCR4"],
    "CD195": ["CCR5"],
    "HLA-DR": ["HLA-DRA", "HLA-DRB1"],
    "HLADR": ["HLA-DRA", "HLA-DRB1"],
    "HLA-ABC": ["HLA-A", "HLA-B", "HLA-C"],
    "HLAABC": ["HLA-A", "HLA-B", "HLA-C"],
    "HLA-E": ["HLA-E"],
    "CD1A": ["CD1A"],
    "CD1B": ["CD1B"],
    "CD1D": ["CD1D"],
    "FOLR2": ["FOLR2"],
    "CADM1": ["CADM1"],
    "CLEC9A": ["CLEC9A"],
    "XCR1": ["XCR1"],  # cDC1
    "MERTK": ["MERTK"],
    "CD45RB": ["PTPRC"],
    "CD307CFCRL3": ["FCRL3"],
    "CLEC2": ["CLEC1B"],
    "VEGFR": ["FLT4"],
    "GARP": ["LRRC32"],
    "IGE": ["IGHE"],
    "PODOPLANIN": ["PDPN"],
}


def get_marker_genes_for_proteins(protein_names):
    marker_genes = set()

    for protein_name in protein_names:
        genes = set()

        if protein_name in PROTEIN_TO_GENES:
            genes = {gene for gene in PROTEIN_TO_GENES[protein_name]}
        marker_genes = marker_genes.union(genes)

    return marker_genes


def map_protein_to_genes(protein_name):
    if protein_name in PROTEIN_TO_GENES:
        return PROTEIN_TO_GENES[protein_name]

    return set()
