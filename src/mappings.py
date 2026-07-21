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

import logging
import re

log = logging.getLogger(__file__)

# antigen names that have no informative PBMC RNA gene -> dropped from D_c
_CONTROL_TOKENS = (
    "isotype", "control", "mouse igg", "rat igg", "igg1", "igg2",
    "hashtag", "totalseq", "adt", "unmapped",
)


def _normalise(name: str) -> str:
    """Strip TotalSeq dedup suffixes / tags and upper-case for dict lookup.

    ADT feature names carry clone/dedup suffixes (e.g. ``CD3-1``, ``CD4-1``,
    ``CD56-1``) that are absent from the curated ``PROTEIN_TO_GENES`` keys; without
    this normalisation almost nothing maps and the ground truth is empty.
    """
    s = str(name).strip()
    s = re.sub(r"[-_.]\d+$", "", s)                 # trailing '-1', '.2', ...
    s = s.replace("_", "").replace(" ", "").replace("/", "")
    return s.upper()


def is_control(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in _CONTROL_TOKENS)

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
PROTEIN_TO_GENES: dict[str, list[str]] = {'CD39': ['ENTPD1', 'ENTPD6', 'ENTPD2', 'ENTPD3', 'ENTPD5'],
 'CD107a': ['LAMP1'],
 'CD62P': ['SELP'],
 'CD30': ['TNFRSF8', 'TNFSF8'],
 'CD31': ['PECAM1'],
 'CD34': ['CD34'],
 'CD35': ['CR1'],
 'CD36': ['CD36', 'SCARB1', 'SCARB2'],
 'CD223': ['LAG3'],
 'TIGIT': ['TIGIT'],
 'CD226': ['CD226'],
 'CD178': ['FASLG'],
 'CD319': ['SLAMF7'],
 'CD171': ['L1CAM'],
 'Siglec-8': ['SIGLEC8'],
 'CD340': ['ERBB2'],
 'VEGFR-3': ['FLT4'],
 'CD29': ['ITGB1'],
 'CD62E': ['SELE'],
 'CD22': ['CD22'],
 'CD20': ['MS4A1', 'MS4A7', 'MS4A6A', 'MS4A4A', 'MS4A3', 'MS4A10', 'MS4A5'],
 'CD27': ['CD27', 'CD70', 'SIVA1'],
 'CD25': ['IL2RA'],
 'CD24': ['CD24'],
 'CD146': ['MCAM'],
 'Galectin-9': ['LGALS9C', 'LGALS9', 'LGALS9B'],
 'CD142': ['F3'],
 'CD141': ['THBD'],
 'CD294': ['PTGDR2'],
 'CX3CR1': ['CX3CR1'],
 'CD303': ['CLEC4C'],
 'GP130': ['LRPPRC', 'IL31RA', 'IL6ST', 'TLE5'],
 'CD253': ['TNFSF10'],
 'CD357': ['TNFRSF18'],
 'CD354': ['TREM1'],
 'CLEC12A': ['CLEC12A'],
 'Folate': ['FOLR2',
  'FOLR3',
  'FOLR1',
  'SLC19A1',
  'IZUMO1R',
  'SLC46A1',
  'FOLH1B',
  'FOLH1',
  'SLC19A4P'],
 'CD209': ['CD209', 'CLEC4M'],
 'CD152': ['CTLA4'],
 'CD154': ['CD40LG'],
 'CD155': ['PVR'],
 'Cadherin': ['CDH15',
  'CDH6',
  'CDH3',
  'CDH12',
  'CDH13',
  'CDH16',
  'CDH10',
  'CDH18',
  'CDH11',
  'CDH2'],
 'CD201': ['PROCR'],
 'CD204': ['MSR1'],
 'CD205': ['LY75'],
 'CD206': ['MRC1'],
 'CD207': ['CD207'],
 'CD1d': ['CD1D'],
 'CD284': ['TLR4'],
 'CD1c': ['CD1C'],
 'Podoplanin': ['PDPN'],
 'CD1a': ['CD1A'],
 'CD366': ['HAVCR2'],
 'IgM': ['JCHAIN', 'CD5L', 'FCMR', 'CD79A'],
 'CD49d': ['ITGA4'],
 'LOX-1': ['ALOX15', 'OLR1'],
 'TIM-4': ['TIMD4'],
 'CD98': ['SLC3A2', 'SLC7A5'],
 'CD370': ['CLEC9A'],
 'CD49a': ['ITGA1'],
 'C5L2': ['C5AR2'],
 'CD124': ['IL4R'],
 'CD127': ['IL7R'],
 'CD126': ['IL6R'],
 'CD279': ['PDCD1'],
 'CD278': ['ICOS'],
 'CD123': ['IL3RA'],
 'CD122': ['IL2RB'],
 'CD96': ['CD96'],
 'CD274': ['CD274'],
 'CD95': ['FAS', 'FASLG'],
 'CD271': ['NGFR'],
 'CD270': ['TNFRSF14'],
 'CD90': ['THY1'],
 'CD272': ['BTLA'],
 'CD16': ['FCGR3B', 'FCGR3A'],
 'CD14': ['CD14'],
 'CD13': ['ANPEP'],
 'CD267': ['TNFRSF13B'],
 'CD200': ['CD200', 'CD200R1', 'CD200R1L'],
 'CD18': ['ITGB2'],
 'CD19': ['CD19'],
 'CD194': ['CCR4'],
 'CD70': ['CD70'],
 'CD71': ['TFRC'],
 'CD72': ['CD72'],
 'CD73': ['NT5E'],
 'CD177': ['CD177'],
 'CD301': ['CLEC10A'],
 'CD140a': ['PDGFRA'],
 'CD140b': ['PDGFRB'],
 'CD305': ['LAIR1'],
 'CD304': ['NRP1'],
 'CD2': ['CD2', 'CD2BP2', 'SLAMF7', 'CD2AP', 'PSTPIP1', 'SH3KBP1', 'SLAMF9'],
 'CD309': ['KDR'],
 'CD85g': ['LILRA4'],
 'CD110': ['MPL'],
 'CD8': ['CD8B', 'CD8A', 'CD8B2'],
 'CD9': ['CD9', 'PTGFRN'],
 'HLA-DR': ['ANP32B', 'SET', 'CD74', 'ANP32A'],
 'CD137': ['TNFRSF9'],
 'CD134': ['TNFRSF4'],
 'CD135': ['FLT3'],
 'CD61': ['ITGB3'],
 'CD192': ['CCR2'],
 'CD268': ['TNFRSF13C'],
 'CD269': ['TNFRSF17'],
 'CD81': ['CD81', 'IGSF8'],
 'CD80': ['CD80'],
 'CD83': ['CD83'],
 'CD193': ['CCR3'],
 'TSLPR': ['CRLF2'],
 'CD86': ['CD86'],
 'CCR10': ['CCR10', 'ACKR2'],
 'Notch-1': ['NOTCH1'],
 'Notch-2': ['NOTCH2'],
 'CD337': ['NCR3'],
 'CD79b': ['CD79B'],
 'CD79a': ['CD79A', 'IGBP1'],
 'CD49b': ['ITGA2'],
 'CD64': ['FCGR1A'],
 'CD63': ['CD63'],
 'CD69': ['CD69'],
 'CD68': ['CD68'],
 'CD314': ['KLRK1'],
 'CD186': ['CXCR6'],
 'CD185': ['CXCR5'],
 'CD184': ['CXCR4'],
 'CD103': ['ITGAE'],
 'CD102': ['ICAM2'],
 'CD106': ['VCAM1'],
 'CD105': ['ENG'],
 'CD66b': ['CEACAM8'],
 'CD252': ['TNFSF4'],
 'CD109': ['CD109'],
 'CD158f': ['KIR2DL5B', 'KIR2DL5A'],
 'CD8a': ['CD8A'],
 'CD203c': ['ENPP3'],
 'CD52': ['CD52'],
 'CD195': ['CCR5'],
 'CD196': ['CCR6'],
 'CD54': ['ICAM1'],
 'CD55': ['CD55'],
 'CD99': ['CD99', 'CD99L2'],
 'CD59': ['CD59'],
 'CD93': ['CD93'],
 'CD244': ['CD244'],
 'CD158': ['KIR3DL3',
  'KIR2DL5B',
  'KIR2DS5',
  'KIR2DS1',
  'KIR2DL4',
  'KIR2DL2',
  'KIR3DL2',
  'KIR2DS4',
  'KIR2DS2',
  'KIR3DL1'],
 'CD273': ['PDCD1LG2'],
 'CD243': ['ABCB1'],
 'CD325': ['CDH2'],
 'CD324': ['CDH1'],
 'CD307e': ['FCRL5'],
 'CD172a': ['SIRPA'],
 'CD307d': ['FCRL4'],
 'CD42b': ['GP1BA', 'GP1BB'],
 'CD115': ['CSF1R'],
 'CD117': ['KIT'],
 'XCR1': ['XCR1'],
 'CD112': ['PVRIG', 'NECTIN2'],
 'MERTK': ['MERTK'],
 'B7-H4': ['VTCN1'],
 'CD21': ['CR2'],
 'CLEC2': ['CLEC1B'],
 'CD48': ['CD48'],
 'CD47': ['CD47'],
 'CD46': ['CD46'],
 'CD41': ['ITGA2B'],
 'CD40': ['CD40', 'CD40LG', 'TRAF3'],
 'CD43': ['SPN'],
 'CD338': ['ABCG2'],
 'CD235a': ['GYPA'],
 'CD335': ['NCR1'],
 'CD119': ['IFNGR1'],
 'CD169': ['SIGLEC1'],
 'CD28': ['CD28', 'TMIGD2'],
 'CD161': ['KLRB1'],
 'CD163': ['CD163', 'CD163L1'],
 'CD164': ['CD164', 'CD164L2'],
 'CD144': ['CDH5'],
 'CD202b': ['TEK'],
 'CD11c': ['ITGAX']}



def get_marker_genes_for_proteins(protein_names):
    marker_genes = set()
    for protein_name in protein_names:
        marker_genes.update(map_protein_to_genes(protein_name))
    return marker_genes


def map_protein_to_genes(protein_name):
    """Encoding gene symbol(s) for an ADT feature name (or [] if none).

    Normalises the antibody/clone name, drops isotype/hashtag controls, and falls
    back to the identity for bare ``CDxx`` antigens whose gene symbol matches.
    """
    if is_control(protein_name):
        return []
    key = _normalise(protein_name)
    if key in PROTEIN_TO_GENES:
        return PROTEIN_TO_GENES[key]
    if re.fullmatch(r"CD\d+[A-Z]?", key):
        return [key]
    return []
