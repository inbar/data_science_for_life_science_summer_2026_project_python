"""Benchmarking dependency measures for marker-gene identification (PBMC CITE-seq).

Importable pipeline modules used by the analysis notebook:

    config            paths, seeds, constants, method labels
    data_io           download/extract -> cached subsampled RNA+ADT MuData
    preprocessing     QC, normalization, HVGs, shared rank matrix, embeddings
    protein_gene_map  curated antibody -> encoding-gene map
    ground_truth      protein-derived driver sets D_c
    scorers           Spearman / partial-correlation / KSG-MI
    mlp               tanh+softmax MLP and Integrated-Gradients attributions
    metric            AUC_rel driver-recovery metric
    benchmark         run all methods x all cell types
    stats             Friedman + paired Wilcoxon, bootstrap/seed stability
    plotting          publication figures (600 dpi, no titles)
"""
