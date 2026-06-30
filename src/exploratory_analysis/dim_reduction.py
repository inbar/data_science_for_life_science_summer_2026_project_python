import scanpy as sc

def perform_pca_in_place(dataset,
                         n_pcs=config.N_PCS,
                         seed=config.DEFAULT_SEED):
    """Perform PCA

    Stores: X_pca
    """
    sc.pp.pca(dataset, n_comps=n_pcs, random_state=seed)


def perform_pca_harmony_in_place(dataset,
                                 donor_key=config.DONOR_KEY):
    """Perform PCA with harmony batch correction

    Stores: X_pca_harmoby
    """

    sc.external.pp.harmony_integrate(dataset, donor_key)


def perform_umap_in_place(dataset,
                          n_pcs=config.N_PCS,
                          seed=config.DEFAULT_SEED):
    """Perform UMAP

    Stores: X_umap
    """
    sc.pp.neighbors(dataset, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(dataset, random_state=seed)


def perform_umap_harmony_in_place(dataset,
                                  seed=config.DEFAULT_SEED):
    """Perform UMAP with harmony batch correction

    Stores: X_umap_harmony
    """
    sc.pp.neighbors(dataset, use_rep=OBSM_NAME_PCA_HARMONY, random_state=seed)
    sc.tl.umap(dataset, random_state=seed, key_added=OBSM_NAME_UMAP_HARMONY)
