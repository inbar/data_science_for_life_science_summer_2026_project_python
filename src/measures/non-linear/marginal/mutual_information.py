# TODO: rewrite

# import numpy as np
# from sklearn.feature_selection import mutual_info_classif
#
# def score(X_ranked: np.ndarray,
#           y: np.ndarray,
#           n_neighbors: int = 3,
#           seed: int = 0,
#           ) -> np.ndarray:
#     """Kraskov-Stoegbauer-Grassberger mutual information per gene.
#
#     Uses ``sklearn.feature_selection.mutual_info_classif`` (the KSG kNN
#     estimator for a continuous feature and a discrete target) on the
#     rank-transformed expression, which mitigates the zero-dropout confound.
#     MI is non-negative and unsigned by construction.
#     """
#     return mutual_info_classif(
#         X_ranked,
#         np.asarray(y).astype(int),
#         discrete_features=False,
#         n_neighbors=n_neighbors,
#         random_state=seed,
#     )
