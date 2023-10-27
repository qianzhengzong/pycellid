import pandas as pd
import numpy as np
import anndata
import scanpy as sc

def RunMCA(
    adata: anndata.AnnData,
    nmcs: int = 50,
    reduction: str = 'MCA',
):
    mat = adata.layers['normalised'].copy()
    print('Computing Fuzzy Matrix')
    MCAPrepRes = MCAStep1(mat)
    print("Computing SVD")
    U, S, VT = fast_svd(MCAPrepRes['Z'].T, k=nmcs)
    print("Computing Coordinates")
    MCA = MCAStep2(Z=MCAPrepRes['Z'].T, V=VT[1:,:].T, Dc=MCAPrepRes['Dc'])
    component = [f'{reduction}_{x}' for x in range(1, MCA['cellsCoordinates'].shape[1]+1)]
    MCA['stdev'] = np.delete(S, 0)
    return MCA

def MCAStep1(X):
    mat = X if type(X) is np.ndarray else X.toarray()
    cmin = np.min(mat, axis=0)
    cmax = np.max(mat, axis=0)
    minmax = cmax - cmin
    mat2 = (mat - cmin)
    mat2 = mat2 / minmax
    mat2 = np.nan_to_num(mat2)
    mat3 = 1 - mat2
    mat_bind = np.hstack((mat2, mat3))
    total = np.sum(mat_bind)
    colsum = np.sum(mat_bind, axis=0)
    rowsum = np.sum(mat_bind, axis=1)
    mat_bind = mat_bind / np.sqrt(rowsum)[:, np.newaxis]
    mat_bind = mat_bind / np.sqrt(colsum)
    mat_bind = np.nan_to_num(mat_bind)
    Dc = 1 / np.sqrt(colsum / total)
    Dc = np.nan_to_num(Dc)
    return {'Z': mat_bind, 'Dc': Dc}

def MCAStep2(Z, V, Dc):
    features_coordinate = np.dot(Z, V) * Dc[:, np.newaxis]
    features_coordinate = np.nan_to_num(features_coordinate)
    half_feature = features_coordinate[:len(features_coordinate)//2]
    Zcol = Z.shape[1]
    cells_coordinate = np.sqrt(Zcol) * V
    cells_coordinate = np.nan_to_num(cells_coordinate)
    return {
        'cellsCoordinates': cells_coordinate,
        'featuresCoordinates': half_feature
    }

def fast_svd(
    x,
    k: int = 50
):
    k = k+1 if k < x.shape[1] else x.shape[1]
    U, S, VT = np.linalg.svd(x, full_matrices=False)
    U_trunc = U[:, :k]
    S_trunc = S[:k]
    VT_trunc = VT[:k, :]
    return U_trunc, S_trunc, VT_trunc
