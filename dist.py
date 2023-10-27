
import numpy as np

def GetCellGeneDistance(
    MCA: dict,
):
    cellsEmb = MCA['cellsCoordinates']
    featuresEmb = MCA['featuresCoordinates']
    dist = parDist(featuresEmb, cellsEmb)
    return dist

def parDist(
    featuresEmb: np.ndarray,
    cellsEmb: np.ndarray,
):
    m, k = cellsEmb.shape
    n, _ = featuresEmb.shape
    An = np.sum(cellsEmb ** 2, axis=1)
    Bn = np.sum(featuresEmb ** 2, axis=1)
    C = -2 * np.dot(cellsEmb, featuresEmb.T)
    C += An[:, np.newaxis]
    C += Bn
    C = np.where(C < 0, 0, C)
    return np.sqrt(C)
