
import json
import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from dist import GetCellGeneDistance

from scipy.stats import hypergeom
from scipy.sparse import coo_matrix

def RunCellHGT():
    pass

def RunCellHGT_local(
    MCA: dict,
    nFeatures: int = 200,
):
    DT = GetCellGeneDistance(MCA)
    sort_index = np.argsort(DT) + 1 #compatible with r
    sort_index_topn = sort_index[:, :nFeatures]
    # DT_topn = np.take_along_axis(DT, sort_index_topn, 1)
    i = sort_index_topn.T #compatible with r
    j = list(np.repeat(range(1, DT.shape[0] + 1), nFeatures))
    features = list(adata.var.index)[:DT.shape[1]]
    cells = list(adata.obs.index)[:DT.shape[1]]
    DT_para = {
        'features': features,
        'cells': cells,
        'i': i,
        'j': j,
        'n.features': nFeatures
    }
    return DT_para

def RunCellHGT_cloud(
    DT_para: list,
    pathways: dict,
    minSize: int = 2,
    pAdjust: bool = True,
    logTrans: bool = True,
):
    i = DT_para['i'].T - 1 
    j = [x-1 for x in DT_para['j']]
    features = DT_para['features']
    cells = DT_para['cells']
    nFeatures = DT_para['n.features']
    data = [1] * i.shape[0] * i.shape[1]
    row_indices = i.flatten()
    col_indices = j
    TargetMatrix = coo_matrix(
        (data, (row_indices , col_indices)),
        shape=(len(features), len(cells))
    )
    pathways_final = {}
    for key, value in pathways.items():
        value = [x for x in value if x in features]
        if len(value) >= minSize:
            pathways_final[key] = value

    pathways_ngenes = []
    row_indices = []
    col_indices = []
    counter = 0
    for key, value in pathways_final.items():
        ngenes = len(value)
        pathways_ngenes.append(ngenes)
        indexes = [features.index(x) for x in value]
        row_indices +=  indexes
        col_indices += [counter] * ngenes
        counter += 1

    print("calculating number of success\n")
    data = [1] * len(row_indices)
    PathwayMatrix = coo_matrix(
        (data, (row_indices , col_indices)),
        shape=(len(features), len(pathways_final))
    )
    q = np.dot(TargetMatrix.T, PathwayMatrix).toarray() - 1
    n = pathways_ngenes
    M = [len(features) - x for x in n]
    N = [nFeatures] * len(pathways_final)
    print("performing hypergeometric test\n")
    hgt_result = np.ones(q.shape)
    for i in range(q.shape[1]):
        hgt_result[:,i] = hypergeomTest(q[:,i], M[i], n[i], n[i])
    if pAdjust:
        hgt_result = np.apply_along_axis(BH_correction, axis=1, arr=hgt_result)
    if logTrans:
        hgt_result = -np.log10(hgt_result + 1e-20)
    return hgt_result

def hypergeomTest(q, M, n, N):
    '''
    q: nubmer of cells' top N genes that in pathways/ success number
    M: all gene number
    n: gene number for each celltype/pathway
    N: top N number/ sampling number
    '''
    hyp = 1 - hypergeom.cdf(q, M, N, n)
    return hyp

import statsmodels.api as sm
from statsmodels.sandbox.stats.multicomp import multipletests

def BH_correction(pvalues):
    reject, p_adjusted, _, _ = multipletests(pvalues, method='fdr_bh')
    return p_adjusted

def FormatPathway(
    pathways: dict,
):
    if len(pathways) == 1:
        organ = list(pathways.keys())[0]
        genelist = pathways[organ]
        new_pathways = dict()
        for key, value in genelist.items():
            new_pathways[key] = list(value.values()) if type(value) == dict else list(value)
        return new_pathways
    else:
        return pathways

def PathwayFromJson(
    json_path: str,
):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def DT_para2rds(
    DT_para: list,
    path: str = 'DT_para.rds'
):
    converter = PythonToRConverter(DT_para)
    r_obj = converter.convert()
    converter.save(r_obj, path)

class PythonToRConverter(object):
    def __init__(self, variable):
        self.r = ro.r
        self.variable = variable
        pandas2ri.activate()
        numpy2ri.activate()
    def to_r_list(self, py_dict):
        r_combined_list = ro.vectors.ListVector([])
        for key, value in py_dict.items():
            converter_value = PythonToRConverter(value)
            r_element = converter_value.convert()
            r_combined_list.rx2[key] = r_element
        return r_combined_list
    def to_r_int(self, py_list):
        return ro.IntVector(py_list)
    def to_r_str(self, py_list):
        return ro.StrVector(py_list)
    def to_r_float(self, py_list):
        return ro.FloatVector(py_list)
    def to_r_dataframe(self, py_dataframe):
        return pandas2ri.pandas2ri.py2rpy_pandasdataframe(py_dataframe)
    def to_r_matrix(self, py_matrix):
        nr,nc = py_matrix.shape
        r_matrix = ro.r.matrix(py_matrix, nrow=nr, ncol=nc)
        return r_matrix
    def convert(self):
        if type(self.variable) == dict:
            return self.to_r_list(self.variable)
        if type(self.variable) == list:
            first_type = type(self.variable[0])
            if first_type in [int, np.int64]:
                return self.to_r_int(self.variable)
            if first_type == str:
                return self.to_r_str(self.variable)
            if first_type == float:
                return self.to_r_float(self.variable)
        if type(self.variable) == np.ndarray:
            return self.to_r_matrix(self.variable)
        if type(self.variable) == pd.DataFrame:
            return self.to_r_dataframe(self.variable)
        if type(self.variable) in [int, np.int64]:
            return self.to_r_int([self.variable])
        if type(self.variable) == str:
            return self.to_r_str([self.variable])
        if type(self.variable) == float:
            return self.to_r_float([self.variable])
    def save(self, value, path):
        ro.r.assign("value", value)
        ro.r("saveRDS(value, file='{}')".format(path))

