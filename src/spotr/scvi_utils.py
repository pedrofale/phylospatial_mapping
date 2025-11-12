# import matplotlib.pyplot as plt
# import numpy as np
# import scanpy as sc
# import scvi
# import seaborn as sns
import torch
from scvi.model import CondSCVI, DestVI


def fit_scLVM(sc_adata, labels_key="clade_level2"):
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key=labels_key)
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train()
    return sc_model

def fit_stLVM(st_adata, sc_model):
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2500)
    return st_model.get_proportions()