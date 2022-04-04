# from hf import *

import numpy as np
import scanpy as sc
import scgen

import os
# =============================== downloading training and validation files ====================================
# we do not use the validation data to apply vectroe arithmetics in gene expression space

def train(data_name="pbmc", cell_type="CD4T", p_type="unbiased"):
    train_path = f"../data/train_{data_name}.h5ad"
    if data_name == "pbmc":
        ctrl_key = "control"
        stim_key = "stimulated"
        cell_type_key = "cell_type"
    elif data_name == "hpoly":
        ctrl_key = "Control"
        stim_key = "Hpoly.Day10"
        cell_type_key = "cell_label"
    elif data_name == "salmonella":
        ctrl_key = "Control"
        stim_key = "Salmonella"
        cell_type_key = "cell_label"
    data = sc.read(train_path)
    print("data has been loaded!")
    ctrl_cell = data[(data.obs["condition"] == ctrl_key) & (data.obs[cell_type_key] == cell_type)]
    stim_cell = data[(data.obs["condition"] == stim_key) & (data.obs[cell_type_key] == cell_type)]

    train_real_cd = data[data.obs["condition"] == "control", :]
    if p_type == "unbiased":
        train_real_cd = scgen.util.balancer(train_real_cd)
    train_real_stimulated = data[data.obs["condition"] == "stimulated", :]
    train_real_stimulated = train_real_stimulated[train_real_stimulated.obs["cell_type"] != "CD4T"]
    if p_type == "unbiased":
        train_real_stimulated = scgen.util.balancer(train_real_stimulated)

    import scipy.sparse as sparse
    if sparse.issparse(train_real_cd.X):
        train_real_cd = train_real_cd.X.A
        train_real_stimulated = train_real_stimulated.X.A
    else:
        train_real_cd = train_real_cd.X
        train_real_stimulated = train_real_stimulated.X
    #if sparse.issparse(ctrl_cell.X):
    #    ctrl_cell.X = ctrl_cell.X.A
    #    stim_cell.X = stim_cell.X.A
    #predicted_cells = predict(train_real_cd, train_real_stimulated, ctrl_cell.X)
    #all_Data = sc.AnnData(np.concatenate([ctrl_cell.X, stim_cell.X, predicted_cells]))
    
    #YW: to handle sparse matrix problem, create new variable and proceed with it
    if sparse.issparse(ctrl_cell.X):
         ctrl_cell_dense = ctrl_cell.X.A
         stim_cell_dense = stim_cell.X.A
    else:
         ctrl_cell_dense = ctrl_cell.X
         stim_cell_dense = stim_cell.X.A
    predicted_cells = predict(train_real_cd, train_real_stimulated, ctrl_cell_dense)
    print("Prediction has been finished")
    all_Data=sc.AnnData(np.concatenate([ctrl_cell_dense, stim_cell_dense, predicted_cells]))
    # YW: finished

    all_Data.obs["condition"] = ["ctrl"] * ctrl_cell.shape[0] + ["real_stim"] * stim_cell.shape[0] + \
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = ctrl_cell.var_names

    # YW: add two lines abov each sc.write, to make sure output dir exist for writing, otherwise sc.write fail
    if p_type == "unbiased":
        filename = f"../data/reconstructed/VecArithm/VecArithm_CD4T.h5ad"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sc.write(f"../data/reconstructed/VecArithm/VecArithm_CD4T.h5ad", all_Data)
    else:
        filename = f"../data/reconstructed/VecArithm/VecArithm_CD4T_biased.h5ad"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sc.write(f"../data/reconstructed/VecArithm/VecArithm_CD4T_biased.h5ad", all_Data)


def predict(cd_x, hfd_x, cd_y, p_type="unbiased"):
    if p_type == "biased":
        cd_ind = np.arange(0, len(cd_x))
        stim_ind = np.arange(0, len(hfd_x))
    else:
        eq = min(len(cd_x), len(hfd_x))
        cd_ind = np.random.choice(range(len(cd_x)), size=eq, replace=False)
        stim_ind = np.random.choice(range(len(hfd_x)), size=eq, replace=False)
    cd = np.average(cd_x[cd_ind, :], axis=0)
    stim = np.average(hfd_x[stim_ind, :], axis=0)
    delta = stim - cd
    predicted_cells = delta + cd_y
    return predicted_cells


if __name__ == "__main__":
    train("pbmc", "CD4T", "unbiased")
