import anndata
import scanpy as sc
import scgen
from scipy import sparse
#YW: added import os (solving saving to non-exist path problem)
import os

def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type_to_train=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
        valid = sc.read("../data/valid_pbmc.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_hpoly.h5ad")
        valid = sc.read("../data/valid_hpoly.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
        valid = sc.read("../data/valid_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
        valid = sc.read("../data/valid_species.h5ad")

    for cell_type in train.obs[cell_type_key].unique().tolist():
        if cell_type_to_train is not None and cell_type != cell_type_to_train:
            continue
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        net_valid_data = valid[~((valid.obs[cell_type_key] == cell_type) & (valid.obs[condition_key] == stim_key))]
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")

        network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size)
        network.sess.close()
        print(f"network_{cell_type} has been trained!")

#YW: added cell_type_to_reconstruct=None as def function default option - (enable constructing only rat LPS dataset)
def reconstruct_whole_data(data_name="pbmc", condition_key="condition",cell_type_to_reconstruct=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_hpoly.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/train_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_study.h5ad")

    all_data = anndata.AnnData()
    # YW: change this line to only reconstruct for a given cell type:
    #for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
    # YW add
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        if cell_type_to_reconstruct is not None and cell_type != cell_type_to_reconstruct:
            continue
    #YW: finish adding
        print(f"Reconstructing for {cell_type}")
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        pred, delta = network.predict(adata=net_train_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        #YW: made this little twist to handle indexing problem after adding cell_type_to_reconstrut
        #if idx == 0:
        #    all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        #else:
        #    all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        if cell_type_to_reconstruct is not None:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            if idx == 0:
                all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
            else:
                all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        #YW: finish editing
        
        print(f"Finish Reconstructing for {cell_type}")
        network.sess.close()

    #YW: added the following two lines to enable output if output file path did not exist
    filename = f"../data/reconstructed/scGen/{data_name}.h5ad"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #YW: finish 
    all_data.write_h5ad(f"../data/reconstructed/scGen/{data_name}.h5ad")


def test_train_whole_data_some_celltypes_out(data_name="pbmc",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=None,
                                             c_in=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train_pbmc.h5ad")
        valid = sc.read("../data/valid_pbmc.h5ad")

    net_train_data = scgen.data_remover(train, remain_list=c_in, remove_list=c_out,
                                        cell_type_key=cell_type_key, condition_key=condition_key)

    net_valid_data = scgen.data_remover(valid, remain_list=c_in, remove_list=c_out,
                                        cell_type_key=cell_type_key, condition_key=condition_key)

    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path=f"../models/scGen/pbmc/heldout/{len(c_out)}/scgen")

    network.train(net_train_data, use_validation=True, valid_data=net_valid_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network has been trained!")
    network.sess.close()


def train_cross_study(data_name="study",
                      z_dim=100,
                      alpha=0.00005,
                      n_epochs=300,
                      batch_size=32,
                      dropout_rate=0.2,
                      learning_rate=0.001):
    train = sc.read("../data/train_study.h5ad")
    valid = sc.read("../data/valid_study.h5ad")

    net_train_data = train
    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate,
                             model_path="../models/scGen/study/scgen")

    network.train(net_train_data, use_validation=True, valid_data=valid, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network_{data_name} has been trained!")
    network.sess.close()


if __name__ == '__main__':
    #test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                       dropout_rate=0.2, learning_rate=0.001)
    #test_train_whole_data_one_celltype_out("hpoly", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                       dropout_rate=0.2, learning_rate=0.001)
    #test_train_whole_data_one_celltype_out("salmonella", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                       dropout_rate=0.2, learning_rate=0.001)
    test_train_whole_data_one_celltype_out("species", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
                                           dropout_rate=0.2, learning_rate=0.001, cell_type_to_train="rat")
    #train_cross_study("study", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                  dropout_rate=0.2, learning_rate=0.001)
    #reconstruct_whole_data("pbmc")
    #reconstruct_whole_data("hpoly")
    #YW: the above two models are successfully trained, skip the bottom two
    #reconstruct_whole_data("salmonella")
    #YW: added cell_type_to_reconstruct='rat', to only reconstruct rat
    reconstruct_whole_data("species",cell_type_to_reconstruct='rat')

    #c_in = ['NK', 'B', 'CD14+Mono']
    #c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'Dendritic']
    #test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                         z_dim=100,
    #                                         alpha=0.00005,
    #                                         n_epochs=300,
    #                                         batch_size=32,
    #                                         dropout_rate=0.2,
    #                                         learning_rate=0.001,
    #                                         condition_key="condition",
    #                                         c_out=c_out,
    #                                         c_in=c_in)
    #c_in = ['CD14+Mono']
    #c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'NK', 'B', 'Dendritic']
    #test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                         z_dim=100,
    #                                         alpha=0.00005,
    #                                         n_epochs=300,
    #                                         batch_size=32,
    #                                         dropout_rate=0.2,
    #                                         learning_rate=0.001,
    #                                         condition_key="condition",
    #                                         c_out=c_out,
    #                                         c_in=c_in)
    #c_in = ['CD8T', 'NK', 'B', 'Dendritic', 'CD14+Mono']
    #c_out = ['CD4T', 'FCGR3A+Mono']
    #test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                         z_dim=100,
    #                                         alpha=0.00005,
    #                                         n_epochs=300,
    #                                         batch_size=32,
    #                                         dropout_rate=0.2,
    #                                         learning_rate=0.001,
    #                                         condition_key="condition",
    #                                         c_out=c_out,
    #                                         c_in=c_in)

