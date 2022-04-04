#YW: implement this script to allow customized latent splace interpolation from one dataset to another
# using the author-implemented function (unused by authors), class VAEArith, def linear_interpolation 
# defined in scgen-reproducibility/code/scgen/models/_vae.py


# USE THIS SCRIPT AFTER THE MODEL IS TRAINED (USING train_scGen.py)

import anndata
import scanpy as sc
import scgen
from scipy import sparse

def interpolate_data(data_name="mouse",
                            use_model_trained_without=None,
                            source_adata=None, source_key=None, 
                            dest_adata=None, dest_key=None, 
                            interpolate_key="timepoint",
                            n_steps=3, use_step=None):
    """
        # Parameters
        use_model_trained_without_list: 
            interpolate using the model trained when leaving out these data: list format, such as ['10'] or ['15','20']
            TODO: need to CHANGE train_scGen_mouse.py, test_train_whole_data_some_celltypes_out, saving model NAME function, 
                  in order to access correct model for my case, modify it into:
                  model_path=f"../models/scGen/pbmc/heldout/{"-".join(c_out)}/scgen"
            TODO: need to CHANGE train_scGen, add one function to train model on all input data (test_train_all_data),
                  and let it save into
                  model_path=f"../models/scGen/pbmc/alldata/scgen"
            (TODO: could be modified into a dict format for more detailed usage)
        interpolate_key:
            which dimension to perform interpolation,for example, 
            if interpolating from wt->sod1, interpolate_key = "status", 
            if interpolating from '10' to '25', interpolate_key = "timepoint"
        
        source_adata:
            adata matrix of source cells in gene expresion space (same as linear_interpolation function requirement)
        source_key: 
            if source_adata is None, use source key to retrieve source data matrix. string format, such as '10'
        dest_adata, 
            adata matrix of destinations cells in gene expresion space (same as linear_interpolation function requirement)
        dest_key, 
            if source_adata is None, use source key to retrieve source data matrix. string format, such as '25'
        n_steps:
            Number of steps to interpolate points between `source_adata` and `dest_adata` (same as linear_interpolation)
            when n = 3: it means only interpolate 1 step in between (middle step)
            
        # Returns:
        linear_interpolation: numpy nd-array
            Returns the `numpy nd-array` of interpolated points in gene expression space.
            (of dimension n_steps, latent_space_dim)
        interpolate_middle_data: adata
            returns adata format (contains only pred,ctrl), similar data format as "Reconstructed"(whcih contains pred,ctrl,real_stim)
            
            
            
            
        # reference:  
        linear_interpolation usage example:
        train_data = anndata.read("./data/train.h5ad")
        validation_data = anndata.read("./data/validation.h5ad")
        network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
        network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
        souece = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "control"))]
        destination = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "stimulated"))]
        interpolation = network.linear_interpolation(souece, destination, n_steps=25)
    """
    
    if data_name == "mouse":
        stim_key = "sod1"
        ctrl_key = "wt"
        cell_type_key = "timepoint"
        train = sc.read("../data/train_pbmc.h5ad")
    #elif data_name == "hpoly":
    #    stim_key = "Hpoly.Day10"
    #    ctrl_key = "Control"
    #    cell_type_key = "cell_label"
    #    train = sc.read("../data/train_hpoly.h5ad")
    #elif data_name == "salmonella":
    #    stim_key = "Salmonella"
    #    ctrl_key = "Control"
    #    cell_type_key = "cell_label"
    #    train = sc.read("../data/train_salmonella.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_species.h5ad")
    #elif data_name == "study":
    #    stim_key = "stimulated"
    #    ctrl_key = "control"
    #    cell_type_key = "cell_type"
    #    train = sc.read("../data/train_study.h5ad")

    all_data = anndata.AnnData()
    
    if len(use_model_trained_without) > 1:
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/heldout/{"-".join(use_model_trained_without)}/scgen")
    elif len(use_model_trained_without) == 1:
        cell_type = use_model_trained_without[0]
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/{data_name}/{cell_type}/scgen")
    elif len(use_model_trained_without) == 0:
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001,
                                 model_path=f"../models/scGen/alldata/scgen")
    network.restore_model()
    
    # prepare data
    # TODO: need to implement raise exception messages, for source_adata, source_key etc.
    #if not source_key:
    #    if not source_adata:
    #        raise Exception("Please provide either a source_key or source_adata")
    # current implementation assumes user will either source_key and dest_key or source_adata and dest_adata
    if interpolate_key=="timepoint":
        print("interpolating between timepoints")
    elif interpolate_key == "status":
        print("interpolation between status")
    if (source_key != None) & (dest_key != None):
        sourece = train_data[(train_data.obs[interpolate_key] == source_key)]
        destination = train_data[(train_data.obs[interpolation_key] == dest_key)]
    elif (source_adata != None) & (dest_adata != None):
        source = source_adata
        destination = dest_adata
    
    
    pred = network.linear_interpolation(sourece, destination, n_steps=n_steps)
    print(pred.shape) # TODO: what's the shape of this? and how to put everything into adata
    
    pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
    
    
    
    

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
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        
        print(f"Finish Interpolating for {cell_type}")
        network.sess.close()
    all_data.write_h5ad(f"../data/reconstructed/scGen/{data_name}.h5ad")
    
if __name__ == '__main__':
    #The below step should already been done before using this script
    #test_train_whole_data_one_celltype_out("pbmc", z_dim=100, alpha=0.00005, n_epochs=300, batch_size=32,
    #                                       dropout_rate=0.2, learning_rate=0.001)
    interpolate_middle_data("mouse")