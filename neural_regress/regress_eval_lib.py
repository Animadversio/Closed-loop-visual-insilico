import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def compute_D2_per_unit(rspavg_resp_peak, rspavg_pred):
    return 1 - np.square(rspavg_resp_peak - rspavg_pred).sum(axis=0) / np.square(rspavg_resp_peak - rspavg_resp_peak.mean(axis=0)).sum(axis=0)

compute_R2_per_unit = compute_D2_per_unit

def compute_pred_dict_D2_per_unit(fit_models_sweep, Xdict, 
                      resp_mat_sel, idx_train=None, idx_test=None): 
    # , figdir, subject_id, modelname
    if idx_train is None and idx_test is None:
        idx_train, idx_test = train_test_split(
            np.arange(len(resp_mat_sel)), test_size=0.2, random_state=42, shuffle=True
        )
        print(f"using random split to split the data into train and test set (N_train={len(idx_train)}, N_test={len(idx_test)})")
    elif idx_train is not None and idx_test is None:
        idx_test = np.setdiff1d(np.arange(len(resp_mat_sel)), idx_train)
        print(f"using provided idx_train as train set, and the rest as test set (N_train={len(idx_train)}, N_test={len(idx_test)})")
    pred_dict = {}
    D2_per_unit_train_dict = {}
    D2_per_unit_test_dict = {}
    D2_per_unit_dict = {}
    for key in fit_models_sweep.keys():
        if len(key) == 3:
            (model_layer, dimred_str, label) = key
            model_dimred = (model_layer, dimred_str)
        elif len(key) == 2:
            (model_dimred, regressor) = key
        fit_model = fit_models_sweep[key]
        Xfeat = Xdict[model_dimred]
        # Xfeat_tfmer = Xtfmer_lyrswp_RidgeCV[(model_dimred)]
        rspavg_pred = fit_model.predict(Xfeat)
        pred_dict[key] = rspavg_pred
        D2_per_unit = compute_D2_per_unit(resp_mat_sel, rspavg_pred)
        D2_per_unit_train = compute_D2_per_unit(resp_mat_sel[idx_train], rspavg_pred[idx_train])
        D2_per_unit_test = compute_D2_per_unit(resp_mat_sel[idx_test], rspavg_pred[idx_test])
        D2_per_unit_train_dict[key] = D2_per_unit_train
        D2_per_unit_test_dict[key] = D2_per_unit_test
    return {
        "pred_dict": pred_dict,
        "D2_per_unit_dict": D2_per_unit_dict,
        "D2_per_unit_train_dict": D2_per_unit_train_dict,
        "D2_per_unit_test_dict": D2_per_unit_test_dict,
    }
    # pkl.dump({
    #     "pred_dict": pred_dict,
    #     "D2_per_unit_dict": D2_per_unit_dict,
    #     "D2_per_unit_train_dict": D2_per_unit_train_dict,
    #     "D2_per_unit_test_dict": D2_per_unit_test_dict,
    # }, open(join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_pred_meta.pkl"), "wb"))
    

def is_nested_index_of_form(df):
    if not isinstance(df.index, pd.MultiIndex):
        return False
    first_index_entry = df.index[0]
    return (
        isinstance(first_index_entry, tuple) and
        len(first_index_entry) == 2 and
        isinstance(first_index_entry[0], tuple) and
        len(first_index_entry[0]) == 2
    )
    

def format_result_df_tuple_index(result_df_lyrswp, ):
    # Assume result_df_lyrswp is your DataFrame
    result_df_lyrswp = result_df_lyrswp.copy()  # avoid modifying in-place if needed
    # Step 1: Convert MultiIndex to DataFrame
    index_df = result_df_lyrswp.index.to_frame(index=False)
    # Step 2: Split the nested tuple in the first column into two separate columns
    index_df[['layer', 'dimred']] = pd.DataFrame(index_df[0].tolist(), index=index_df.index)
    # Step 3: Add the 'regressor' column and drop the original nested tuple column
    index_df['regressor'] = index_df[1]
    index_df = index_df.drop(columns=[0, 1])
    # Step 3.5: Create a 'layer_dimred' column by joining 'layer' and 'dimred' with an underscore
    index_df['layer_dimred'] = index_df['layer'] + '_' + index_df['dimred']
    # Step 4: Join with the original data
    # result_df_lyrswp = result_df_lyrswp.reset_index(drop=True).join(index_df)
    result_df_lyrswp = index_df.join(result_df_lyrswp.reset_index(drop=True))
    return result_df_lyrswp


def format_result_df(result_df, dimred_list=()):
    # format the result_df to be a dataframe with layer, dimred, regressor, train_score, test_score, parse the key index as layer_dimred, regressor , if it has column unnamed then rename it to layer_dimred, regressor
    """ if index is a multi-index, parse it as layer_dimred, regressor , if it has column unnamed then rename it to layer_dimred, regressor
    The latter case if possible when the frame is loaded from a csv file. 
    """
    
    if "layer" in result_df.columns and "dimred" in result_df.columns and "regressor" in result_df.columns:
        print("already formatted, pass")
        return result_df
    
    if is_nested_index_of_form(result_df):
        # this is the simple case, the first entry is a tuple, then we can easily parse it 
        result_df_formatted = format_result_df_tuple_index(result_df)
        return result_df_formatted
    
    # otherwise, we need to parse the index  into layer and dimred 
    # first check if the index is a multi-index
    if isinstance(result_df.index, pd.MultiIndex):
        result_df_formatted = result_df.reset_index()
        result_df_formatted.rename(columns={"level_0": "layer_dimred", "level_1": "regressor", }, inplace=True)
    else:
        # usually this is the case when the frame is loaded from a csv file
        result_df_formatted = result_df
        result_df_formatted.rename(columns={"Unnamed: 0": "layer_dimred", "Unnamed: 1": "regressor", }, inplace=True)
    
    def split_layer_dimred(x, dimred_list):
        for dimred in dimred_list:
            if x.endswith(dimred):
                # Split at the last _ before the dimred
                prefix = x[:-len(dimred)-1] # Remove dimred and the _
                return prefix, dimred
        raise ValueError(f"Could not find any matching dimred type in {dimred_list} for {x}")
    
    if not len(dimred_list) == 0:
        result_df_formatted["layer"], result_df_formatted["dimred"] = zip(*result_df_formatted["layer_dimred"].apply(lambda x: split_layer_dimred(x, dimred_list)))
    else:
        result_df_formatted["layer"] = result_df_formatted["layer_dimred"].apply(lambda x: x.split("_")[0])
        result_df_formatted["dimred"] = result_df_formatted["layer_dimred"].apply(lambda x: "_".join(x.split("_")[1:]))
    return result_df_formatted


def plot_result_df_per_layer(result_df, shorten_func=None, dimred_list=[], sharey=False, ylim=(None, None), grid=False):
    """
    Plot the result_df per layer, separated by train and test scores; 
    each line is a different dimred x regressor pair
    """
    if shorten_func is None:
        shorten_func = lambda x: x.replace("Bottleneck", "B").replace(".layer", "L")
    
    result_df_formatted = format_result_df(result_df, dimred_list)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=sharey)
    plt.sca(axs[0])
    sns.lineplot(data=result_df_formatted, x="layer", 
                y="train_score", style="regressor", hue="dimred", ax=axs[0], marker="o")
    plt.xticks(rotation=45)
    xticklabels = plt.gca().get_xticklabels()
    xticklabels = [shorten_func(label.get_text()) for label in xticklabels]
    plt.xticks(ticks=range(len(xticklabels)), labels=xticklabels, rotation=45)
    plt.title("Training R2")
    if ylim[0] is not None or ylim[1] is not None:
        plt.ylim(ylim[0], ylim[1])
    axs[0].grid(grid)
    
    plt.sca(axs[1])
    sns.lineplot(data=result_df_formatted, x="layer", 
                y="test_score", style="regressor", hue="dimred", ax=axs[1], marker="o")
    plt.xticks(rotation=45)
    xticklabels = plt.gca().get_xticklabels()
    xticklabels = [shorten_func(label.get_text()) for label in xticklabels]
    plt.xticks(ticks=range(len(xticklabels)), labels=xticklabels, rotation=45)
    plt.title("Test R2")
    if ylim[0] is not None or ylim[1] is not None:
        plt.ylim(ylim[0], ylim[1])
    axs[1].grid(grid)
    plt.tight_layout()
    return fig



def construct_result_df_masked(D2_per_unit_train_dict, D2_per_unit_test_dict, mask=None):
    """
    Construct a new result_df_lyrswp but only considering averaging over a masked set of channels, not all channels
    """
    if mask is None:
        mask = slice(None)
    index = list(D2_per_unit_train_dict.keys())
    result_summary = {}
    for layer_dimred_regressor_idx in index:
        D2_per_unit_train_masked = D2_per_unit_train_dict[layer_dimred_regressor_idx][mask]
        D2_per_unit_test_masked = D2_per_unit_test_dict[layer_dimred_regressor_idx][mask]
        result_summary[layer_dimred_regressor_idx] = \
                {"train_score": D2_per_unit_train_masked.mean(), "test_score": D2_per_unit_test_masked.mean(), }
                # "alpha": alpha, "n_feat": nfeat, "runtime": end_time - start_time
    result_df_masked = pd.DataFrame(result_summary).T
    return result_df_masked