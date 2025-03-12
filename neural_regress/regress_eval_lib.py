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
    if idx_train is None or idx_test is None:
        idx_train, idx_test = train_test_split(
            np.arange(len(resp_mat_sel)), test_size=0.2, random_state=42, shuffle=True
        )
    pred_dict = {}
    D2_per_unit_train_dict = {}
    D2_per_unit_test_dict = {}
    D2_per_unit_dict = {}
    for (model_dimred, regressor) in fit_models_sweep.keys():
        fit_model = fit_models_sweep[(model_dimred, regressor)]
        Xfeat = Xdict[(model_dimred)]
        # Xfeat_tfmer = Xtfmer_lyrswp_RidgeCV[(model_dimred)]
        rspavg_pred = fit_model.predict(Xfeat)
        pred_dict[(model_dimred, regressor)] = rspavg_pred
        D2_per_unit = compute_D2_per_unit(resp_mat_sel, rspavg_pred)
        D2_per_unit_train = compute_D2_per_unit(resp_mat_sel[idx_train], rspavg_pred[idx_train])
        D2_per_unit_test = compute_D2_per_unit(resp_mat_sel[idx_test], rspavg_pred[idx_test])
        D2_per_unit_train_dict[(model_dimred, regressor)] = D2_per_unit_train
        D2_per_unit_test_dict[(model_dimred, regressor)] = D2_per_unit_test
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
    
    
def format_result_df(result_df):
    # format the result_df to be a dataframe with layer, dimred, regressor, train_score, test_score, parse the key index as layer_dimred, regressor
    result_df_formatted = result_df.reset_index()
    result_df_formatted.rename(columns={"level_0": "layer_dimred", "level_1": "regressor", }, inplace=True)
    result_df_formatted["layer"] = result_df_formatted["layer_dimred"].apply(lambda x: x.split("_")[0])
    result_df_formatted["dimred"] = result_df_formatted["layer_dimred"].apply(lambda x: "_".join(x.split("_")[1:]))
    return result_df_formatted


def plot_result_df_per_layer(result_df, shorten_func=None):
    """
    Plot the result_df per layer, separated by train and test scores; 
    each line is a different dimred x regressor pair
    """
    if shorten_func is None:
        shorten_func = lambda x: x.replace("Bottleneck", "B").replace(".layer", "L")
    result_df_formatted = format_result_df(result_df)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(axs[0])
    sns.lineplot(data=result_df_formatted, x="layer", 
                y="train_score", style="regressor", hue="dimred", ax=axs[0], marker="o")
    plt.xticks(rotation=45)
    xticklabels = plt.gca().get_xticklabels()
    xticklabels = [shorten_func(label.get_text()) for label in xticklabels]
    plt.xticks(ticks=range(len(xticklabels)), labels=xticklabels, rotation=45)
    plt.title("Training R2")

    plt.sca(axs[1])
    sns.lineplot(data=result_df_formatted, x="layer", 
                y="test_score", style="regressor", hue="dimred", ax=axs[1], marker="o")
    plt.xticks(rotation=45)
    xticklabels = plt.gca().get_xticklabels()
    xticklabels = [shorten_func(label.get_text()) for label in xticklabels]
    plt.xticks(ticks=range(len(xticklabels)), labels=xticklabels, rotation=45)
    plt.title("Test R2")

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