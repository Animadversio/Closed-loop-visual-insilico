import numpy as np

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