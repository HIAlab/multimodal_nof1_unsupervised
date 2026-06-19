import random
random.seed(100)

import sys
import os
from pathlib import Path

from src.config_utils import ensure_sys_path, load_config

ROOT = Path(__file__).resolve().parents[1]
yaml_params = load_config()
SYS_PATH = yaml_params.get("SYS_PATH")
ensure_sys_path(SYS_PATH, yaml_params)

import pickle

import scipy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import re
import statsmodels
import statsmodels.api as sm
from tqdm import tqdm
import rpy2.robjects.packages as rpackages

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.vectors import StrVector, ListVector
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

#ALPHA = yaml_params["ALPHA"]
#TESTS_TO_COMPARE_FROM_PAPER = yaml_params["TESTS_TO_COMPARE_FROM_PAPER"]
#target_column = yaml_params["TARGET_COLUMN_PCA1"]
#treatment_column = yaml_params["TREATMENT_COLUMN"]
#TESTS_TO_RUN = yaml_params["TESTS_TO_RUN"]


import warnings
warnings.filterwarnings("ignore", message="An unsupported index was provided and will be ignored when e.g. forecasting.")


from scipy.stats import permutation_test 

def create_phase_column(data, treatment_column):
    data = data.reset_index(drop=True)
    data[treatment_column] = data[treatment_column].astype(int)
    
    data["phase_change"] = data[treatment_column].diff().fillna(0) != 0

    phase_types = data[treatment_column].replace({0: 'A', 1: 'B'})
    phase_number = (data["phase_change"].cumsum() // 2 + 1).astype(str)

    data["Phase"] = phase_types + phase_number
    data["phase"] = data["phase_change"].cumsum() + 1

    data.drop(columns=["phase_change"], inplace=True)

    return data

def prep_embedding_pca_data_for_tests(data_path, emb_file, treatment_column, target_column, components_uni = 1, **kwargs):

    df = pd.read_csv(f"{data_path}/{emb_file}.csv")
 
    df.rename(columns={"Image Id(Id-Timestamp)":"ImageId",
                      "Intervention (Boolean)":"Intervention"}, inplace=True)
    
    try:
        df[["ImageId1","ImageId2"]] = df["ImageId"].str.split("_", expand=True)

        for imageid_col in ["ImageId1","ImageId2"]:
            df[imageid_col] = df[imageid_col].astype(int)

        df = df.sort_values(["Id","ImageId2"])
    except:
        if "ImageId" not in df.columns:
            df["ImageId"] = ""
        df = df.sort_values(["Id","ImageId"])
    df = df.reset_index(drop=True)
    
    emb_cols = [c for c in df.columns if "Embedding" in c]

    all_embeddings = df[emb_cols]
    all_meta_data = df.drop(columns=emb_cols)
    
    if "Reconstruction Error" not in all_meta_data.columns:
        all_meta_data["Reconstruction Error"] = ""
    
    col_dict = {
        "Id": all_meta_data["Id"].values,
        "Intervention": all_meta_data["Intervention"].apply(lambda x: x==True),
        "RecErr": all_meta_data["Reconstruction Error"],
        
    }

        
    data = pd.DataFrame(col_dict)
    
    if "orig_treatment" in data.columns:
        data["orig_treatment"] = all_meta_data["orig_treatment"]
        phase_treatment_col = "orig_treatment"
    else:
        phase_treatment_col = treatment_column
    data = data.groupby("Id").apply(create_phase_column, phase_treatment_col)

    data = data.reset_index(drop=True)
        
    for col in ["AvgScore","Date"]:
        if col in all_meta_data.columns:
            data[col] = all_meta_data[col]
    
    if target_column == "PCA1_uni":
        pca_uni = PCA(n_components=components_uni)
        pca_result_uni = pca_uni.fit_transform(all_embeddings).T
        print(f"PCA variance explained: {pca_uni.explained_variance_ratio_}")

        for comp in range(len(pca_result_uni)):
            data[f"PCA{comp+1}_uni"] = pca_result_uni[comp]
            data[f"PCA{comp+1}_uni_ind"] = 0
            

        datas = []
        for i in data["Id"].unique():
            id_data = data.loc[data["Id"]==i].reset_index(drop=True)
            pca_uni_ind = PCA(n_components=components_uni)
            pca_result_uni_ind = pca_uni_ind.fit_transform(all_embeddings[data["Id"]==i]).T
           


            for comp in range(components_uni): 
                id_data[f"PCA{comp+1}_uni_ind"] = pd.Series(pca_result_uni_ind[comp])

            datas.append(id_data)
        datas = pd.concat(datas)
    else:
        data[target_column] = all_meta_data[target_column]
        datas = data
        pca_uni = []

    
    datas["Id"] = datas["Id"].astype(str)
    IDs = datas.Id.unique()
    
    
    return datas, IDs, pca_uni

def prep_rpy2_packages():
    # import R's utility package
    utils = rpackages.importr('utils')
    # select a mirror for R packages
 #   utils.chooseCRANmirror(ind=52) # select the first mirror in the list
    # R package names
    packnames = ('SCRT','nlme')
    # Selectively install what needs to be installed.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))
    scrt = rpackages.importr('SCRT')
    nlme = rpackages.importr('nlme')
    
    return scrt,nlme

def prep_rpy_dfs(data, IDs, treatment_columns, target_column, num_shifts=4):
    scrt_dfs = {}
    for ID in IDs:
        for treatment_column in treatment_columns:
            if not treatment_column:
                treatment_columns = data[f"{ID}"].filter(like="Treatment").dropna(axis=1).columns
                treatment_col = treatment_columns[0]
            else:
                treatment_col = treatment_column
        scrt_dfs[f"{ID}"] = {}
        df_scrt = pd.DataFrame()
        
        df_scrt["Phase"] = data[data["Id"]==f"{ID}"][["Phase"]].copy()
        df_scrt["Value"] = data[data["Id"]==f"{ID}"][[target_column]].copy()
        for treatment_col in treatment_columns:
            df_scrt.reset_index(drop=True,inplace=True)
            df_scrt[treatment_col] = data[data["Id"]==f"{ID}"][treatment_col].astype(int).reset_index(drop=True).copy()
        df_scrt[target_column] = data[data["Id"]==f"{ID}"][target_column].reset_index(drop=True).copy()
        scrt_dfs[f"{ID}"]["full_df"] = df_scrt
        
        abab = df_scrt.iloc[:df_scrt[df_scrt["Phase"] == "B2"].last_valid_index()+1]
        scrt_dfs[f"{ID}"]["abab_df"] = abab
        

#        with ro.default_converter + pandas2ri.converter:
#            rdf_scrt = ro.conversion.get_conversion().py2rpy(df_scrt.dropna())
#            rabab = ro.conversion.get_conversion().py2rpy(abab)

        with localconverter(ro.default_converter + pandas2ri.converter):
            rdf_scrt = ro.conversion.py2rpy(df_scrt.dropna())
            rabab = ro.conversion.py2rpy(abab)

        scrt_dfs[f"{ID}"]["rfull_df"] = rdf_scrt
        scrt_dfs[f"{ID}"]["rabab_df"] = rabab
    
    return scrt_dfs


def prep_split_and_full_data_per_ID(data, ID, treatment_column, target_column="PCA1_uni"):
    
    df_uni = data[(data["Id"] == ID)].reset_index(drop=True).copy()
    df_uni[f"{treatment_column}_int"] = df_uni[treatment_column].astype(int)
    df_uni[f"{treatment_column}_int_lagged"] = df_uni[f"{treatment_column}_int"].shift(1)
    df_uni[f"{target_column}_lagged"] = df_uni[target_column].shift(1)
    df_uni["Interaction"] = df_uni[f"{treatment_column}_int_lagged"]*df_uni[f"{target_column}_lagged"]
    df_uni["Phase_int"] = df_uni["phase"].astype(int)
    df_uni["Phase_lagged"] = df_uni["phase"].shift(1)
    df_uni["Phase_int_lagged"] = df_uni["Phase_int"].shift(1)

    return df_uni

def prep_ftest_data(data, treatment_column,target_column="PCA1_uni"):
    split_full_datas = {}

    for ID in data.Id.unique():
        df_uni= prep_split_and_full_data_per_ID(data, ID, treatment_column, target_column)        
        split_full_datas[f"{ID}"] = df_uni
    return split_full_datas


def LMM(df, target_column, predictor_cols, group_column):
    
    formula = f"{target_column} ~ {' + '.join([v for v in predictor_cols])}"
    md = statsmodels.regression.mixed_linear_model.MixedLM(endog=df[target_column], exog=df[predictor_cols], groups=df[group_column])
    mdf = md.fit(method=["bfgs"])
    return mdf

def run_LMM_per_ID(df, IDs, target_column, predictor_cols, group_column):
    
    LMMS = {}

    for ID in IDs:

        LMMS[f"{ID}"] = {}
        lmm_intervention = LMM(df[f"{ID}"].iloc[1:], target_column, predictor_cols, f"{group_column}")
        lmm_phase = LMM(df[f"{ID}"].iloc[1:], target_column, predictor_cols, "Phase_int")
     
        LMMS[f"{ID}"][f"lmm_{group_column.lower()}"] = lmm_intervention
        LMMS[f"{ID}"]["lmm_phase_int"] = lmm_phase
    
    return LMMS

def get_all_lmm_coefs(all_nested_models, group_col, ID):
    return all_nested_models[f"{ID}"][f"lmm_{group_col.lower()}"].pvalues[f"{group_col.capitalize()}"]

def get_all_lmm_ftest_results(all_nested_models, IDs, group_col):
    lmm_ftest_results = {}
    intervention_hypothesis = "Intervention_int = 0"
    phase_hypothesis = "Phase_int = 0"
    for ID in IDs:
        lmm_ftest_results[f"{ID}"] = {}
        lmm_ftest_results[f"{ID}"][f"treat_ftest_lmm_results_{group_col.lower()}"] = all_nested_models[f"{ID}"][f"lmm_{group_col.lower()}"].f_test(phase_hypothesis).pvalue
        lmm_ftest_results[f"{ID}"][f"lmm_coef_{group_col.lower()}"] = get_all_lmm_coefs(all_nested_models, group_col,ID)
    return lmm_ftest_results

def return_gls_pvalues_for_all_ids(nlme,scrt_dfs, IDs, treatment_columns):
    base = rpackages.importr('base')
    fmla_intervention = Formula('Value ~ Intervention') # + AvgScore_lagged ')
    gls_fits = {}
    for i in IDs:
        gls_fits[f"{i}"] ={}
        fit = nlme.gls(fmla_intervention, data=scrt_dfs[f"{i}"]["rfull_df"], correlation = nlme.corAR1())
        for treatment_col,nu in zip(treatment_columns,[13]):#[18,23]): 
            gls_fits[f"{i}"][f"lm_ar1_results_{treatment_col.lower()}"] = str(base.summary(fit)[18]).split()[nu] # 18 instead of 13 _lagged
    return gls_fits


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def abs_statistic(x, y, axis):
    return np.abs(np.mean(x, axis=axis) - np.mean(y, axis=axis))

def run_scipy_permutation_tests(abab, IDs,treatment_column,target_column):
    scipy_permutation_tests = {}
    for ID in IDs:
        scipy_permutation_tests[f"{ID}"] = {}
        perm_res_ab = permutation_test(
            (abab[(abab[treatment_column]==0) & (abab["Id"]==ID)][target_column].values, 
        abab[(abab[treatment_column]==1)  & (abab["Id"]==ID)][target_column].values), statistic, vectorized=True,
        )

        scipy_permutation_tests[f"{ID}"]["scipy_AB"] = perm_res_ab.pvalue

    return scipy_permutation_tests




def ttest_all_IDs(df, IDs, treatment_column,target_column):
    res_ttest = {}
    for ID in IDs:
        res_ttest[f"{ID}"] = {}
        a = df[f"{ID}"][(lambda x:x[treatment_column]==True)][target_column].dropna()
        b = df[f"{ID}"][(lambda x:x[treatment_column]==False)][target_column].dropna()
        ttest = scipy.stats.ttest_ind(a, b)
        res_ttest[f"{ID}"]["ttest_results"] = ttest.pvalue

    return pd.DataFrame(res_ttest)

def create_nested_ols_models(split_full_datas, ID, ols_or_glsar="ols", target_column="PCA1_uni", pred_column="Intervention_int"):
    split_full_models = {}

    X = split_full_datas[f"{ID}"].loc[1:,][[pred_column]]
    X = sm.add_constant(X, has_constant='add')

    full_endog = split_full_datas[f"{ID}"].loc[1:,target_column]
    full_exog=X
    no_endog = split_full_datas[f"{ID}"].loc[1:,target_column]
    no_exog=X[["const"]]
    treat_endog = split_full_datas[f"{ID}"].loc[1:,target_column]
    treat_exog=X[["const", pred_column]]
    
    if ols_or_glsar.lower() == "ols":
        full_model = sm.OLS(full_endog, full_exog).fit()
        no_model = sm.OLS(no_endog, no_exog).fit()
        treat_model = sm.OLS(treat_endog, treat_exog).fit()
    else:
        full_model = sm.GLSAR(full_endog, full_exog, rho=1).fit()
        no_model = sm.GLSAR(no_endog, no_exog, rho=1).fit()
        treat_model = sm.GLSAR(treat_endog, treat_exog, rho=1).fit()
    
    split_full_models[f"full_{ols_or_glsar.lower()}_model"] = full_model
    split_full_models[f"no_{ols_or_glsar.lower()}_model"] = no_model
    split_full_models[f"treat_{ols_or_glsar.lower()}_model"] = treat_model
        
    return split_full_models

def get_all_ols_ftest_results(all_nested_models, IDs,pred_column):
    ols_ftest_results = {}
    treat_hypothesis = f"{pred_column} = 0"
    for ID in IDs:
        ols_ftest_results[f"{ID}"] = {}
        ols_ftest_results[f"{ID}"]["treat_ftest_ols_results"] = all_nested_models[f"{ID}"]["nested_ols_models"]["full_ols_model"].f_test(treat_hypothesis).pvalue
    return ols_ftest_results



def get_ols_lr_test(nested_models, IDs):
    lr_results = {}
    for ID in IDs:
        lr_results[f"{ID}"] = {}
        ll_full = nested_models[f"{ID}"]["nested_ols_models"]["full_ols_model"].llf
        ll_reduced = nested_models[f"{ID}"]["nested_ols_models"]["no_ols_model"].llf

        LR_statistic = -2*(ll_reduced-ll_full)
        p_val = scipy.stats.chi2.sf(LR_statistic,2)
        lr_results[f"{ID}"]["ols_lr_results"] = p_val
    return lr_results


def get_all_results(IDs, group_col, resultslist):

    all_vals = []
    for ID in IDs:
        for result in resultslist:
            for key in result[f"{ID}"].keys():
                all_vals.append((f"{ID}", key, result[f"{ID}"][key]))

    return all_vals



def run_hypothesis_tests(DATA_PATH, emb_file, treatment_column,  target_column="PCA1_uni",tests_to_run=["t-test","lmar1","scipy"],**kwargs):
        
    
    group_col = "Phase_int"
    resultslist = []
    
    data, IDs,pca_uni = prep_embedding_pca_data_for_tests(DATA_PATH, emb_file, treatment_column, target_column, **kwargs)
    
    num_phases = data.phase.max()
    num_obs = int(round(data.groupby("Id").size().unique().mean()))
    limit = int(num_obs/num_phases)

    complete_all_vals = []

    split_full_datas = prep_ftest_data(data, treatment_column=treatment_column, target_column=target_column)

    if "lmm" in tests_to_run:
        try:
            all_LMMS = run_LMM_per_ID(split_full_datas, IDs, target_column, [f"{treatment_column}_int", "Phase_int"], "Phase_int")
            all_lmm_results = get_all_lmm_ftest_results(all_LMMS, IDs, group_col= "Phase_int")
            resultslist.append(all_lmm_results)
        except:
            pass

    if "t-test" in tests_to_run:
        try:
            all_ttests = ttest_all_IDs(split_full_datas,IDs,treatment_column, target_column)
            resultslist.append(all_ttests)

        except:
            print("t-test failed")
            pass
    

    if "scipy" in tests_to_run:
        try:
            scipy_permutation_results = run_scipy_permutation_tests(data, IDs,treatment_column,target_column)
            resultslist.append(scipy_permutation_results)
        except:
            print("scipy failed")
            pass


    all_vals = get_all_results(IDs, group_col, resultslist)

    return all_vals


def prep_alpha_vals(all_vals):
    vals = pd.DataFrame(all_vals.copy())

    vals.columns = ["ID", "test", "p_value"]
    ID = vals["ID"].astype(int).unique()
    
    vals =vals.pivot(index="ID",columns="test", values="p_value")
        
    for col in vals.columns:
        vals[col] = vals[col].astype(float)

    vals.reset_index(inplace=True)
    vals.rename(columns={"test":"ID"}, inplace=True)

    
    return vals

def find_key_name(data_path,target_column):
    path_split = data_path.split("/")
    path_split1 = path_split[-1].split("_")
    path_split2 = path_split[-2].split("_")

    mat = [re.search(f"^(?!.*effect|complex).*", path_spli) for path_spli in path_split1]
    mat = list(filter(None,[ma.string  if hasattr(ma,"string") else '' for ma in mat ]))
    mat = '_'.join(mat)

    mat2 = [re.search(f"rad", path_spli) for path_spli in path_split2]
    mat2 = list(filter(None,[ma.string  if hasattr(ma,"string") else '' for ma in mat2 ]))
    mat2 = '_'.join(mat2)
    mat = '_'.join([mat2,mat])

    key_name = f"{mat}_{path_split2[0]}".lstrip("_")
    key_name = f"{key_name}_{target_column}"
    return key_name

def return_all_vals(data_paths, target_column, treatment_column, tests_to_run):
    all_pca_vals = {}
    complete_pca_vals = {}
    for data_path in data_paths:
        
        key_name = find_key_name(data_path,target_column)
        
        if key_name not in all_pca_vals.keys():
            all_pca_vals[key_name] = {}

        emb_files =[o for o in os.listdir(data_path) if "Embeddings_Meta" in o]
        emb_files = list(set([re.match("Embeddings_Meta_\d|", emb_file).group(0) for emb_file in emb_files]))
        #emb_files =[o for o in os.listdir(data_path) if "Embeddings_Meta_10_shuffle" in o]
        emb_files = [x for x in emb_files if x]
        for i,emb_file in enumerate(emb_files):
            
            file_num = i+1
            print(f"reading in {emb_file}")
            
            pca_vals = run_hypothesis_tests(data_path, emb_file, target_column=target_column, treatment_column=treatment_column, tests_to_run=tests_to_run)
                                                       
            all_vals = prep_alpha_vals(pca_vals)
            all_pca_vals[key_name][emb_file] = all_vals
    
    print("Finished pca vals!")
    
    return all_pca_vals

def return_all_pca_vals(all_pca_vals_list,remove_exception=False):
    all_dfs_list = []
    for p in all_pca_vals_list:
        for k in p.keys():
            try:
                pointer = 1
                df = pd.concat(p[k][list(p[k].keys())[0]])
                df["radius"] = k.split("_")[0]
                df["scenario"] = list(p[k].keys())[0]
                all_dfs_list.append(df)
            except:
                pointer = 2
                df = pd.concat(p[k])
                df["scenario"] = list(p.keys())[0]
                all_dfs_list.append(df)
    all_pca_vals = pd.concat(all_dfs_list)

    all_pca_vals = all_pca_vals.reset_index().rename(columns={"level_0":"chunk"}).drop(columns="level_1")
    if pointer == 1:
        all_pca_vals = all_pca_vals.melt(id_vars=["ID", "chunk", "radius", "scenario"])
    elif pointer == 2:
        all_pca_vals = all_pca_vals.melt(id_vars=["ID", "chunk", "scenario"])

    all_pca_vals["value"]=all_pca_vals["value"].astype(float)
    if remove_exception:
        all_pca_vals = all_pca_vals[all_pca_vals["test"] != "exception"]
        
    return all_pca_vals

def sort_radius(df):
    df["radius"] = [a[0] for a in df["scenario"].str.split("_")]
    sorter = ["rad0","rad1","rad2","rad5","rad7","rad10","rad20","rad35","rad40","rad50", "rad60"]
    df.radius = df.radius.astype("category")
    df.radius = df.radius.cat.set_categories(sorter)
    df = df.sort_values(["scenario","radius"])
    return df

def sort_adherence(df):
    sorter = ["0_adherence_Uncertain_Low_Back_Pain","02_adherence_Uncertain_Low_Back_Pain",
              "05_adherence_Uncertain_Low_Back_Pain","07_adherence_Uncertain_Low_Back_Pain",
              "09_adherence_Uncertain_Low_Back_Pain","1_adherence_Uncertain_Low_Back_Pain"]
    df.scenario = df.scenario.astype("category")
    df.scenario = df.scenario.cat.set_categories(sorter)
    df = df.sort_values(["scenario"])
    return df

def load_test_results(results_path,files):
    all_pca_vals_list = []
    for scen in files:
        if os.path.isfile(f"{results_path}/{scen}.pkl"):
            pca_vals = pd.read_pickle(f"{results_path}/{scen}.pkl")
            all_pca_vals_list.append(pca_vals)
    return all_pca_vals_list

def display_power(all_pca_vals_list, cols_to_groupby,
                  tests_to_compare_from_paper, sorter=None, alpha_level=0.05):
    all_pca_vals_resized = return_all_pca_vals(all_pca_vals_list,remove_exception=False)

    power_and_length_resized =all_pca_vals_resized.drop_duplicates().groupby(cols_to_groupby).agg({"value":[lambda x:sum(x<alpha_level)/len(x),
                                                                                      lambda x:len(x)]}).reset_index()

    power_and_length_resized.columns = cols_to_groupby + ["power", "length"]

    if sorter == "radius":
        power_and_length_resized = sort_radius(power_and_length_resized)
    elif sorter == "scenario":
        power_and_length_resized = sort_adherence(power_and_length_resized)
    

    display(power_and_length_resized[lambda x:
                                         (x["test"].isin(tests_to_compare_from_paper))])
    
    return power_and_length_resized



def return_complete_data_and_pcas_from_paths(data_paths,treatment_column,target_column,current_rad,current_scenario):
    complete_data = pd.DataFrame()
    pcas = {}
    for data_path in data_paths:
        #current_rad = data_path.split("/")[-2].split("_")[0]
        #current_scenario = data_path.split('/')[-1]

        if current_rad not in pcas.keys():
            pcas[current_rad] = {}
        
        files = os.listdir(data_path)

        files = [file for file in files if "Embeddings_Meta" in file]
        files = [file.split(".")[0] for file in files]

        comp_data, _,pca_uni = prep_embedding_pca_data_for_tests(data_path, files[0],treatment_column, target_column)
        if current_scenario not in pcas[current_rad].keys():
            pcas[current_rad][current_scenario] = []
        pcas[current_rad][current_scenario].append(pca_uni.explained_variance_ratio_)
        for i in files[1:]:
            data, _,pca_uni = prep_embedding_pca_data_for_tests(data_path, f"{i}",treatment_column, target_column)
            comp_data = pd.concat([comp_data,data])
            pcas[current_rad][current_scenario].append(pca_uni.explained_variance_ratio_)


        comp_data["scenario"] = current_scenario #f"{data_path.split('/')[-1]}"
        comp_data["radius"] = current_rad #data_path.split("/")[-2].split("_")[0]
        complete_data = pd.concat([complete_data,comp_data])
    return complete_data, pcas
