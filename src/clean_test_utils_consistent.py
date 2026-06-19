import numpy as np
import pandas as pd

from src import clean_test_utils as _orig
from src.clean_test_utils import *  # noqa: F401,F403


EXPECTED_RESULT_COLUMNS = [
    "lm_ar1_results_coef",
    "lm_ar1_results_intervention",
    "scipy_AB",
    "scipy_AB_statistic",
    "ttest_results_estimand",
    "ttest_results",
    "exception",
]


LAST_GLS_EXCEPTIONS = {}


def _safe_float(value):
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return np.nan


def _summary_tokens_for_gls(base, fit):
    summary_obj = base.summary(fit)
    try:
        return str(summary_obj[18]).split()
    except Exception:
        return str(summary_obj).split()


def _extract_gls_table_value(ttable, row_name, col_name):
    try:
        value = ttable.rx(row_name, col_name)
    except Exception:
        value = ttable.rx2(row_name).rx2(col_name)

    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            return _safe_float(value[0])
        except Exception:
            pass
    return _safe_float(value)


def _extract_gls_results(base, fit):
    summary_obj = base.summary(fit)
    try:
        ttable = summary_obj.rx2("tTable")
        coef = _extract_gls_table_value(ttable, "Intervention", "Value")
        pvalue = _extract_gls_table_value(ttable, "Intervention", "p-value")
        return coef, pvalue
    except Exception:
        tokens = _summary_tokens_for_gls(base, fit)
        coef = _safe_float(tokens[10]) if len(tokens) > 10 else np.nan
        pvalue = _safe_float(tokens[13]) if len(tokens) > 13 else np.nan
        return coef, pvalue


def _ensure_expected_result_columns(df, default_exception=0):
    df = df.copy()
    if "exception" not in df.columns:
        df["exception"] = default_exception
    for column in EXPECTED_RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def return_gls_pvalues_for_all_ids(nlme, scrt_dfs, IDs, treatment_columns):
    base = _orig.rpackages.importr("base")
    fmla_intervention = _orig.Formula("Value ~ Intervention")
    gls_fits = {}
    LAST_GLS_EXCEPTIONS.clear()

    for i in IDs:
        gls_fits[f"{i}"] = {}
        try:
            fit = nlme.gls(
                fmla_intervention,
                data=scrt_dfs[f"{i}"]["rfull_df"],
                correlation=nlme.corAR1(),
            )
            coef, pvalue = _extract_gls_results(base, fit)

            for treatment_col in treatment_columns:
                gls_fits[f"{i}"][f"lm_ar1_results_{treatment_col.lower()}"] = pvalue

            gls_fits[f"{i}"]["lm_ar1_results_coef"] = coef
            gls_fits[f"{i}"]["exception"] = 0
        except Exception as exc:
            for treatment_col in treatment_columns:
                gls_fits[f"{i}"][f"lm_ar1_results_{treatment_col.lower()}"] = np.nan
            gls_fits[f"{i}"]["lm_ar1_results_coef"] = np.nan
            gls_fits[f"{i}"]["exception"] = 1
            LAST_GLS_EXCEPTIONS[f"{i}"] = str(exc)

    return gls_fits


def get_last_gls_exceptions():
    if not LAST_GLS_EXCEPTIONS:
        return pd.DataFrame(columns=["ID", "exception_message"])
    rows = [{"ID": key, "exception_message": value} for key, value in LAST_GLS_EXCEPTIONS.items()]
    return pd.DataFrame(rows)


def run_scipy_permutation_tests(abab, IDs, treatment_column, target_column):
    scipy_permutation_tests = {}
    for ID in IDs:
        scipy_permutation_tests[f"{ID}"] = {}
        perm_res_ab = _orig.permutation_test(
            (
                abab[(abab[treatment_column] == 0) & (abab["Id"] == ID)][target_column].values,
                abab[(abab[treatment_column] == 1) & (abab["Id"] == ID)][target_column].values,
            ),
            _orig.statistic,
            vectorized=True,
        )

        scipy_permutation_tests[f"{ID}"]["scipy_AB"] = perm_res_ab.pvalue
        scipy_permutation_tests[f"{ID}"]["scipy_AB_statistic"] = _safe_float(perm_res_ab.statistic)

    return scipy_permutation_tests


def ttest_all_IDs(df, IDs, treatment_column, target_column):
    res_ttest = {}
    for ID in IDs:
        res_ttest[f"{ID}"] = {}
        a = df[f"{ID}"][(lambda x: x[treatment_column] == True)][target_column].dropna()
        b = df[f"{ID}"][(lambda x: x[treatment_column] == False)][target_column].dropna()
        ttest = _orig.scipy.stats.ttest_ind(a, b)
        res_ttest[f"{ID}"]["ttest_results"] = ttest.pvalue
        res_ttest[f"{ID}"]["ttest_results_estimand"] = _safe_float(a.mean() - b.mean())

    return pd.DataFrame(res_ttest)

def print_all_test_vals(vals, tests_to_run):
    vals = pd.DataFrame(vals.copy())

    vals.columns = ["ID", "test", "p_value"]

    vals =vals.pivot(columns="test", values="p_value")

    vals2 = pd.DataFrame(columns=vals.columns)

    for col in vals.columns:
        vals2[col] = vals[col].dropna().reset_index(drop=True).astype(float)

    vals2.reset_index(inplace=True)
    vals2.rename(columns={"index":"ID"}, inplace=True)

    vals2["ID"] = range(1,6)
    
    return vals2


def calculate_tests_from_embeddings(
    prepath,
    radius,
    scenarios,
    treatment_column,
    target_column,
    tests_to_run,
    cols_to_groupby,
    alpha_level,
    sim
):
    full_list = []

    for rad in radius:
        for scen in scenarios:
            if sim == True:
                path = f"{prepath}/{rad}_no_drift_white_spots_images_resized/{scen}"
            else:
                path = prepath

            dataset, _ = _orig.return_complete_data_and_pcas_from_paths(
                [path], treatment_column, target_column,rad,scen
            )

            if sim == True:
                data = dataset[dataset["scenario"] == scen].copy()
            else:
                data = dataset.copy()
            IDs = data.Id.unique()
            num_phases = data.phase.max()
            num_obs = int(round(data.groupby("Id").size().unique().mean()))
            limit = int(num_obs / num_phases)

            data = data.sort_values(["Id", "phase"]).copy()

            split_full_datas = _orig.prep_ftest_data(
                data,
                treatment_column=treatment_column,
                target_column=target_column,
            )

            resultslist = []

            if "t-test" in tests_to_run:
                try:
                    all_ttests = ttest_all_IDs(
                        split_full_datas,
                        IDs,
                        treatment_column,
                        target_column,
                    )
                    resultslist.append(all_ttests)
                except Exception:
                    print("t-test failed")

            scrt, nlme = _orig.prep_rpy2_packages()
            scrt_dfs = _orig.prep_rpy_dfs(
                data,
                IDs,
                treatment_columns=[treatment_column],
                target_column=target_column,
            )
            if "scrt" in tests_to_run:
                try:
                    scrt_pvalues = _orig.return_scrt_pvalues(
                        scrt,
                        scrt_dfs,
                        IDs,
                        design="ABAB",
                        limit=limit,
                    )
                    resultslist.append(scrt_pvalues)
                except Exception:
                    pass

            if "lmar1" in tests_to_run:
                try:
                    gls_pvalues = return_gls_pvalues_for_all_ids(
                        nlme,
                        scrt_dfs,
                        IDs,
                        treatment_columns=[treatment_column],
                    )
                    resultslist.append(gls_pvalues)
                except Exception:
                    print("lmar1 failed")

            if "scipy" in tests_to_run:
                try:
                    scipy_permutation_results = run_scipy_permutation_tests(
                        data,
                        IDs,
                        treatment_column,
                        target_column,
                    )
                    resultslist.append(scipy_permutation_results)
                except Exception:
                    print("scipy failed")

            if "mlx" in tests_to_run:
                try:
                    mlxtend_permutation_results = _orig.run_mlxtend_permutation_tests(
                        data,
                        IDs,
                        treatment_column,
                        target_column,
                    )
                    resultslist.append(mlxtend_permutation_results)
                except Exception:
                    print("mlx failed")


            all_vals = _orig.get_all_results(IDs, treatment_column, resultslist)
            if sim:
                all_alpha_vals = _orig.prep_alpha_vals(all_vals)
                all_alpha_vals["radius"] = rad
                all_alpha_vals["scenario"] = scen
                all_alpha_vals.rename(columns={"ttest_results_pvalue": "ttest_results"}, inplace=True)
                all_alpha_vals = _ensure_expected_result_columns(all_alpha_vals)
    
                all_alpha_vals_melt = all_alpha_vals.melt(
                    id_vars=["ID", "radius", "scenario"],
                    value_vars=EXPECTED_RESULT_COLUMNS,
                    var_name="test",
                )
                
    
                power_and_length_resized = (
                    all_alpha_vals_melt.drop_duplicates()
                    .groupby(list(cols_to_groupby))
                    .agg({"value": [lambda x: sum(x < alpha_level) / len(x), lambda x: len(x)]})
                    .reset_index()
                )
    
                power_and_length_resized.columns = list(cols_to_groupby) + ["power", "length"]
                full_list.append(power_and_length_resized)
            else:
                df_all_vals = print_all_test_vals(all_vals,tests_to_run)
                full_list.append(df_all_vals)

    full_df = pd.concat(full_list)
    return full_df
