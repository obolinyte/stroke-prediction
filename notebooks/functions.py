#!/usr/bin/env python
# coding: utf-8

# In[82]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
)
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from typing import List

import statsmodels.stats.api as sms
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline


# ### Project specific functions

# In[83]:


def find_glucose_range(age: float, glucose_level: float) -> str:

    if age <= 5:
        if glucose_level <= 100:
            return "too low"
        elif 100 < glucose_level < 180:
            return "in range"
        else:
            return "too high"
    if 6 <= age <= 9:
        if glucose_level <= 80:
            return "too low"
        elif 80 < glucose_level < 140:
            return "in range"
        else:
            return "too high"
    if age >= 10:
        if glucose_level <= 70:
            return "too low"
        elif 70 < glucose_level < 140:
            return "in range"
        else:
            return "too high"


# In[84]:


def create_age_bins(age: float) -> str:

    if age < 3:
        return "baby"
    elif 3 <= age < 11:
        return "child"
    elif 11 <= age < 18:
        return "adolescent"
    elif 18 <= age < 30:
        return "young adult"
    elif 30 <= age < 65:
        return "adult"
    return "elderly"


# In[85]:


def plot_norm_value_counts(column: str, ax: plt.Axes, df: pd.DataFrame) -> None:
    sns.barplot(
        x=(df[column].value_counts(normalize=True) * 100).index,
        y=(df[column].value_counts(normalize=True) * 100),
        ax=ax,
    )


# In[86]:


def find_percent_by_group(
    col_to_group: str, col_to_count: str, df: pd.DataFrame
) -> pd.DataFrame:

    percent_by_group = (
        df.groupby(col_to_group)[col_to_count]
        .value_counts(normalize=True)
        .to_frame()
        .unstack()
        .fillna(0)
        .stack()
        .rename(columns={col_to_count: "percentage"})
        .reset_index()
    )

    percent_by_group["percentage"] = percent_by_group["percentage"] * 100
    return percent_by_group


# In[87]:


def plot_stroke_hyper_percent(
    feature: str, target_1: str, target_2: str, df: pd.DataFrame
) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    stroke_percent = find_percent_by_group(feature, target_1, df)
    sns.barplot(
        x=feature,
        y="percentage",
        hue=target_1,
        data=stroke_percent,
        ci=None,
        ax=ax[0],
    )
    set_bar_values(ax[0], 9, sign="%", y_location=0.3)
    set_labels(
        ax[0],
        "% of patients that suffered a stroke",
        feature,
        "percentage",
    )

    hyper_percent = find_percent_by_group(feature, target_2, df)
    sns.barplot(
        x=feature,
        y="percentage",
        hue=target_2,
        data=hyper_percent,
        ci=None,
        ax=ax[1],
    )
    set_bar_values(ax[1], 9, sign="%", y_location=0.3)
    set_labels(
        ax[1],
        "% of patients that have hypertension",
        feature,
        "percentage",
    )

    plt.show()


# In[88]:


def plot_stroke_hyper_mean(
    feature: str, target_1: str, target_2: str, df: pd.DataFrame
) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        x=target_1,
        y=feature,
        data=df,
        ax=ax[0],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
    )
    set_labels(
        ax[0],
        "distribution of " + feature + " by stroke",
        "had a stroke?",
        feature,
    )

    sns.boxplot(
        x=target_2,
        y=feature,
        data=df,
        ax=ax[1],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
    )
    set_labels(
        ax[1],
        "distribution of " + feature + " by hypertension",
        "have hypertension?",
        feature,
    )

    plt.show()


# In[89]:


def evaluate_imputer_by_corr(
    list_of_imputers: List, X: pd.DataFrame, target: str, interval_cols: List
) -> pd.DataFrame:

    imputer_by_corr = pd.DataFrame()

    for imputer_name, imputer in list_of_imputers:

        X_num = factorize(X)

        if imputer_name == "Manual":
            X_num["bmi"] = X_num["bmi"].fillna(imputer)
        else:
            imputed_bmi = imputer.fit_transform(X_num)
            X_num = pd.DataFrame(imputed_bmi, columns=X_num.columns)

        corr_matrix = X_num.phik_matrix(interval_cols=interval_cols)
        corr_with_target = (
            corr_matrix[corr_matrix[target] != 1][target].sort_values(ascending=False)
        ).round(3)

        corr_with_target = corr_with_target.to_frame().rename(
            columns={target: imputer_name}
        )

        imputer_by_corr = pd.concat([imputer_by_corr, corr_with_target], axis=1)

    return imputer_by_corr


# In[90]:


def plot_numeric_by_target(
    df: pd.DataFrame, numeric_cols: List, target: str, title: str = "Distribution"
) -> None:

    fig = plt.figure(figsize=(16, 22))

    fig.suptitle(
        title,
        fontsize=22,
        fontweight="semibold",
        y=1.01,
    )

    for i, column in enumerate(numeric_cols, 1):
        ax = plt.subplot(4, 3, i)
        sns.kdeplot(
            x=df[df[target] == "no"][column],
            color="#7bbbce",
            alpha=0.6,
            fill=True,
            legend=True,
            label=str(target) + " - no",
        )
        sns.kdeplot(
            x=df[df[target] == "yes"][column],
            color="#475cbc",
            fill=True,
            alpha=0.6,
            legend=True,
            label=str(target) + " - yes",
        )

        ax.set_xlabel(
            str(column).replace("_", " ").capitalize(),
            fontdict={"fontsize": 16},
            labelpad=12,
        )
        ax.legend()

    plt.tight_layout(h_pad=3, w_pad=3)
    plt.show()


# ### Utility functions

# In[91]:


def set_bar_values(ax: plt.Axes, fontsize: str, sign: str = "", y_location=3) -> None:

    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = f"{round(p.get_height(),1)}{sign}"
        ax.text(
            _x,
            _y + y_location,
            value,
            verticalalignment="bottom",
            ha="center",
            fontsize=fontsize,
        )


# In[92]:


def set_bar_float(ax: plt.Axes, fontsize: str, sign: str = "") -> None:

    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = f"{round(p.get_height(),3)}"
        ax.text(
            _x,
            _y,
            value,
            verticalalignment="bottom",
            ha="center",
            fontsize=fontsize,
        )


# In[93]:


def normalize_bmi(string: str) -> str:
    for word in ["bmi", "Bmi"]:
        if word in string:
            string = string.replace(word, "BMI")
    return string


# In[94]:


def set_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    title = normalize_bmi(title)
    xlabel = normalize_bmi(xlabel)
    ylabel = normalize_bmi(ylabel)

    ax.set_title(title, pad=20, fontsize=14, fontweight="semibold")
    ax.set_xlabel(
        xlabel,
        fontsize=12,
        labelpad=12,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )


# In[95]:


# def set_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:

#     title = title.replace("_", " ").capitalize()
#     xlabel = xlabel.replace("_", " ").capitalize()
#     ylabel = ylabel.replace("_", " ").capitalize()

#     ax.set_title(title, pad=20, fontsize=14, fontweight="semibold")
#     ax.set_xlabel(
#         xlabel,
#         fontsize=12,
#         labelpad=12,
#     )
#     ax.set_ylabel(
#         ylabel,
#         fontsize=12,
#     )


# In[96]:


def highlight_max(s, num_to_highlight):
    is_large = s.nlargest(num_to_highlight).values
    return ["background-color: #a7d0e4" if v in is_large else "" for v in s]


# In[97]:


def map_values(df_to_map: pd.DataFrame, list_of_columns: List) -> pd.DataFrame:

    df = df_to_map.copy()

    for col in list_of_columns:
        df[col] = df[col].map({1: "yes", 0: "no"})

    return df


# In[98]:


def make_mi_scores(
    X: pd.DataFrame, y: pd.Series, discrete_features: pd.Series
) -> pd.Series:

    if y.dtype == "object":
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    else:
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)

    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


# In[99]:


def factorize(X: pd.DataFrame) -> pd.DataFrame:

    X_num = X.copy()

    cols_to_factorize = X.select_dtypes(object).columns
    X_num[cols_to_factorize] = X_num[cols_to_factorize].apply(
        lambda x: pd.factorize(x)[0]
    )

    return X_num


# In[100]:


def return_mi(imputer, X: pd.DataFrame, y: pd.Series) -> pd.Series:

    X_num = factorize(X)

    if imputer != None:
        imputed_bmi = imputer.fit_transform(X_num)
        X_num = pd.DataFrame(imputed_bmi, columns=X_num.columns)

    discrete_features = X_num.dtypes == int
    mi_scores = make_mi_scores(X_num, y, discrete_features)

    return round(mi_scores, 3)


# In[101]:


def insert_axvline(
    line_x: int,
    line_ymin: int,
    line_ymax: int,
    text_x1: int,
    text_x2: int,
    text_str1: str,
    text_str2: str,
) -> None:

    plt.axvline(line_x, line_ymin, line_ymax, linestyle="dashed")

    plt.text(
        x=text_x1,
        y=30,
        s=text_str1,
        horizontalalignment="center",
        color="black",
        rotation=90,
        alpha=1,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )
    plt.text(
        x=text_x2,
        y=30,
        s=text_str2,
        horizontalalignment="center",
        color="black",
        rotation=90,
        alpha=1,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )


# ### Statistical inference functions

# In[102]:


def create_cont_table(col_as_index: str, col: str, df: pd.DataFrame) -> pd.DataFrame:
    cont_table = pd.crosstab(df[col_as_index], df[col])
    cont_table["total"] = cont_table["no"] + cont_table["yes"]
    cont_table = cont_table.drop(columns="no")

    return cont_table


# In[103]:


def evaluate_pvalue(val: int) -> str:

    if val < 0.05:
        return "Stat. significant difference"
    return "Not enough evidence"


# In[104]:


def compare_proportions_ztest(
    contingency_table: pd.DataFrame, label: str = "difference in proportions"
) -> pd.DataFrame:

    successes = np.array(contingency_table["yes"])
    samples = np.array(contingency_table["total"])

    stat, p_value = proportions_ztest(
        count=successes, nobs=samples, alternative="two-sided"
    )

    lb, ub = confint_proportions_2indep(
        count1=np.array(contingency_table.iloc[0])[0],
        nobs1=np.array(contingency_table.iloc[0])[1],
        count2=np.array(contingency_table.iloc[1])[0],
        nobs2=np.array(contingency_table.iloc[1])[1],
    )

    diff_in_proportions = pd.DataFrame(
        data=[p_value, stat, lb, ub],
        index=["p-value", "z-statistic", "CI lower", "CI upper"],
        columns=[label],
    ).T

    diff_in_proportions["significance"] = diff_in_proportions["p-value"].apply(
        evaluate_pvalue
    )

    return diff_in_proportions


# In[105]:


def compare_means_ztest(
    sample_1: pd.DataFrame, sample_2: pd.DataFrame, label: str = "difference in means"
) -> pd.DataFrame:

    cm = sms.CompareMeans(sms.DescrStatsW(sample_1), sms.DescrStatsW(sample_2))
    z_stat, p_val = cm.ztest_ind(usevar="unequal")
    lb, ub = cm.tconfint_diff(usevar="unequal")

    diff_in_means = pd.DataFrame(
        data=[p_val, z_stat, lb, ub],
        index=["p-value", "z-statistic", "CI lower", "CI upper"],
        columns=[label],
    ).T

    diff_in_means["significance"] = diff_in_means["p-value"].apply(evaluate_pvalue)

    return diff_in_means


# In[106]:


def append_diff_in_props(
    cont_table_1: pd.DataFrame, cont_table_2: pd.DataFrame
) -> pd.DataFrame:

    stroke_prop_diff = compare_proportions_ztest(cont_table_1, "stroke")
    hyper_prop_diff = compare_proportions_ztest(cont_table_2, "hypertension")

    return stroke_prop_diff.append(hyper_prop_diff)


# In[107]:


def append_diff_in_means(feature: str, df: pd.DataFrame) -> pd.DataFrame:

    stroke_mean_diff = compare_means_ztest(
        df[df["stroke"] == "no"][feature],
        df[df["stroke"] == "yes"][feature],
        "stroke",
    )

    hyper_mean_diff = compare_means_ztest(
        df[df["hypertension"] == "no"][feature],
        df[df["hypertension"] == "yes"][feature],
        "hypertension",
    )

    return stroke_mean_diff.append(hyper_mean_diff)


# ### ML related functions

# In[108]:


def set_labels_cm(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str, fontsize=12
) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    ax.set_title(title, pad=14, fontsize=fontsize, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=10, labelpad=12)
    ax.set_ylabel(
        ylabel,
        fontsize=10,
    )


# In[109]:


def compare_classifiers(
    lst_of_classifiers: List, preprocessor, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:

    cv_comparison = pd.DataFrame(
        columns=["Classifier", "Fit_time", "Roc_auc", "F1-score", "Precision", "Recall"]
    )

    for model_name, model in lst_of_classifiers:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )
        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=6,
            scoring=("f1_macro", "roc_auc", "precision", "recall"),
        )

        cv_comparison = cv_comparison.append(
            {
                "Classifier": model_name,
                "Fit_time": results["fit_time"].mean(),
                "Roc_auc": results["test_roc_auc"].mean(),
                "F1-score": results["test_f1_macro"].mean(),
                "Precision": results["test_precision"].mean(),
                "Recall": results["test_recall"].mean(),
            },
            ignore_index=True,
        )

    return cv_comparison


# In[110]:


def plot_cm_comparison(
    lst_of_classifiers: List, preprocessor, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:

    f, ax = plt.subplots(2, 3, figsize=(12, 9))
    f.suptitle("Confusion matrix (normalized by actuals)", fontsize=16, y=1)

    for i in range(len(lst_of_classifiers)):
        j = i // 3
        k = i % 3

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", lst_of_classifiers[i][1]),
            ]
        )
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=6)
        g = sns.heatmap(
            confusion_matrix(y_train, y_pred, normalize="true"),
            ax=ax[j][k],
            annot=True,
            fmt=".1%",
            cmap="RdBu",
            cbar=False,
        )
        g.set_title(lst_of_classifiers[i][0])
        g.set_xlabel("Predicted")
        g.set_ylabel("Actual")
        plt.grid(False)

    plt.tight_layout(h_pad=2)


# In[111]:


def create_confusion(test: pd.Series, pred: list) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(test, pred, normalize="true")

    sns.heatmap(
        cm,
        cmap="RdBu_r",
        annot=True,
        ax=ax[0],
        cbar=False,
        square=True,
        annot_kws={"fontsize": 11},
        fmt=".1%",
    )
    set_labels_cm(
        ax[0],
        "confusion matrix (normalized by actuals)",
        "predicted",
        "actual",
    )

    ax[1].axis("off")
    plt.show()


# In[112]:


def create_curves(y_test: pd.Series, pred_proba: List):

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    fpr, tpr, thresh = roc_curve(y_test, pred_proba[:, 1])
    under = auc(fpr, tpr)

    ax[0].plot(fpr, tpr, label="ROC curve (area = %.2f)" % under, color="#2166ac", lw=2)
    ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random guess")
    set_labels_cm(
        ax[0],
        "ROC-auc curve",
        "False Positive Rate",
        "True Positive Rate",
    )
    ax[0].legend()

    precision, recall, threshold = precision_recall_curve(y_test, pred_proba[:, 1])
    ap = average_precision_score(y_test, pred_proba[:, 1])
    ax[1].plot(
        recall, precision, color="#2166ac", lw=2, label="PR curve (AP = %.2f)" % ap
    )
    set_labels_cm(ax[1], "precision-recall curve", "Recall", "Precision")
    ax[1].legend()

    plt.show()


# In[113]:


def get_feature_names(
    composer, num_of_transformers: int, numeric_features: List
) -> List:

    feature_names = list()

    for num in range(0, num_of_transformers):
        cat_feature_names = (
            composer.best_estimator_["preprocessor"]
            .transformers_[num][1]
            .get_feature_names_out()
        )
        feature_names.extend(list(cat_feature_names))

    feature_names.extend(list(numeric_features))

    return feature_names


# In[114]:


def append_cls_metrics(
    metrics_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str,
) -> pd.DataFrame:

    metrics_df = metrics_df.append(
        {
            "Model": model_name,
            "Roc-auc": roc_auc_score(y_test, y_pred_proba[:, 1], average="macro"),
            "F1-score": f1_score(y_test, y_pred, average="macro"),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        },
        ignore_index=True,
    )

    return metrics_df


# In[115]:


def evaluate_model(y_test: pd.Series, predicted_values: list) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--",
        lw=2,
        color="r",
        alpha=0.4,
    )
    sns.scatterplot(x=y_test, y=predicted_values, color="#2166ac", ax=ax[0], s=60)
    set_labels(ax[0], "actuals vs predicted values", "actuals", "predicted")

    residuals = y_test - predicted_values
    ax[1].axhline(y=0, color="r", linestyle="--", lw=2, alpha=0.4)
    sns.scatterplot(x=predicted_values, y=residuals, color="#2166ac", ax=ax[1], s=60)
    set_labels(ax[1], "residuals against predicted values", "predicted", "residuals")

    plt.tight_layout()
    plt.show()


# In[116]:


def compare_regressors(
    lst_of_regressors: List, preprocessor, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:

    cv_comparison = pd.DataFrame(columns=["Regressor", "Fit_time", "R-squared", "RMSE"])

    for model_name, model in lst_of_regressors:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )
        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=6,
            scoring=("r2", "neg_root_mean_squared_error"),
        )

        cv_comparison = cv_comparison.append(
            {
                "Regressor": model_name,
                "Fit_time": results["fit_time"].mean(),
                "R-squared": results["test_r2"].mean(),
                "RMSE": results["test_neg_root_mean_squared_error"].mean() * (-1),
            },
            ignore_index=True,
        )

    return cv_comparison


# In[117]:


def plot_reg_comparison(
    lst_of_regressors: List, preprocessor, X_train: pd.DataFrame, y_train: pd.Series
) -> None:

    f, ax = plt.subplots(2, 3, figsize=(13, 9))
    f.suptitle("Actuals vs Predicted values", fontsize=16, y=1.01)

    for i in range(len(lst_of_regressors)):
        j = i // 3
        k = i % 3

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", lst_of_regressors[i][1]),
            ]
        )
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=6)
        g = sns.scatterplot(
            x=y_train, y=y_pred, ax=ax[j][k], color="#2166ac", alpha=0.6
        )
        ax[j][k].plot(
            [y_train.min(), y_train.max()],
            [y_train.min(), y_train.max()],
            "--",
            lw=1.5,
            color="r",
            alpha=0.4,
        )
        g.set_title(lst_of_regressors[i][0])
        g.set_xlabel("Actuals")
        g.set_ylabel("Predicted")

    plt.tight_layout(h_pad=3)


# In[118]:


def plot_residuals_comparison(
    lst_of_regressors: List, preprocessor, X_train: pd.DataFrame, y_train: pd.Series
) -> None:

    f, ax = plt.subplots(2, 3, figsize=(14, 9))
    f.suptitle("Residuals", fontsize=16, y=1.01)

    for i in range(len(lst_of_regressors)):
        j = i // 3
        k = i % 3

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", lst_of_regressors[i][1]),
            ]
        )
        y_pred = cross_val_predict(pipeline, X_train, y_train, cv=6)
        residuals = y_train - y_pred
        g = sns.histplot(
            residuals,
            kde=True,
            ax=ax[j][k],
            color="#2166ac",
            bins=20,
            edgecolor="white",
        )
        g.set_title(lst_of_regressors[i][0])
        g.set_xlabel("Residuals")
        g.set_ylabel("Count")

    plt.tight_layout(h_pad=3)


# In[119]:


def append_rgss_metrics(
    metrics_df: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, model_name: str
) -> pd.DataFrame:

    metrics_df = metrics_df.append(
        {
            "Model": model_name,
            "R2-score": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred,
                )
            ),
        },
        ignore_index=True,
    )

    return metrics_df


# In[120]:


class AgeBinner(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, encoder=None):
        self.columns = columns
        self.encoder = encoder

    def fit(self, X, y=None):
        return self

    def create_age_bins(self, age):

        if age < 3:
            return "baby"
        elif 3 <= age < 11:
            return "child"
        elif 11 <= age < 18:
            return "adolescent"
        elif 18 <= age < 30:
            return "young adult"
        elif 30 <= age < 65:
            return "adult"
        return "elderly"

    def transform(self, X, y=None):

        cols_for_binning = self.columns
        X["age_group"] = X["age"].apply(self.create_age_bins)

        X = X.drop(["age"], axis=1)
        X = self.encoder.fit_transform(X)
        self.X = X
        return X

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()


# In[121]:


class GlucoseLevelBinner(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, encoder=None):  # default params
        self.columns = columns
        self.encoder = encoder

    def find_glucose_range(self, age, glucose_level):
        if age <= 5:
            if glucose_level <= 100:
                return "too low"
            elif 100 < glucose_level < 180:
                return "in range"
            else:
                return "too high"
        if 6 <= age <= 9:
            if glucose_level <= 80:
                return "too low"
            elif 80 < glucose_level < 140:
                return "in range"
            else:
                return "too high"
        if age >= 10:
            if glucose_level <= 70:
                return "too low"
            elif 70 < glucose_level < 140:
                return "in range"
            else:
                return "too high"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols_for_binning = self.columns
        X["glucose_range"] = X.apply(
            lambda x: self.find_glucose_range(
                x[cols_for_binning[0]], x[cols_for_binning[1]]
            ),
            axis=1,
        )

        X = X.drop(["age", "avg_glucose_level"], axis=1)
        X = self.encoder.fit_transform(X)
        self.X = X
        return X

    def get_feature_names_out(self):
        return self.encoder.get_feature_names_out()

