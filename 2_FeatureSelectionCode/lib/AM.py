import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


def standardize_dataframe(df):
    """
    standardize the dataframe

    :parameter
    - df (pandas.DataFrame): input DataFrame。

    :return
    - pandas.DataFrame: standardized DataFrame
    """
    scaler = StandardScaler()

    df_numeric = df.astype('float64')

    standardized_data = scaler.fit_transform(df_numeric)

    df_standardized = pd.DataFrame(standardized_data, columns=df_numeric.columns)

    for col in df.columns:
        if col not in df_numeric.columns:
            df_standardized[col] = df[col]
    return df_standardized


def Anova_M_Info(X, y):

    sel = VarianceThreshold()
    sel = sel.fit(X)
    cols = X.columns[sel.variances_ > 0]
    cols = list(cols)

    # standardize the Dataframe
    X = standardize_dataframe(X)

    # ANOVA
    f_classif_p = f_classif(X, y)[1]
    f_classif_cols = []
    for pValue, colname in zip(f_classif_p, cols):
        if pValue < 0.01:
            f_classif_cols.append(colname)

    # Mutual Info
    MI = mutual_info_classif(X[cols], y)
    MI_threshold = MI.mean() * 0.1
    MI_cols = []

    for MIvalue, colname in zip(MI, cols):
        if MIvalue > MI_threshold:
            MI_cols.append(colname)

    selected_cols = list(set(f_classif_cols) & set(MI_cols))

    return selected_cols

