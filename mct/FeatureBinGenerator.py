# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import numpy as np
import pandas as pd
import scipy.stats as sp

import mct.Constants as Constants
from mct.Utilities import get_is_null_column_name


class FeatureBinGenerator(object):
    """
    Class to bin numeric features
    """

    def __init__(self):
        self.__logger = logging.getLogger("mct")
        return

    @staticmethod
    def create_top_bins(data: pd.DataFrame, column_name: str, number_of_bins: int,
                        minimum_size: int, other_bin_name: str, add_is_null_column: bool) -> pd.DataFrame:
        """
        Create maximum of number_of_bins bins by selecting the top number_of_bins frequent values
        and combining the rest into other bin.

        :param data: dataframe
        :param column_name: column name to bin
        :param number_of_bins:
        :param minimum_size:
        :param other_bin_name:
        :param add_is_null_column:
        :return: binned column
        """

        column = data[column_name].astype(str)
        feats = column.value_counts(dropna=(not add_is_null_column)).to_frame('count').reset_index().sort_values(
            ['count', 'index'], ascending=[False, True])
        feats = feats[:number_of_bins]
        feats = feats.loc[feats['count'] >= minimum_size, 'index']
        column[~column.isin(feats)] = other_bin_name

        return column

    def create_percentile_bins(self, df: pd.DataFrame, num_cols: list, add_null: bool = False,
                               num_bins: int = 4) -> pd.DataFrame:
        """
        Method to bin numerical features by their percentile.

        Numerical Variables
        * Bins data by percentile.
        * Encodes the new variables with variable name, GTE (greater than or equal) to LTE syntax
        * Add a dummy *_nan* variable for recording the fact that feature was null

        Categorical Variables
        * Returns a warning. Doesn't bin the feature.

        :param df: input pandas dataframe
        :param num_cols: a list of the names of the numerical columns in df to bin.
        :param add_null: whether to add the *_nan* features to data (default False)
        :param num_bins: the number of bins to break the data into the percent width of each bin is 100/num_bins.
                         (default 4, is quartile break down, set low to avoid blow up.)

        :returns: Transformed pandas dataframe
        """

        dummy_frames = []

        num_cols = [col for col in num_cols if col in df.columns]

        for col in num_cols:
            # make sure numerical column for binning.
            if df[col].dtype == np.object:
                self.__logger.warning("Warning: Feature {0} is not numerical and wasn't binned.".format(col))
                continue

            # get percentiles
            dt = df[col]
            dt = ((dt.rank() - 1) * num_bins / len(dt.dropna())).apply(np.floor)

            dt_agg = df.groupby(dt)[col].agg([np.min, np.max]).rename(columns={'amin': 'min', 'amax': 'max'})

            for bin_num in dt.unique():
                if np.isnan(bin_num):
                    continue
                if dt_agg.loc[bin_num]['min'] == dt_agg.loc[bin_num]['max']:
                    dt.replace(bin_num, 'is_{}'.format(dt_agg.loc[bin_num]['min']), inplace=True)
                else:
                    dt.replace(bin_num, 'GTE_{}_LTE_{}'.format(dt_agg.loc[bin_num]['min'], dt_agg.loc[bin_num]['max']),
                               inplace=True)

            add_is_null = add_null and df[col].isnull().any()
            dummy_frames.append(pd.get_dummies(dt, prefix=col, dummy_na=add_is_null))

            df.drop(col, axis=1, inplace=True)

        df = df.join(dummy_frames, sort=True)
        return df

    @staticmethod
    def get_feature_dummies(df: pd.DataFrame, feature_cols, target_col, add_null=True, p_thresh=.01,
                            min_data_points=500, max_categories=5, apply_feature_target_metric_dependence_test=True):
        """
        Method to transform a dataframe wrt to a target variable to be used with classification models.

        Numerical Variables
        * Null values are left in dataframe.
        * Add a dummy *is_null* variable for recording the fact that feature was null

        Categorical Variables
        * One hot encode categorical variables
        * To avoid blow up:
            max_categories: Maximum number of categories a feature can have.
                            The rest are collected into a category 'other'
        * To maintain informativeness:
            min_data_points:  The minimum number of points required to create a bin.
            p_thresh:  A chi-squared test is run against target_col.
                       Variable is kept if resulting p_value < p_thresh.  Otherwise dropped.

        :param apply_feature_target_metric_dependence_test:
        :param add_null:
        :param target_col:
        :param df: input pandas dataframe
        :param feature_cols: A list of the feature names in the df to be transformed.
        :param max_categories: Maximum number of categories.
        :param min_data_points: Minimum number of point in categorical bin.
        :param p_thresh: Critical value for chi-squared test.

        :returns: Transformed pandas dataframe, a list of columns that were treated as numerical.

        Future Work:  Algorithm that selects the candidate categorical binning based on information gain as
        opposed to data size.
        """

        is_null_frame = {}
        numerical_columns = []
        dummy_frames = []

        # Drop columns with only single value including null
        for col in feature_cols:
            if df[col].nunique(dropna=False) == 1:
                df.drop(col, axis=1, inplace=True)
                continue

            # For numerical columns {int, float} create is_null.
            if not (df[col].dtype == np.object or df[col].dtype == np.bool):
                if df[col].isnull().any() and add_null:
                    is_null_frame[get_is_null_column_name(col)] = np.isnan(df[col])

                if df[col].nunique(dropna=True) == 1:
                    df.drop(col, axis=1, inplace=True)
                else:
                    numerical_columns.append(col)
            else:
                # For categorical columns create feature dummies.
                dt_col = FeatureBinGenerator.create_top_bins(df, col, max_categories, min_data_points,
                                                             Constants.other_feature_cluster_name,
                                                             add_null)

                if apply_feature_target_metric_dependence_test:
                    chi2, p_val, dof, expected = sp.chi2_contingency(pd.crosstab(dt_col, df[target_col]))
                    if p_val > p_thresh:
                        df.drop(col, axis=1, inplace=True)
                        continue
                # TODO: Refactor the code to create dummies when we need them to optimize the performance
                dummy_frames.append(pd.get_dummies(dt_col, prefix=col))
                df.drop(col, axis=1, inplace=True)

        if add_null:
            null_frame = pd.DataFrame(is_null_frame)
            df = df.join(null_frame, sort=True)

        df = df.join(dummy_frames, sort=True)

        return df, numerical_columns
