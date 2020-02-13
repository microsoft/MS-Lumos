# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import numpy as np
import pandas as pd

import mct.Constants as Constants
from mct.FeatureBinGenerator import FeatureBinGenerator
from mct.HypothesisTester import ci_proportion_bounds


class FeatureRanker(object):
    """
    Feature ranking for metric delta.
    """

    def __init__(self, config):
        self.__config = config
        self.__bin_generator = FeatureBinGenerator()
        self.__logger = logging.getLogger("mct")
        return

    def compute_ranks(self, df_metric: pd.DataFrame, df_metric_not_normalized: pd.DataFrame,
                      numerical_cols: list) -> pd.DataFrame:
        """
        Feature ranking for metric delta.
        """
        config = self.__config
        target_col = config[Constants.metric_column]
        add_null = config[Constants.add_is_null_column]
        sorting_type = config[Constants.sort_type]
        # Compute featuring ranking
        df_feature_ranking = self.__decompose_metric_univar(df_metric, df_metric_not_normalized, target_col,
                                                            numerical_cols, add_null)

        if sorting_type == Constants.sorting_type_delta_count:
            sorted_feature = df_feature_ranking.sort_values(Constants.count_delta, ascending=False, inplace=False)
        elif sorting_type == Constants.sorting_type_delta_percent:
            sorted_feature = df_feature_ranking.sort_values(Constants.percent_delta, ascending=False, inplace=False)
        else:
            sorted_feature = df_feature_ranking.sort_values(Constants.hazard_score, ascending=False, inplace=False)

        sorted_feature.reset_index(inplace=True, drop=True)

        return sorted_feature

    def __decompose_metric_univar(self, df_metric, df_metric_not_normalized, target_col, numerical_cols, add_null):
        """
        Computes the univariate feature ranking.
        """
        df_uni_var_un_norm, df_uni_var_norm, categorical_cols = self.__set_univar_frames(df_metric,
                                                                                         df_metric_not_normalized,
                                                                                         numerical_cols, target_col,
                                                                                         add_null)

        expected = []
        actual = []
        contribution = []
        is_sig = []
        feature_results = pd.DataFrame(categorical_cols, columns=[Constants.feature])

        # categorical feature ranking
        for col in categorical_cols:
            sig, exp, act, con = self.__feat_cat_significant(col,
                                                             df_uni_var_norm,
                                                             df_uni_var_un_norm,
                                                             target_col,
                                                             Constants.group_column_name)
            is_sig.append(sig)
            contribution.append(con)
            expected.append(exp)
            actual.append(act)

        feature_results[Constants.is_stat_sig] = is_sig
        feature_results[Constants.count_delta] = contribution
        feature_results[Constants.expected_failures] = expected
        feature_results[Constants.actual_failures] = actual

        # numerical feature ranking
        for col in numerical_cols:
            sig, binned_feats, is_bin_sig, exp, act, con = self.__feat_num_significant(col, df_uni_var_norm,
                                                                                       df_uni_var_un_norm,
                                                                                       target_col,
                                                                                       Constants.group_column_name,
                                                                                       add_null)
            if sig:
                feature_results = feature_results.append(
                    pd.DataFrame(list(zip(*[binned_feats, is_bin_sig, exp, act, con])),
                                 columns=[Constants.feature,
                                          Constants.is_stat_sig,
                                          Constants.expected_failures,
                                          Constants.actual_failures,
                                          Constants.count_delta]),
                    sort=True)

                # Set up columns for output feature ranking.
        feature_results[Constants.percent_delta] = np.abs(
            feature_results[Constants.count_delta] / feature_results[Constants.expected_failures] * 100)
        n_fail_exp = \
            df_metric_not_normalized.loc[
                df_metric_not_normalized[Constants.group_column_name] == Constants.control_group][
                target_col].sum() / len(
                df_metric_not_normalized.loc[
                    df_metric_not_normalized[Constants.group_column_name] == Constants.control_group]) * len(
                df_metric_not_normalized.loc[
                    df_metric_not_normalized[Constants.group_column_name] == Constants.treatment_group])
        n_fail_act = df_metric_not_normalized.loc[
            df_metric_not_normalized[Constants.group_column_name] == Constants.treatment_group][target_col].sum()
        feature_results[Constants.hazard_score] = (feature_results[Constants.actual_failures] / n_fail_act -
                                                   feature_results[Constants.expected_failures] / n_fail_exp) * 100

        feature_results.reset_index(inplace=True, drop=True)

        output_columns = [Constants.feature, Constants.hazard_score, Constants.expected_failures,
                          Constants.actual_failures, Constants.count_delta, Constants.percent_delta]

        stat_sig_features = feature_results[feature_results[Constants.is_stat_sig] == True][output_columns]
        return stat_sig_features

    def __set_univar_frames(self, df_metric, df_metric_not_normalized, numerical_cols, target_col, add_null):
        """
        Prepares data_frames for univariate feature ranking. One before and one after bias normalization.
        Sets up categorical and numerical features.
        """

        config = self.__config
        invar_target_cols = config[Constants.invariant_columns] + [target_col]

        df_invar_target = df_metric_not_normalized[invar_target_cols]
        frame_invar_target, num_cols = self.__bin_generator.get_feature_dummies(
            df_invar_target.copy(),
            config[Constants.invariant_columns],
            target_col,
            min_data_points=1,
            max_categories=config[Constants.num_bins_categorical],
            p_thresh=0.25,
            add_null=add_null,
            apply_feature_target_metric_dependence_test=self.__config['apply_feature_target_metric_dependence_test'])

        # normalized and non-normalized data_frames for feature ranking
        df_uni_var_un_norm = df_metric_not_normalized[
            config[Constants.feature_columns] + [Constants.group_column_name]].merge(frame_invar_target,
                                                                                     left_index=True,
                                                                                     right_index=True)
        df_uni_var_norm = df_metric[config[Constants.feature_columns] + [Constants.group_column_name]].merge(
            frame_invar_target, left_index=True, right_index=True)

        feature_cols = list(df_uni_var_un_norm.columns)
        feature_cols.remove(Constants.group_column_name)
        feature_cols.remove(target_col)
        categorical_cols = [col for col in feature_cols if col not in numerical_cols]

        return df_uni_var_un_norm, df_uni_var_norm, categorical_cols

    def __feat_cat_significant(self, col, df_uni_var_norm, df_uni_var_un_norm, target_col, group_col):
        """
        Determines if categorical col is significant on the normalized dataset.  If it is it computes the impact
        on the non-normalized data set and returns the contribution.
        """
        num_c, len_c, num_t, len_t = self.__feat_info_cat(col, df_uni_var_norm, target_col, group_col)

        sig = self.__sig_check(num_c, len_c, num_t, len_t)
        if not sig:
            return False, 0, 0, 0

        # If number of drops is significant return the number of impacted calls on the original data set.
        num_c, len_c, num_t, len_t = self.__feat_info_cat(col, df_uni_var_un_norm, target_col, group_col)

        return True, num_c * len_t / len_c, num_t, num_t - num_c * len_t / len_c

    def __feat_info_cat(self, col, df_uni_var, target_col, group_col):
        """
        sets up the appropriate dataframe and returns the number of failure associated with the given feature
        on the treatment and control datasets
        """

        return self.__feat_info(df_uni_var[[target_col, group_col, col]], col, target_col, group_col)

    @staticmethod
    def __feat_info(df_col, col, target_col, group_col):
        """
        computes the number of failure associated with the given feature on the treatment and control datasets
        """
        df = df_col.copy()

        df['targ_and_col'] = df[target_col] & df[col]
        col_info = df.groupby(group_col)['targ_and_col'].agg({'size', 'sum'})

        num_c = col_info.loc[Constants.control_group, 'sum']
        len_c = col_info.loc[Constants.control_group, 'size']
        num_t = col_info.loc[Constants.treatment_group, 'sum']
        len_t = col_info.loc[Constants.treatment_group, 'size']

        return num_c, len_c, num_t, len_t

    @staticmethod
    def __sig_check(num_c, len_c, num_t, len_t):
        """
        checks if the change in failures is significant between treatment and control datasets
        """
        low_c, up_c = ci_proportion_bounds(num_c / len_c, len_c)
        low_t, up_t = ci_proportion_bounds(num_t / len_t, len_t)

        if (low_c <= low_t) and (up_c >= low_t):
            return False
        if (low_t <= low_c) and (up_t >= low_c):
            return False

        return True

    def __feat_num_significant(self, col, df_uni_var_norm, df_uni_var_un_norm, target_col, group_col, add_null):
        """
        Determines if the binning for a numerical col is significant on the normalized dataset.
        If it is it computes the impact on the non-normalized data set by creating a new binning,
        and returning significance and contributions.
        """
        df_target_col = df_uni_var_norm[[target_col, group_col, col]]
        df_bin_col = self.__bin_generator.create_percentile_bins(df_target_col.copy(), [col], 
                                                                 num_bins=self.__config[Constants.num_bins_numerical],
                                                                 add_null=add_null)
        binned_feats = [feat for feat in df_bin_col.columns if col in feat]

        sig = False

        for feat in binned_feats:
            num_c, len_c, num_t, len_t = self.__feat_info(df_bin_col[[target_col, group_col, feat]],
                                                          feat,
                                                          target_col,
                                                          group_col)
            sig = self.__sig_check(num_c, len_c, num_t, len_t)
            if sig:
                break

        # if none of the binned features are significant return False, 0 impact
        if not sig:
            return False, 0, 0, 0, 0, 0

        # contribution on the non-normalized data set
        df_target_col = df_uni_var_un_norm[[target_col, group_col, col]]
        df_bin_col = self.__bin_generator.create_percentile_bins(df_target_col.copy(), [col], 
                                                                 num_bins=self.__config[Constants.num_bins_numerical],
                                                                 add_null=add_null)
        binned_feats = [feat for feat in df_bin_col.columns if col in feat]

        expected = []
        actual = []
        contribution = []
        is_sig = []

        for feat in binned_feats:
            num_c, len_c, num_t, len_t = self.__feat_info(df_bin_col[[target_col, group_col, feat]], feat, target_col,
                                                          group_col)
            contribution.append(num_t - num_c * len_t / len_c)
            actual.append(num_t)
            expected.append(num_c * len_t / len_c)
            is_sig.append(self.__sig_check(num_c, len_c, num_t, len_t))

        return True, binned_feats, is_sig, expected, actual, contribution
