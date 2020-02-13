# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import mct.Constants as Constants
import mct.Utilities as Utils
from mct.HypothesisTester import chi_square_bias_test


class BiasTester(object):
    """
    Perform a bias check between the control and treatment dataframes.
    """
    __group_control = 'group_control'
    __group_treatment = 'group_treatment'
    __index_group = 'index_group'
    __rf_propensity_scores = 'rf_propensity_scores'

    def __init__(self, config: json):
        self.config = config
        self.__logger = logging.getLogger("mct")
        return

    def check_bias(self, control_df: pd.DataFrame, treatment_df: pd.DataFrame) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame, bool):
        """
        # Compares the destruction of invariant features separately and flag any stat sig
        # difference that satisfies the given minimum percentage deviation threshold
        :param control_df: control dataframe
        :param treatment_df: treatment dataframe
        :return:
        """

        self.__logger.debug('Checking for Population Bias')

        invariant_features = self.config[Constants.invariant_columns]
        p_value_threshold = self.config[Constants.p_value_threshold]
        percentage_deviation_threshold = self.config[Constants.resample_threshold]
        small_bin_percent_threshold = self.config[Constants.small_bin_percent_threshold]

        bias_results, deviation, is_biased = chi_square_bias_test(control_df[invariant_features],
                                                                  treatment_df[invariant_features],
                                                                  groups=[Constants.control_group,
                                                                          Constants.treatment_group],
                                                                  group_column_name=Constants.group_column_name,
                                                                  other_threshold=small_bin_percent_threshold,
                                                                  p_value=0.01)

        bias_results[Constants.num_of_bins] = bias_results[Constants.degree_of_freedom] + 1

        bias_results[Constants.resample] = 'no'
        bias_results.loc[(bias_results[Constants.percentage_deviation] > percentage_deviation_threshold)
                         & (bias_results[Constants.p_value_threshold] < p_value_threshold),
                         Constants.resample] = 'yes'

        # Sort and round Bias results
        bias_results = bias_results.sort_values(by=Constants.percentage_deviation, ascending=False)
        bias_results.sort_values(by=[Constants.percentage_deviation, Constants.feature], ascending=False, inplace=True)

        is_biased = is_biased and (bias_results[Constants.resample] == 'yes').any()
        self.__logger.info("Is Data biased: {0}".format(is_biased))

        # Sort and round deviations.
        deviation.sort_values(
            by=[Constants.feature, Constants.bin_column],
            ascending=False,
            inplace=True)

        return bias_results, deviation, is_biased

    def normalize_bias(self, control: pd.DataFrame, treatment: pd.DataFrame, bias_results: pd.DataFrame,
                       random_state=None) -> (pd.DataFrame, pd.DataFrame):
        """
        Normalize and correct for the major biases.

        bias_results - needs to include columns to normalize, and dof
        """
        self.__logger.debug("Bias Normalization: started")

        Utils.add_group_columns(control, treatment)

        if self.config[Constants.normalization_type] != 'rf':
            message = 'Currently only supported normalization type is random forest'
            self.__logger.error(message)
            raise Exception(message)

        if not bias_results.empty:
            resample_columns = bias_results[Constants.feature]
            max_categories = bias_results[Constants.num_of_bins]

            data_splits = [(self.__group_control, control), (self.__group_treatment, treatment)]

            feature_transforms = [('categorical', x, y) for x, y in zip(resample_columns, max_categories)]

            self.__logger.info('Using RF propensity scores with caliper based matching.')

            # Get data after sampling.
            df_metric = self.__sample_propensity(data_splits, feature_transforms, random_state=random_state)
            df_control = df_metric[df_metric[Constants.group_column_name] == Constants.control_group]
            df_treatment = df_metric[df_metric[Constants.group_column_name] == Constants.treatment_group]

            return df_control, df_treatment
        else:
            self.__logger.info("Bias Normalization skipped.")
        self.__logger.debug("Bias Normalization finished. ")

    # Transform the input data
    def __transform(self, input_frame, features):
        train = pd.DataFrame(index=input_frame.index)
        for func, feat, max_categories in features:
            # Reduce cardinality of input_frame
            dt = input_frame[feat].astype(str)
            feat_counts = dt.value_counts()
            if len(feat_counts) > max_categories:
                dt[~dt.isin(feat_counts[:max_categories].index)] = Constants.other_feature_cluster_name
            # OneHot encode the features
            train = train.join(pd.get_dummies(dt, prefix=feat))

        return train

    def __rf_propensity(self, data, target, random_state=None):

        scalar = StandardScaler()
        data_transformed = scalar.fit_transform(data)

        clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=random_state, n_estimators=10)
        clf.fit(data_transformed, target)
        scores = clf.predict_proba(data_transformed)

        return scores[:, 1]

    def ___matching_1_1(self, df, random_state=None):
        df_c = df[df[self.__index_group] == self.__group_control]
        df_t = df[df[self.__index_group] == self.__group_treatment]
        df_ps = pd.DataFrame(df_c[self.__rf_propensity_scores].value_counts()).join(
            pd.DataFrame(df_t[self.__rf_propensity_scores].value_counts()),
            on=None,
            how='inner',
            lsuffix='l',
            rsuffix='r',
            sort=False)
        df_ps['num2use'] = df_ps[['rf_propensity_scoresl', 'rf_propensity_scoresr']].min(axis=1)
        index_c = []
        index_t = []

        random.seed(a=random_state)
        for i in df_ps.index:
            kk = df_ps.loc[i]['num2use']
            index_c += random.sample([ind for ind in df_c[df_c[self.__rf_propensity_scores] == i].index_original], k=kk)
            index_t += random.sample([ind for ind in df_t[df_t[self.__rf_propensity_scores] == i].index_original], k=kk)

        return index_c, index_t

    def __matching_caliper(self, df,  random_state=None):
        caliper_coeff=self.config[Constants.caliper_coefficient]
        caliper_width = caliper_coeff * df[self.__rf_propensity_scores].std()
        df[self.__rf_propensity_scores] = (df[self.__rf_propensity_scores] / caliper_width).astype(int)
        return self.___matching_1_1(df, random_state=random_state)

    def __sample_propensity(self, splits, feats, match_type='caliper', random_state=None):
        # concatenates the split dataframes, keeping the labels

        df = pd.concat([i for _, i in splits], keys=[splits[0][0], splits[1][0]],
                       names=[self.__index_group, 'index_original'])

        # Note: resetting index, to prevent potential problems with having the same index values after the concat.
        df.reset_index(inplace=True)

        # Set up data frame for classification algorithm.
        pred_frame = self.__transform(df, feats)

        # Get propensity scores using RF algorithm.s
        df[self.__rf_propensity_scores] = self.__rf_propensity(pred_frame, df[self.__index_group],
                                                               random_state=random_state)

        # Perform 1-1 matching based on the propensity scores.
        if match_type == 'caliper':
            ind_c, ind_t = self.__matching_caliper(df, random_state=random_state)
        else:
            ind_c, ind_t = self.___matching_1_1(df, random_state=random_state)

        self.__logger.info("Resampled data size: {}, Percent of retained data: {}:"
                           .format(len(ind_c) * 2, int(len(ind_c) * 2 / len(df) * 100)))
        self.__logger.info("Percent retained in Control {}, Percent of retained in Treatment {}:"
                           .format(int(len(ind_c) / len(df[df[self.__index_group] == self.__group_control]) * 100),
                                   int(len(ind_c) / len(df[df[self.__index_group] == self.__group_treatment]) * 100)))

        return pd.concat([splits[0][1].filter(ind_c, axis=0), splits[1][1].filter(ind_t, axis=0)])
