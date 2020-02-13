# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os

import numpy as np
import pandas as pd

import mct.Constants as Constants
import mct.Utilities as Utils
from mct.FeatureBinGenerator import FeatureBinGenerator


class PreProcessor(object):

    def __init__(self, config: json):
        self.__config = config
        self.__logger = logging.getLogger("mct")
        return

    def pre_process_data(self, control_df: pd.DataFrame, treatment_df: pd.DataFrame) -> \
            (pd.DataFrame, list):
        """
        # Validating data against the config input.
        # Adding is_null column based on config
        # OneHot-encoding the Categorical features
        # Dropping non-informative features:
        #   - Categorical: based on Chi^2 test
        #   - Numerical: when the feature has only single non-null value
        """

        self.__validate_config()
        self.__validate_column_types(control_df, treatment_df)

        # Partition columns into target, invariant and variant features
        df = Utils.merge_control_treatment(control_df, treatment_df)
        df_feature_target, df_invariant, df_metric_group, feature_columns = self.__partition_columns(df)
        self.__validate_data_set(df)

        # Encode Categorical features - remove ones with 0 variation, or with no impact to the metric.
        # Keep track of numerical columns (possibly containing NULL values)
        df_feature_target_binned, num_cols = FeatureBinGenerator.get_feature_dummies(
            df_feature_target,
            feature_columns,
            self.__config[Constants.metric_column],
            add_null=self.__config[Constants.add_is_null_column],
            p_thresh=0.25,
            min_data_points=1,
            max_categories=self.__config[Constants.num_bins_categorical],
            apply_feature_target_metric_dependence_test=self.__config['apply_feature_target_metric_dependence_test'])

        # Drop target metric column
        df_metric = self.__merge_columns(df_feature_target_binned, df_invariant, df_metric_group)

        return df_metric, num_cols

    def __merge_columns(self, df_feature_target_binned, df_invariant_columns, df_metric_columns):
        metric_column = self.__config[Constants.metric_column]
        df_feature_target_binned.drop(metric_column, axis=1, inplace=True)
        feature_columns = list(df_feature_target_binned.columns)
        if not feature_columns:
            raise Exception("There is no feature left, that meets the threshold criteria")
        self.__config[Constants.feature_columns] = feature_columns
        # Join the feature, invariant and target data_frames.
        df_metric = df_feature_target_binned.merge(df_invariant_columns, copy=False, left_index=True, right_index=True)
        df_metric = df_metric.merge(df_metric_columns, left_index=True, right_index=True)
        df_metric.reset_index(drop=True, inplace=True)
        return df_metric

    def __partition_columns(self, df):
        # Set the metric columns: contains the metric column and Constants.group_column_name column
        metric_column = self.__config[Constants.metric_column]
        df_metric_group = df[[Constants.group_column_name, metric_column]]
        # Set invariant columns.
        invariant_columns = self.__get_available_features(df, self.__config[Constants.invariant_columns])
        df[invariant_columns] = df[invariant_columns].astype('object')
        df[invariant_columns] = df[invariant_columns].fillna('NULL')
        df_invariant = df[invariant_columns]
        # Set feature columns.
        feature_columns = self.__get_available_features(df, self.__config[Constants.feature_columns])
        df_feature_columns = df[feature_columns]
        # Merge features and metric column.
        df_feature_target = df_feature_columns.merge(
            pd.DataFrame(df_metric_group[metric_column]),
            left_index=True,
            right_index=True)
        return df_feature_target, df_invariant, df_metric_group, feature_columns

    def __validate_config(self):
        config = self.__config
        successful = True

        # Check access to the output folder
        output_folder = config[Constants.results_dir]
        if not os.access(output_folder, os.W_OK):
            successful = False
            self.__logger.error('There is no write access to the output folder: {0}'.format(output_folder))

        # Make sure all config parameters exist
        missing_keys = [key for key in Constants.required_config_keys if key not in config.keys()]
        if missing_keys:
            missing = ','.join(str(x) for x in missing_keys)
            successful = False
            self.__logger.error('Following config parameters are missing: {0}'.format(missing))

        # Make sure there is no intersection of metric_col, invariant_columns and feature_columns
        # and deduplicate if there have common features. Giving priorities in the following order:
        # 1) metric_col
        # 2) invariant_columns
        # 3) feature_columns
        config[Constants.feature_columns] = list(set(config[Constants.feature_columns]))
        config[Constants.invariant_columns] = list(set(config[Constants.invariant_columns]))

        if config[Constants.metric_column] in config[Constants.invariant_columns]:
            config[Constants.invariant_columns].remove(config[Constants.metric_column])
            self.__logger.warning(
                'Metric column {0} cannot be part of invariant columns.'.format(config[Constants.metric_column]))

        if config[Constants.metric_column] in config[Constants.feature_columns]:
            config[Constants.feature_columns].remove(config[Constants.metric_column])
            self.__logger.warning(
                'Metric column {0} cannot be part of feature columns.'.format(config[Constants.metric_column]))

        intersection = set(config[Constants.feature_columns]).intersection(config[Constants.invariant_columns])
        if len(intersection) > 0:
            config[Constants.feature_columns] = [feat for feat in config[Constants.feature_columns] if
                                                 feat not in intersection]
            common = ','.join(str(x) for x in intersection)
            self.__logger.warning('Features {0} are set as invariant and cannot be part of a features.'.format(common))

        if not successful:
            raise Exception('The config-file validation has failed!')
        return

    def __validate_data_set(self, data: pd.DataFrame):
        # Check:
        # 1) There are no duplicate columns
        # 2) No reserved prefix/suffix is not used in column name
        # 3) No reserved values is used; e.g 'others'

        successful = True

        feature_columns_set = set(data.columns)

        if len(feature_columns_set) != len(data.columns):
            successful = False
            self.__logger.error('Dataset has duplicate features.')

        if self.__config[Constants.add_is_null_column]:
            for column in feature_columns_set:
                is_null_name = Utils.get_is_null_column_name(column)
                if is_null_name in feature_columns_set:
                    successful = False
                    self.__logger.error('{0} suffix is reserved for a computed is_null column for feature {1} '.format(
                        Utils.get_is_null_column_name(''), column))

        for feature in self.__config[Constants.feature_columns]:
            if Constants.other_feature_cluster_name in data[feature]:
                successful = False
                self.__logger.error('Value {0} is a reserved name and it appears as a value in feature {1}'.format(
                    Constants.other_feature_cluster_name, feature))

        if not successful:
            raise Exception('The data-set validation has failed!')
        return

    def __validate_column_types(self, control: pd.DataFrame, treatment: pd.DataFrame):
        # 1) Validate reserved column names are not used in the control/treatment data
        # 2) Validate that the target metric is either 0/1 or True/False.
        # 3) There is no column of type DateTime or TimeDelta

        successful = True
        reserved_column_names: set = {Constants.group_column_name}
        feature_columns_list = set(control.columns).union(set(treatment.columns))
        reserved_in_use: set = reserved_column_names.intersection(set(feature_columns_list))

        if len(reserved_in_use) > 0:
            successful = False
            self.__logger.error('Dataset has features called {0}. These are reserved keywords.'.format(
                ','.join(list(reserved_in_use))))

        metric_values_c = control[self.__config[Constants.metric_column]].unique().astype(int)
        metric_values_t = treatment[self.__config[Constants.metric_column]].unique().astype(int)
        if (len([value for value in metric_values_c if value not in [1, 0]]) > 0) or (
                len([value for value in metric_values_t if value not in [1, 0]]) > 0):
            successful = False
            self.__logger.error('We currently only support binary target metric.')

        features = set(self.__config[Constants.feature_columns]).union(set(self.__config[Constants.invariant_columns]))
        for feature in features:
            if (control[feature].dtype == np.datetime64) or (treatment[feature].dtype == np.datetime64) or \
                    (control[feature].dtype == np.timedelta64) or (treatment[feature].dtype == np.timedelta64):
                if feature in self.__config[Constants.feature_columns]:
                    self.__config[Constants.feature_columns].remove(feature)
                if feature in self.__config[Constants.invariant_columns]:
                    self.__config[Constants.invariant_columns].remove(feature)
                message = 'Date/Time features are not supported. We have removed column {0} in our the analysis'.format(
                    feature)
                self.__logger.warning(message)

        if not successful:
            raise Exception('The column-type validation has failed!')
        return

    def __get_available_features(self, df: pd.DataFrame, feature_set: list) -> list:
        df_cols = set(df.columns)
        feature_cols = set()
        for feature in feature_set:
            if feature in df_cols:
                feature_cols.add(feature)
            else:
                self.__logger.warning('Feature {0} is missing in the data-set.'.format(feature))

        return list(feature_cols)
