# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os

from pandas import DataFrame

import mct.Constants as Constants
from mct.BiasTester import BiasTester
from mct.FeatureRanker import FeatureRanker
from mct.MetricComparer import MetricComparer
from mct.PreProcessor import PreProcessor
from mct.Visualizer import Visualizer


class MCT(object):
    def __init__(self, config: json):
        self.__config = config
        log_file = os.path.join(config[Constants.results_dir], Constants.log_file_name)
        logger = logging.getLogger("mct")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(Constants.log_format))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    def process(self, control: DataFrame, treatment: DataFrame, random_state=None):
        preprocessor = PreProcessor(self.__config)
        visualizer = Visualizer(self.__config)

        df_metric, numerical_cols = preprocessor.pre_process_data(control, treatment)
        df_metric_not_norm = df_metric.copy()

        # Compare Control vs Treatment
        delta_comparer = MetricComparer(self.__config)
        control = df_metric[df_metric[Constants.group_column_name] == Constants.control_group]
        treatment = df_metric[df_metric[Constants.group_column_name] == Constants.treatment_group]
        metric_delta = delta_comparer.compare(control, treatment)

        # Bias checker
        bias_tester = BiasTester(self.__config)
        visualizer.create_metric_delta_report(metric_delta, "initial_metric_comparison.html")

        bias_results, deviation, is_biased = bias_tester.check_bias(control, treatment)
        visualizer.create_bias_result_report(bias_results, deviation)

        if is_biased and (self.__config[Constants.normalization_type] != Constants.skip):
            n_control, n_treatment = bias_tester.normalize_bias(control, treatment, bias_results, random_state)

            df_metric = n_control.append(n_treatment)

            b_metric_delta = delta_comparer.compare(n_control, n_treatment)
            visualizer.create_metric_delta_report(b_metric_delta, "normalized_metric_comparison.html")
            n_bias_results, n_deviation, n_is_biased = bias_tester.check_bias(n_control, n_treatment)

        feature_ranker = FeatureRanker(self.__config)
        feature_rank = feature_ranker.compute_ranks(df_metric, df_metric_not_norm, numerical_cols)
        visualizer.create_feature_rank_report(feature_rank)
