# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os

import numpy as np
import pandas as pd

import mct.Constants as Constants


# Class to create a visualization of the result of the comparison


class Visualizer(object):
    """
    Class to create a visualization/report of the result of the comparison
    """

    def __init__(self, config: json):
        self.config = config
        self.__logger = logging.getLogger("mct")
        return

    def create_metric_delta_report(self, metric_delta: pd.DataFrame, result_file: str):
        # Output metric_delta as HTML.
        metric_delta.sort_values(
            by=[Constants.mean_difference, Constants.mean_control, Constants.mean_treatment],
            inplace=True)
        metric_delta[Constants.mean_difference] = np.round(metric_delta[Constants.mean_difference] * 100, 2)
        metric_delta[Constants.mean_control] = np.round(metric_delta[Constants.mean_control] * 100, 2)
        metric_delta[Constants.mean_treatment] = np.round(metric_delta[Constants.mean_treatment] * 100, 2)
        metric_delta[Constants.p_value] = np.round(metric_delta[Constants.p_value], 4)
        # Output metric_delta as HTML.
        result_file = os.path.join(self.config[Constants.results_dir], result_file)
        metric_delta[
            [Constants.mean_control, Constants.mean_treatment, Constants.mean_difference, Constants.p_value,
             Constants.is_stat_sig]].to_html(
            result_file, index=False, justify='center', index_names=False)

    def create_bias_result_report(self, bias_results: pd.DataFrame, deviation: pd.DataFrame):
        results_dir = self.config[Constants.results_dir]

        bias_results[Constants.p_value_threshold] = np.round(bias_results[Constants.p_value_threshold], 4)
        bias_results[Constants.percentage_deviation] = np.round(bias_results[Constants.percentage_deviation], 2)

        # Sort and round Bias results
        bias_results = bias_results.sort_values(by=Constants.percentage_deviation, ascending=False)
        bias_results[Constants.percentage_deviation] = np.round(bias_results[Constants.percentage_deviation], 2)
        bias_results.sort_values(by=[Constants.percentage_deviation, Constants.feature], ascending=False, inplace=True)

        bias_file = os.path.join(results_dir, "bias_results.html")
        bias_result_columns = [Constants.feature, Constants.num_of_bins, Constants.p_value_threshold,
                               Constants.percentage_deviation,
                               Constants.resample]
        bias_results[bias_result_columns].to_html(bias_file, index=False, justify='center', index_names=False)

        # Sort and round deviations
        deviation.sort_values(by=[Constants.control_group, Constants.feature, Constants.bin_column], ascending=False,
                              inplace=True)
        deviation_file = os.path.join(results_dir, "bias_deviations.html")
        deviation_result_columns = [Constants.feature, Constants.bin_column, Constants.control_percent,
                                    Constants.treatment_percent]
        deviation[Constants.control_percent] = np.round(deviation[Constants.control_percent], 2)
        deviation[Constants.treatment_percent] = np.round(deviation[Constants.treatment_percent], 2)
        deviation[deviation_result_columns].to_html(deviation_file, index=False, justify='center', index_names=False)

    def create_feature_rank_report(self, ranked_feature: pd.DataFrame):
        feature_ranking_file_csv = os.path.join(self.config[Constants.results_dir], "feature_ranking.csv")
        sorted_feature = ranked_feature.sort_values(
            by=[Constants.hazard_score, Constants.percent_delta, Constants.count_delta, Constants.feature,
                Constants.expected_failures],
            ascending=False, inplace=False)

        sorted_feature.reset_index(inplace=True, drop=True)
        sorted_feature.to_csv(feature_ranking_file_csv)
