# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os

import numpy as np
import pandas as pd

import mct.Constants as Constants

# Class to create a visualization of the result of the comparison

_index_ = """<!DOCTYPE html>
<html>
<body>
<section>
    <h2>Initial Metric Comparison:</h2>
    <iframe src=".\\initial_metric_comparison.html" frameBorder="0" height="100%" style="width:100%;height:100px"></iframe>
</section>
<section>
    <h2>Top Level Bias Check:</h2>
    <iframe src=".\\bias_results.html" frameBorder="0" height="100%" style="width:100%;height:200px"></iframe>
</section>
<section>
    <h2>Detailed Bias Check:</h2>
    <iframe src=".\\bias_deviations.html" frameBorder="0" height="100%" style="width:100%;height:400px"></iframe>
</section>
<section>
    <h2>Normalized Metric Comparison (adjusting for biases):</h2>
    <iframe src=".\\normalized_metric_comparison.html" frameBorder="0" height="100%" style="width:100%;height:100px"></iframe>
</section>
<section>
    <h2>Features Explaining Metric Difference:</h2>
    <iframe src=".\\feature_ranking.html" frameBorder="0" height="100%" style="width:100%;height:300px"></iframe>
</section>
<section>
    <h2>Debug:</h2>
    <iframe src="" frameBorder="0" height="100%" style="width:100%;height:20px"></iframe>
</section>
</body>
</html> """


class Visualizer(object):
    """
    Class to create a visualization/report of the result of the comparison
    """

    def __init__(self, config: json):
        self.config = config
        self.__logger = logging.getLogger("mct")
        self.__create_index_file()
        return

    def __create_index_file(self):
        results_dir = self.config[Constants.results_dir]
        index_html = os.path.join(results_dir, "index.html")
        with open(index_html,mode='w') as index:
            index.write(_index_)

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
        results_dir = self.config[Constants.results_dir]
        feature_ranking_file_csv = os.path.join(results_dir, "feature_ranking.csv")
        feature_ranking_file_html = os.path.join(results_dir, "feature_ranking.html")
        sorted_feature = ranked_feature.sort_values(
            by=[Constants.hazard_score, Constants.percent_delta, Constants.count_delta, Constants.feature,
                Constants.expected_failures],
            ascending=False, inplace=False)

        sorted_feature.reset_index(inplace=True, drop=True)
        sorted_feature.to_csv(feature_ranking_file_csv)
        sorted_feature.to_html(feature_ranking_file_html)
