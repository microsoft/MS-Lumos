# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import pandas as pd

import mct.Constants as Constants
from mct.HypothesisTester import chi_squared_results


class MetricComparer(object):
    """
    Class to compare a metric on two datasets
    """

    def __init__(self, config):
        self.config = config
        self.__logger = logging.getLogger("mct")
        return

    def compare(self, control: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
        """
        :param control: control dataframe
        :param treatment: treatment dataframe
        :return: dataframe [Constants.mean_difference,
                            Constants.mean_control,
                            Constants.mean_treatment,
                            Constants.p_value,
                            Constants.is_stat_sig]
        """
        control_metric = control[self.config[Constants.metric_column]]
        treatment_metric = treatment[self.config[Constants.metric_column]]

        mean_diff, mean_control, mean_treatment, chi2, p_val = chi_squared_results(control_metric, treatment_metric)

        metric_delta = pd.DataFrame(
            [{
                Constants.mean_difference: mean_diff,
                Constants.mean_control: mean_control,
                Constants.mean_treatment: mean_treatment,
                Constants.p_value: p_val,
                Constants.is_stat_sig: (p_val < self.config[Constants.p_value_threshold])
            }])

        return metric_delta
