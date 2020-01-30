# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import scipy.stats as sp
import statsmodels.stats.api as sms

import mct.Constants as Constants


# t-test
def get_t_test_result(a, b):
    # Run t-test on control and treatment.
    (mean_diff, control_diff, stat, p_value) = __two_sample_t_test(a, b)
    (lower, upper) = __t_test_conf_interval(a, b)
    mean_rel = mean_diff / control_diff
    lower = lower
    upper = upper
    return mean_diff, mean_rel, lower, upper, stat, p_value


def __two_sample_t_test(a, b):
    # Run t-test on control and treatment.
    (stat, p_value) = sp.ttest_ind(a, b, equal_var=False)
    control_mean = np.mean(a)
    mean_diff = control_mean - np.mean(b)
    return mean_diff, control_mean, stat, p_value


def __t_test_conf_interval(a, b):
    cm = sms.CompareMeans(sms.DescrStatsW(a), sms.DescrStatsW(b))
    return cm.tconfint_diff(usevar='pooled')


# bernoulli test - Test of Proportions
def chi_squared_results(a, b):
    # Run Chi-Squared Test on control and treatment.
    mean_control = np.mean(a)
    mean_treatment = np.mean(b)
    mean_diff = mean_treatment - mean_control

    df = pd.concat([
        pd.DataFrame(data={'metric': a, 'label': Constants.control_group}),
        pd.DataFrame(data={'metric': b, 'label': Constants.treatment_group})
    ])

    cont_table = pd.crosstab(df['label'], df['metric'])
    chi2, p_val, dof, expected = sp.chi2_contingency(cont_table)

    return mean_diff, mean_control, mean_treatment, chi2, p_val


def ci_proportion_bounds(p, n):
    err = __ci_proportion(p, n)
    return (p - err), (p + err)


def __se_proportion(p, n):
    return np.sqrt((p * (1 - p)) / n)


def __ci_proportion(p, n):
    return 1.96 * __se_proportion(p, n)


"""
Created on Fri Apr  8 09:48:53 2016

Python utility to perform if data is biased between two groups.

NOTE: This only performs a bias check for categorical values.
It does not perform a bias check for numeric variables.

See examples in https://onlinecourses.science.psu.edu/stat414/node/311

@author: jayagup
"""


def chi_square_bias_test(control, treatment, groups, group_column_name, other_threshold, p_value):
    """
    Compute the chi square test of homogeneity between two
    groups.

    See details in https://onlinecourses.science.psu.edu/stat414/node/311

    :input data_1: The first data frame.
    :input data_2: The second data frame.
    :input groups: The name of the groups.
    :input p_value: The p-value with which to evaluate univariate table.

    :returns is_biased: True if biased else False
    :returns table_biased: A table of results with differences..
    """

    contingency_table_c = __get_contingency_table(control)
    contingency_table_t = __get_contingency_table(treatment)

    # Add group information to the dataframes.
    contingency_table_c[group_column_name] = groups[0]
    contingency_table_t[group_column_name] = groups[1]

    # Create a single contingency table including both groups.
    contingency_table = contingency_table_c.append(contingency_table_t)

    # Run the bias check.
    return chi_square_bias_test_contingency(contingency_table, other_threshold, p_value)


def chi_square_bias_test_contingency(df_cont_table, other_threshold, p_value):
    """
    Chi square test of homogeneity over all features.
    """
    # Perform the bias check for all features, one feature at a time.
    bias_results = []
    deviation = pd.DataFrame()
    for feature in df_cont_table[Constants.feature].unique():
        chi2, p_val, dof, perc_dev, feature_deviation = bias_check_covariate(
            df_cont_table, feature, other_threshold)
        feature_deviation[Constants.feature] = feature
        deviation = deviation.append(feature_deviation)

        bias_results.append({
            Constants.feature: feature,
            "chi_square": chi2,
            "p_value": p_val,
            Constants.degree_of_freedom: dof,
            Constants.percentage_deviation: perc_dev})

    df_bias_results = pd.DataFrame(bias_results)

    # Test whether each feature meets the p-value criterion.
    p_value_check = (np.sum(df_bias_results.p_value < p_value) > 0)

    return df_bias_results, deviation, p_value_check


def bias_check_covariate(df_cont_table, feature, other_threshold=1.0):
    """
    Chi square test of homogeneity for single feature.

    :input df_cont_table: Counts for the feature.
    :input feature: The name of the feature.
    :return outcome of the chi square bias check.
    """
    # Filter the feature.
    df_cont_feature = df_cont_table[df_cont_table.feature == feature]

    # Pivot the counts to create R X C format.
    df_cont_pivot = pd.pivot_table(
        df_cont_feature,
        values='count',
        columns=Constants.group_column_name,
        index=Constants.bin_column)

    df_cont_pivot, grps = __combine_small_bins(df_cont_pivot, other_threshold)

    # Feed the contingency table to chi square test.
    chi2, p_val, dof, expected = sp.chi2_contingency(df_cont_pivot)

    # Compute the probability deviation from expected.
    diff_percent = np.abs(expected - df_cont_pivot) / sum(expected) * 100.0

    # Compute percentage for each bin.
    grp_percent = list(map(lambda x: x + "_percent", grps))  # control_percent, treatment_percent
    diff_percent[grp_percent] = df_cont_pivot[grps] / df_cont_pivot[grps].sum() * 100.0
    diff_percent.reset_index(inplace=True)

    # Sum of differences of expected probability and observed probability
    # Note that the sum should be the same for the two columns.
    perc_dev = np.max(diff_percent[grps.tolist()].max())

    return chi2, p_val, dof, perc_dev, diff_percent


def __combine_small_bins(df_cont_pivot, other_threshold):
    """
    Combine bins that are too small in both control and treatment group into Constants.other_feature_cluster_name

    :input df_cont_pivot: contingency pivot table.
    """
    # if there are bins which are too small in both groups
    # then set them to the other group.
    grps = df_cont_pivot.columns
    df_cont_pivot.reset_index(inplace=True)
    df_cont_pivot.fillna(0, inplace=True)
    other_grp_1 = df_cont_pivot[grps[0]] / df_cont_pivot[grps[0]].sum() * 100.0 < other_threshold
    other_grp_2 = df_cont_pivot[grps[1]] / df_cont_pivot[grps[1]].sum() * 100.0 < other_threshold
    other_grp = other_grp_1 & other_grp_2
    df_cont_pivot.loc[other_grp, Constants.bin_column] = Constants.other_feature_cluster_name

    # Combine all the others by grouping by bin again
    df_cont_pivot = df_cont_pivot.groupby(Constants.bin_column).sum()
    df_cont_pivot.fillna(0, inplace=True)

    return df_cont_pivot, grps


def __get_contingency_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a contingency table for the dataframe.

        :input df: The dataframe.
        :returns The contingency table data frame with [Constants.bin_column ,'count', Constants.feature] columns
    """
    contingency_data_frame = pd.DataFrame()

    for c in data.columns:
        contingency_data_frame = contingency_data_frame.append(_get_feature_values_distribution(data[c]))

    return contingency_data_frame


def _get_feature_values_distribution(feature):
    """
        Get the count for each feature value.
    """

    # TODO: this would perform poorly when we have a numerical feature.
    distribution = pd.DataFrame(feature
                                .groupby(feature.values)
                                .agg('count')
                                .reset_index()
                                .rename(columns={"index": Constants.bin_column, feature.name: 'count'})
                                )

    distribution[Constants.feature] = feature.name

    return distribution
