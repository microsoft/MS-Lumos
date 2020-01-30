# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Code Constants
skip = 'skip'
sorting_type_delta_count = 'delta_count'
sorting_type_delta_percent = 'delta_percent'
# Reserved column names
group_column_name = "group"
other_feature_cluster_name = 'other'
control_group = 'control'
treatment_group = 'treatment'
# Computed columns
feature = 'feature'
resample = 'resample'
expected_failures = '# of Expected Failures in Treatment'
actual_failures = '# of Actual Failures in Treatment'
num_of_bins = 'num bins'
degree_of_freedom = 'dof'
percentage_deviation = 'Percentage Deviation'
mean_difference = 'Percent Difference'
mean_control = 'Percent Control'
control_percent = 'control_percent'
treatment_percent = 'treatment_percent'
mean_treatment = 'Percent Treatment'
percent_delta = '% Delta'
count_delta = 'Delta (Count)'
hazard_score = 'Hazard Score'
bin_column = 'bin'
p_value = 'P-Value'
is_stat_sig = 'Is Stat-Sig'
# Config parameters
small_bin_percent_threshold = 'small_bin_percent_threshold'
p_value_threshold = 'p_value'
sort_type = 'sort_type'
normalization_type = 'normalization_type'
metric_column = "metric_col"
invariant_columns = 'invariant_columns'
feature_columns = 'feature_columns'
results_dir = "results_dir"
add_is_null_column = 'add_null'
resample_threshold = 'resample_threshold'
decomposition_type = 'decomposition_type'
required_config_keys = [metric_column, invariant_columns, feature_columns, resample_threshold, results_dir,
                        p_value_threshold, decomposition_type, normalization_type, sort_type, add_is_null_column,
                        'apply_feature_target_metric_dependence_test']

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_file_name = 'mct.log'
