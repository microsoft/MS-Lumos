# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

import mct.Constants as Constants


def get_is_null_column_name(col):
    return col + "_is_null"


def add_group_columns(control: pd.DataFrame, treatment: pd.DataFrame,
                      group_column_name: str = Constants.group_column_name, force: bool = False):
    if force or (group_column_name not in control.columns):
        control[group_column_name] = Constants.control_group
    if force or (group_column_name not in treatment.columns):
        treatment[group_column_name] = Constants.treatment_group
    return


def merge_control_treatment(control: pd.DataFrame,
                            treatment: pd.DataFrame,
                            group_column_name: str = Constants.group_column_name) -> pd.DataFrame:
    add_group_columns(control, treatment, group_column_name=group_column_name, force=True)
    df = control.append(treatment)
    df.reset_index(drop=True, inplace=True)
    return df
