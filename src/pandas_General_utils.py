import pandas as pd

def filter_iterable(dataframe, agent_feats, feat_list):
    # Initialize a filter condition
    filter_condition = True

    # Loop through the dynamic column names and values to build the filter condition
    for col, value in zip(agent_feats, feat_list):
        filter_condition &= dataframe[col] == value
    # Apply the filter to the DataFrame
    dataframe_temp = dataframe[filter_condition].copy()

    return dataframe_temp

def set_iterable(dataframe, agent_feats, feat_list):
    for col, value in zip(agent_feats, feat_list):
        dataframe[col] = value
    return dataframe

def greater_than_or_equal(dataframe, col, value, anti = False):
    if anti:
        return ~dataframe[col] >= value
    return dataframe[col] >= value

def greater_than(dataframe, col, value, anti = False):
    if anti:
        return ~dataframe[col] > value
    return dataframe[col] > value

def equal(dataframe, col, value, anti = False):
    if anti:
        return ~dataframe[col] == value
    return dataframe[col] == value

def less_than(dataframe, col, value, anti = False):
    if anti:
        return ~dataframe[col] < value
    return dataframe[col] < value

def less_than_or_equal(dataframe, col, value, anti = False):
    if anti:
        return ~dataframe[col] <= value
    return dataframe[col] <= value

def isin(dataframe, col, value_list, anti = False):
    if anti:
        return ~dataframe[col].isin(value_list)
    return dataframe[col].isin(value_list)

filters = {
    'greater_than_or_equal' : greater_than_or_equal,
    'greater_than' : greater_than,
    'equal' : equal,
    'less_than' : less_than,
    'less_than_or_equal' : less_than_or_equal,
    'isin' : isin
}

def filter_pandas(dataframe, masking_method , masking_args):
    func = filters[masking_method]
    # print(len(dataframe))
    mask = func(dataframe, **masking_args)
    # print(len(dataframe[mask]))
    return dataframe[mask]

filtering_functions = {
    'filter_iterable' : filter_iterable,
    'greater_than_or_equal' : greater_than_or_equal,
    'greater_than' : greater_than,
    'equal' : equal,
    'less_than' : less_than,
    'less_than_or_equal' : less_than_or_equal,
    'isin' : isin,
    'filter_pandas' : filter_pandas
}

def aggregation(dataframe, merge_on, aggregation_rules):
    return dataframe.groupby(merge_on).agg(**aggregation_rules).reset_index()

def subtract_columns(dataframe, col1, col2, new_col_name):
    dataframe[new_col_name] = dataframe.apply(lambda x: x[col1] - x[col2],axis = 1)
    return dataframe

def merge(dataframe1, dataframe2, merge_args, datetime_cols = None):
    if 'merge_on' in merge_args:
        merge_args['on'] = merge_args['merge_on']
        del merge_args['merge_on']
    dataframe_merge = pd.merge(dataframe1, dataframe2, **merge_args)
    if datetime_cols:
        for col in datetime_cols:
            dataframe_merge[col] = pd.to_datetime(dataframe_merge[col])
    return dataframe_merge

custom_pandas_functions = {
    'subtract_columns' : subtract_columns,
    'aggregation' : aggregation,
    'merge' : merge
}

def datetime_conversion(dataframe, col):
    dataframe[col] = pd.to_datetime(dataframe[col])
    return dataframe

def type_conversion(dataframe,col,type_):
    dataframe[col] = dataframe[col].astype(type_)
    return dataframe

datetime_pandas_functions = {
    'datetime_conversion' : datetime_conversion,
    'type_conversion' : type_conversion
}
    
    