import datetime
import holidays
import pandas as pd
from .pandas_General_utils import *

# Time Granularity Function
def time_difference(dataframe, column_start, column_end, timedelta_value, timedelta_units, new_col_name):
    dataframe[new_col_name] = (dataframe[column_end] - dataframe[column_start])/pd.Timedelta(value = timedelta_value, unit = timedelta_units)
    return dataframe

def time_difference_row(row, column_start, column_end, timedelta_value, timedelta_units):
    return (pd.to_datetime(row[column_end]) - pd.to_datetime(row[column_start]))/pd.Timedelta(value = timedelta_value, unit = timedelta_units)


def get_date(dataframe, date_col_to_convert, new_col_name):
    dataframe[new_col_name] = dataframe[date_col_to_convert].dt.date
    dataframe[new_col_name] = pd.to_datetime(dataframe[new_col_name] )
    return dataframe

time_granularity_functions = {
    'time_difference' : time_difference,
    'get_date' : get_date
}

# Value Mapping Functions
def direct_mapping(dataframe, column, new_col_name, mapper, default_value = None):
    dataframe[new_col_name] = dataframe[column].apply(lambda x: mapper.get(x, default_value))
    return dataframe

def value_mapping(dataframe, ref_column, new_col_name, mapper, default_value = None):
    mapper_ = {v_:k  for k,v in mapper.items() for v_ in v} 
    dataframe[new_col_name] = dataframe[ref_column].apply(lambda x : mapper_.get(x, default_value))
    return dataframe

value_mapping_functions = {
    'direct_mapping' : direct_mapping,
    'value_mapping' : value_mapping
}

# Value Capping Functions
def filtered_capping(dataframe, cap_column, cap_value, cap_addtl_filters):
    data_filters = []
    for filter_ in cap_addtl_filters:
        func = filtering_functions[filter_['method']]
        boolvals = func(dataframe, **filter_['args'])
        data_filters.append(boolvals)
    filterer = pd.DataFrame(zip(*data_filters)).all(axis = 1)
    dataframe.loc[filterer.values, cap_column] = cap_value
    return dataframe

value_capping_functions = {
    'filtered_capping' : filtered_capping
}

# Date Related Functions
def get_weekends(dataframe, start_date_column, end_date_column):

    year_min = int(dataframe[start_date_column].dt.year.min())
    year_max = int(dataframe[end_date_column].dt.year.max())
    # create a list of all dates in the year
    all_dates = []
    for year in range(year_min, year_max + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    all_dates.append(datetime.date(year, month, day))
                except ValueError:
                    pass
    # filter out the weekends
    weekend_dates = [date for date in all_dates if date.weekday() in (5, 6)]
    weekend_dates = pd.Series(weekend_dates)
    weekend_dates = pd.to_datetime(weekend_dates)

    return weekend_dates

def is_weekend(dataframe, dataframe_weekend, date_col):
    dataframe['is_weekend'] = dataframe[date_col].apply(lambda x: 1 if x in dataframe_weekend.tolist() else 0)
    return dataframe

def get_holidays(dataframe, location_column, start_date_column, end_date_column):
    import datetime

    year_min = int(dataframe[start_date_column].dt.year.min())
    year_max = int(dataframe[end_date_column].dt.year.max())
    # create a list of all dates in the year
    dataframes = []
    for market in dataframe[location_column].unique():
        data = holidays.CountryHoliday(market, years = list(range(year_min, year_max+1))).items()
        market_holidays = pd.DataFrame(data, columns = ['date','holiday'])
        market_holidays['Date'] = pd.to_datetime(market_holidays['date'])
        market_holidays[location_column] = market
        dataframes.append(market_holidays)
    return pd.concat(dataframes, ignore_index=True)

def is_holiday(dataframe, dataframe_holiday, city_col, date_col, holiday_date_col, ):
    dataframe['is_holiday'] = dataframe.apply(lambda x: 1 if x[date_col] in dataframe_holiday[dataframe_holiday[city_col] == x[city_col]][holiday_date_col].tolist() else 0, axis = 1)
    return dataframe

def get_weekday(dataframe, date_col):
    dataframe['weekday'] = dataframe[date_col].dt.weekday
    return dataframe

def get_holiday_eve(dataframe, filters, date_col, holiday_col, holiday_lag = 1):
    data = pd.DataFrame()
    for filter_values in dataframe[filters].drop_duplicates().values:
        #City
        filtered_data = filter_iterable(dataframe, filters, filter_values)
        data_to_append = filtered_data.copy()
        filtered_data = filtered_data[filters+ [date_col, holiday_col]].drop_duplicates()
        filtered_data[f'holiday_eve_{holiday_lag}'] = filtered_data[holiday_col].shift(-1*holiday_lag)
        filtered_data.fillna({f'holiday_eve_{holiday_lag}' : 0}, inplace = True)
        filtered_data.set_index(filters + [date_col], inplace = True)
        holiday_lag_dct = filtered_data.to_dict()[f'holiday_eve_{holiday_lag}']
        data_to_append[f'holiday_eve_{holiday_lag}'] = data_to_append.apply(lambda x: holiday_lag_dct.get(tuple([x[col] for col
                                                                                                                 in filters + [date_col]]), 0),
                                                                            axis = 1)
        data = pd.concat([data, data_to_append])
    return data

def get_post_holiday(dataframe, filters, date_col, holiday_col, holiday_lag = 1):
    data = pd.DataFrame()
    for filter_values in dataframe[filters].drop_duplicates().values:
        #City
        filtered_data = filter_iterable(dataframe, filters, filter_values)
        data_to_append = filtered_data.copy()
        filtered_data = filtered_data[filters+ [date_col, holiday_col]].drop_duplicates()
        filtered_data[f'post_holiday_{holiday_lag}'] = filtered_data[holiday_col].shift(holiday_lag)
        filtered_data.fillna({f'post_holiday_{holiday_lag}' : 0}, inplace = True)
        filtered_data.set_index(filters + [date_col], inplace = True)
        holiday_lag_dct = filtered_data.to_dict()[f'post_holiday_{holiday_lag}']
        data_to_append[f'post_holiday_{holiday_lag}'] = data_to_append.apply(lambda x: holiday_lag_dct.get(tuple([x[col] for col
                                                                                                                 in filters + [date_col]]), 0),
                                                                            axis = 1)
        data = pd.concat([data, data_to_append])
    return data

def moving_average(dataframe, data_column, window_size = 3):
    moving_averages = []
    for i in range(len(dataframe) - window_size + 1):
        data_window = dataframe[data_column][i : i + window_size]
        data_avg = sum(data_window) / window_size
        moving_averages.append(data_avg)
    dataframe.loc[window_size-1:,f"{data_column}_MA{window_size}"] = moving_averages
    dataframe[f"{data_column}_MA{window_size}"].bfill(inplace=True)
    return dataframe

def get_full_range(dataframe, filter_columns, date_range_column):
    # Ignore `df`` input parameter for coherence with other methods
    full_df = pd.DataFrame()
    for filter_values in dataframe[filter_columns].drop_duplicates().values.tolist():
        # Apply the filter to the DataFrame
        df_group = filter_iterable(dataframe, filter_columns, filter_values)

        # create a Pandas series with daily frequency
        date_range = pd.date_range(
            start=df_group[date_range_column].min(), end=df_group[date_range_column].max(), freq="D"
        )
        temp_df = pd.DataFrame(columns=filter_columns)
        temp_df[date_range_column] = date_range
        temp_df = set_iterable(temp_df, filter_columns, filter_values)
        full_df = pd.concat([full_df, temp_df])
    full_df[date_range_column] = pd.to_datetime(full_df[date_range_column])
    return full_df

date_related_functions = {
    'get_weekends' : get_weekends,
    'is_weekend' : is_weekend,
    'get_holidays' : get_holidays,
    'is_holiday' : is_holiday,
    'get_weekday' : get_weekday,
    'get_holiday_eve' : get_holiday_eve,
    'get_post_holiday' : get_post_holiday,
    'get_full_range' : get_full_range
}

