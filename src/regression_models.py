import numpy as np
import pandas as pd
import xgboost as xgb
import holidays

from src.python_utils import *
from src.pandas_General_utils import *
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Baseline Models
def moving_average_model(training_data,
                   test_data,
                   data_column,
                   window_size,
                   forecast_size = 0,
                   rounding_rule = np.ceil,
                   cols_to_copy = ['City', 'Market', 'Role', 'Shift Date'],):
    # Make copies
    training_data = training_data.copy()
    test_data = test_data.copy()
    
    # Training
    moving_averages = []
    for i in range(len(training_data) - window_size + 1):
        data_window = training_data[data_column][i : i + window_size]
        data_avg = sum(data_window)/window_size
        moving_averages.append(data_avg)
    training_data[f"{data_column}_MA{window_size}"] = None
    
    mapping_indexes = training_data.iloc[window_size-1:].index
    training_data.loc[mapping_indexes, f"{data_column}_MA{window_size}"] = moving_averages
    training_data[f"{data_column}_MA{window_size}"].bfill(inplace = True)
    training_data[f"{data_column}_MA{window_size}_rounded"] = rounding_rule(training_data[f"{data_column}_MA{window_size}"])
    
    # Test Data
    overlap_data = training_data.iloc[-window_size:].copy()
    overlap_data['from_training'] = True
    
    i = 0
    for row_id, row_info in test_data.iterrows():
        data_window = overlap_data[data_column][i : i + window_size]
        data_avg = sum(data_window)/window_size
        
        row_info['from_training'] = False
        row_info[f"{data_column}_MA{window_size}"] = data_avg
        row_info[f"{data_column}_MA{window_size}_rounded"] = rounding_rule(data_avg)
        
        overlap_data = pd.concat([overlap_data, pd.DataFrame(row_info).T], axis = "rows")
        i += 1
        
    # Inference Data
    if not forecast_size:
        return (training_data, 
                overlap_data[overlap_data.from_training == False], 
                None)
    elif forecast_size > 0:
        inference_data = overlap_data.iloc[-window_size:].copy()
        inference_data['from_training'] = True
        
        for i in range(forecast_size):
            new_row = inference_data.iloc[-1].copy()
            last_date = new_row['Shift Date']
            new_row[new_row.index.difference(cols_to_copy)] = None
            new_row['Shift Date'] = new_row['Shift Date'] + pd.Timedelta(days=7)
            new_row['Year'] = new_row['Shift Date'].year
            
            data_window = inference_data[data_column][i : i + window_size] # Checks Actual Headcount
            data_avg = sum(data_window) / window_size
            
            new_row[f"{data_column}_MA{window_size}"] = data_avg
            new_row[f"{data_column}_MA{window_size}_rounded"] = rounding_rule(data_avg)
            
            # Create a date range between the start and end dates
            date_range = pd.date_range(start=last_date, end=new_row['Shift Date'] )

            # Fetch the holidays for a specific country (e.g., 'US')
            city_holidays = holidays.CountryHoliday(new_row['City'])
            
            # Set Value for Holiday Counts
            new_row['is_holiday'] = sum(1 for date in date_range if date in city_holidays)
            
            new_row[data_column] = rounding_rule(data_avg)
            new_row['from_training'] = False
            inference_data = pd.concat([inference_data, pd.DataFrame(new_row).T], axis = 'rows')
        return (training_data, 
                overlap_data[overlap_data.from_training == False], 
                inference_data[inference_data.from_training == False])
    else:
        raise(ValueError, "Invalid Data Value for Forecast Size - must be either a positive value or zero.")
            
        
def persistence_model(training_data,
                      test_data,
                      data_column,
                      movements="monthly", # week
                      forecast_size = 0,
                      cols_to_copy = ['City', 'Market', 'Role', 'Shift Date'],):
    # Make copies
    training_data = training_data.copy()
    test_data = test_data.copy()
    
    training_data['data_source'] = 'training'
    test_data['data_source'] = 'testing'
    
    full_data = pd.concat([training_data, test_data], axis = 'rows')
    full_data.sort_values('Shift Date', inplace = True)
    full_data['Month'] = full_data['Shift Date'].dt.month
    
    if movements == 'monthly':
        persistence_lookup = full_data.groupby(['Year','Month'])[data_column].agg('last').reset_index()
        
        persistence_data = pd.DataFrame()
        for row_index, row_info in full_data.iterrows():
            year = row_info['Year']
            month = row_info['Month']
            if ((month == persistence_lookup.Month.iloc[0]) & (year == persistence_lookup.Year.iloc[0])):
                row_info[f'Persisted {data_column}'] = persistence_lookup[data_column].iloc[0]
            elif month == 1:
                row_info[f"Persisted {data_column}"] = filter_iterable(persistence_lookup, ['Year','Month'], [year-1, 12]).iloc[0][data_column]
            else:
                row_info[f"Persisted {data_column}"] = filter_iterable(persistence_lookup, ['Year','Month'], [year, month-1]).iloc[0][data_column]
            persistence_data = pd.concat([persistence_data, pd.DataFrame(row_info).T], axis = 'rows')
    elif movements == 'weekly':
        persistence_data = full_data.copy()
        persistence_data[f"Persisted {data_column}"] = persistence_data[data_column].shift(1).bfill()
        
    if not forecast_size:
        return (persistence_data[persistence_data.data_source == 'training'],
                persistence_data[persistence_data.data_source == 'testing'],
                None)
    else:
        for i in range(forecast_size):
            new_row = persistence_data.iloc[-1].copy()
            last_date = new_row['Shift Date']
            new_row[new_row.index.difference(cols_to_copy)] = None
            new_row['Shift Date'] = new_row['Shift Date'] + pd.Timedelta(days=7)
            new_row['Year'] = new_row['Shift Date'].year
            new_row['Month'] = new_row['Shift Date'].month
            
            if movements == 'monthly':
                if month == 1:
                    new_row[f"Persisted {data_column}"] = filter_iterable(persistence_data, ['Year','Month'], [year-1, 12]).iloc[-1][data_column]
                else:
                    new_row[f"Persisted {data_column}"] = filter_iterable(persistence_data, ['Year','Month'], [year, month-1]).iloc[-1][data_column]
            elif movements == 'weekly':
                new_row[f'Persisted {data_column}'] = persistence_data[data_column].iloc[-1]
            
            # Create a date range between the start and end dates
            date_range = pd.date_range(start=last_date, end=new_row['Shift Date'] )

            # Fetch the holidays for a specific country (e.g., 'US')
            city_holidays = holidays.CountryHoliday(new_row['City'])
            
            # Set Value for Holiday Counts
            new_row['is_holiday'] = sum(1 for date in date_range if date in city_holidays)
            new_row['data_source'] = 'inference'
            persistence_data = pd.concat([persistence_data, pd.DataFrame(new_row).T], axis = 'rows')
            
        return (persistence_data[persistence_data.data_source == 'training'],
                persistence_data[persistence_data.data_source == 'testing'],
                persistence_data[persistence_data.data_source == 'inference'])
        
def expanding_window_model(training_data,
                     test_data,
                     data_column,
                     method = np.mean,
                     forecast_size = 0,
                     rounding_rule = np.ceil,
                     cols_to_copy = ['City', 'Market', 'Role', 'Shift Date'],):
    # Make copies
    training_data = training_data.copy()
    test_data = test_data.copy()
    
    training_data['data_source'] = 'training'
    test_data['data_source'] = 'testing'
    
    full_data = pd.concat([training_data, test_data], axis = 'rows')
    full_data.sort_values(by='Shift Date', inplace = True)
    
    stat_values = []
    for i in range(1,len(full_data)+1):
        data_values = full_data.iloc[:i][data_column]
        stat_values.append(rounding_rule(method(data_values)))
    full_data[f'Expanded {method.__name__}_{data_column}'] = stat_values
    
    if not forecast_size:
        return(full_data[full_data.data_source == 'training'],
               full_data[full_data.data_source == 'testing'],
               None)
    else:
        for i in range(forecast_size):
            new_row = full_data.iloc[-1].copy()
            last_date = new_row['Shift Date']
            new_row[new_row.index.difference(cols_to_copy)] = None
            new_row['Shift Date'] = new_row['Shift Date'] + pd.Timedelta(days=7)
            
            new_row[f'Expanded {method.__name__}_{data_column}'] = rounding_rule(method(full_data[data_column]))
            new_row[data_column] = rounding_rule(method(full_data[data_column]))
        
            # Create a date range between the start and end dates
            date_range = pd.date_range(start=last_date, end=new_row['Shift Date'] )

            # Fetch the holidays for a specific country (e.g., 'US')
            city_holidays = holidays.CountryHoliday(new_row['City'])
            
            # Set Value for Holiday Counts
            new_row['is_holiday'] = sum(1 for date in date_range if date in city_holidays)
            new_row['data_source'] = 'inference'
            full_data = pd.concat([full_data, pd.DataFrame(new_row).T], axis = 'rows')
        return(full_data[full_data.data_source == 'training'],
               full_data[full_data.data_source == 'testing'],
               full_data[full_data.data_source == 'inference'])
    
    
def train_XGBRegressor(dataframe,
                       config,
                       forecasting_on,
                       forecasting_what,
                       addtl_columns = [],
                       objective = "reg:squarederror",
                       n_estimators = 1000,
                       max_depth = 50,
                       min_child_weight = 1,
                       learning_rate = 0.0125,
                       reg_lambda = 0.3,
                       reg_alpha = 0.3):
    
    model = xgb.XGBRegressor(objective = objective,
                            n_estimators = n_estimators,
                            max_depth = max_depth,
                            min_child_weight = min_child_weight,
                            learning_rate = learning_rate,
                            reg_lambda = reg_lambda,
                            reg_alpha = reg_alpha)
    
    model.fit(dataframe[config['modelSettings'][forecasting_on]["Exog_columns"] +
                        config['modelSettings'][forecasting_on][forecasting_what]["Addtl_Exog_columns"] +
                        addtl_columns],
              dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']])
    return model
    
def train_ARIMA(dataframe,
                config,
                forecasting_on,
                forecasting_what,
                addtl_columns = [],
                p = 1,
                d = 1,
                q = 1,
                trend = 'c',
                freq = 'W-Fri'
                ):
    dataframe = dataframe.set_index('Shift Date')
    model = ARIMA(endog=dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']],
                  exog = dataframe[config['modelSettings'][forecasting_on]['Exog_columns'] + addtl_columns],
                  order = (p,d,q),
                  trend = trend,
                  freq=freq)
    return model.fit()


def test_BaselineRegressor(dataframe, actual_col, pred_col, metric = 'rmse'):
    if metric == 'mse':
        mse = mean_squared_error(dataframe[pred_col], dataframe[actual_col])
        return mse
    
    elif metric == 'rmse':
        mse = mean_squared_error(dataframe[pred_col], dataframe[actual_col])
        rmse = np.sqrt(mse)
        return rmse
    
def test_ARIMARegressor(model, config, dataframe, addtl_columns = [], forecasting_on = None, forecasting_what = None, metric = 'rmse'):
    dataframe = dataframe.set_index('Shift Date')
    y_pred = model.forecast(steps = len(dataframe),
                            exog = dataframe[config['modelSettings'][forecasting_on]["Exog_columns"] +
                                     config['modelSettings'][forecasting_on][forecasting_what]["Addtl_Exog_columns"] + 
                                     addtl_columns])

    if metric == 'mse':
        mse = mean_squared_error(y_pred, dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']])
        return y_pred, mse
    
    elif metric == 'rmse':
        mse = mean_squared_error(y_pred, dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']])
        rmse = np.sqrt(mse)
        return y_pred, rmse
    
def test_ModelRegressor(model, config, dataframe, addtl_columns = [], forecasting_on = None, forecasting_what = None, metric = 'mse'):
    y_pred = model.predict(dataframe[config['modelSettings'][forecasting_on]["Exog_columns"] +
                                     config['modelSettings'][forecasting_on][forecasting_what]["Addtl_Exog_columns"] +
                                     addtl_columns])
    
    if metric == 'mse':
        mse = mean_squared_error(y_pred, dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']])
        return y_pred, mse
    
    elif metric == 'rmse':
        mse = mean_squared_error(y_pred, dataframe[config['modelSettings'][forecasting_on][forecasting_what]['Endog_column']])
        rmse = np.sqrt(mse)
        return y_pred, rmse
    
def infer_ModelRegressor(model, config, dataframe, addtl_columns = [], forecasting_on = None, forecasting_what = None, metric = 'rmse'):
    y_pred = model.predict(dataframe[config['modelSettings'][forecasting_on]["Exog_columns"] +
                                     config['modelSettings'][forecasting_on][forecasting_what]["Addtl_Exog_columns"] + 
                                     addtl_columns])
    return y_pred

def infer_ARIMARegressor(model, config, dataframe, addtl_columns = [], forecasting_on = None, forecasting_what = None, metric = 'rmse'):
    dataframe = dataframe.set_index('Shift Date')
    y_pred = model.forecast(steps = len(dataframe),
                            exog = dataframe[config['modelSettings'][forecasting_on]["Exog_columns"] +
                                     config['modelSettings'][forecasting_on][forecasting_what]["Addtl_Exog_columns"] + 
                                     addtl_columns])
    return y_pred