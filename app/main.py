# Import Libraries
import toml
import yaml
import math
import datetime
import holidays
import numpy as np
import pandas as pd
from yaml.loader import SafeLoader

# Import Streamlit and Data Viz Libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth

# Import Custom Modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.python_utils import *
from src.streamlit_utils import *
from src.regression_models import *
from src.pandas_General_utils import *
from src.pandas_FeatEngineering_utils import *

from app.historical_capacity_planner import *
from app.forecast_capacity_planner import *

# Import Pages
from sidebar import *

# Restrict Warnings to Non-General Warnings
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Specify Project Root Folder
project_root = str(Path(__file__).parent.parent)

###### Authenticator ######
# Load in Secrets for Streamlit Authenticator
secrets_data_path = Path(f"{project_root}/config/secrets.yaml")
with open(secrets_data_path.resolve()) as file:
    config = yaml.load(file, Loader=SafeLoader)
    
# Page config
st.set_page_config(page_title="Capacity Planning Simulator", layout="wide")

# Initialize Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Load in Authenticator
authenticator.login('Login', 'main')
    
# Initialize Page if Succesful Login
if st.session_state['authentication_status']:
    
    # Add in LogOut Button
    authenticator.logout("Logout", "main", key="unique_key")
    st.write("Welcome to Meta Capacity Planner user!")
    
    # Load in Configuration Files
    settings, config, instructions, readme = load_config(
        "config_settings.toml", "config_streamlit.toml", 
        "config_instructions.toml", "config_readme.toml"
    )
    
    # Load in Activity Log
    activity_log_data_path = Path(get_project_root()) / settings['filePaths']['activity_log']
    activity_log = pd.read_csv(activity_log_data_path.resolve())
    activity_log['Shift Date'] = pd.to_datetime(activity_log['Shift Date'])
    
    # Remove 2023 September Data due to Incompleteness - Remove Once Updated Dataset
    activity_log['Month'] = activity_log['Shift Date'].dt.month
    activity_log = activity_log[~((activity_log["Year"] == 2023) & (activity_log['Month'] == 9))]
        
    # Load Start and End Lookup
    start_end_lookup_data_path = Path(get_project_root()) / settings['filePaths']['start_end_lookup']
    start_end_lookup = pd.read_csv(start_end_lookup_data_path.resolve())
    for dcol in ['start','end','max_train_date']:
        start_end_lookup[dcol] = pd.to_datetime(start_end_lookup[dcol])  

    # Remove Double Entries
    misassigned = activity_log[(activity_log['Year'] == 2022) & (activity_log['Shift Date'] == datetime.datetime(2023,1,6))].index
    activity_log.drop(misassigned, axis = 'index', inplace = True)
    activity_log['ActiveCount'] = activity_log['ActiveCount'].astype(float)
    
    # Start WebPage
    
    ## Sidebar

    ### Header
    st.sidebar.image(load_image("logo.png"), use_column_width = True)
    st.sidebar.subheader("Unlocking Efficiency within the Metaverse", divider = "blue")
    st.sidebar.write("")
    
    ### LOB Filter
    st.sidebar.title("Level of Business")

    #### Data Filters
    with st.sidebar.container():
        
        # Load in LOB Filters
        city, market, role = get_lob_filters(activity_log)
        
        # Apply Data Filters
        filtered_data = filter_iterable(activity_log, ['City','Market','Role'], [city,market,role])
        
        lob_col1, lob_col2= st.columns(2)
        
        with st.expander("Metric Targets", expanded = False):
            Util_Target = st.number_input("Utilization Target", 
                                          min_value = 0.0,
                                          max_value = 1.0,
                                          step = 0.001,
                                          format ="%1.3f",
                                          value=settings['simulator']['Targets']['Utilization'])
            Util_Client_Target = st.number_input("Client Utilization Target", 
                                          min_value = 0.0,
                                          max_value = 1.0,
                                          step = 0.001,
                                          format ="%1.3f",
                                          value=settings['simulator']['Targets']['Utilization_Client'])
            OOO_Target = st.number_input("Out-of-Office Shrinkage Target", 
                                          min_value = 0.0,
                                          max_value = 1.0,
                                          step = 0.001,
                                          format ="%1.3f",
                                         value=settings['simulator']['Targets']['OOO'])
            WIO_Target = st.number_input("Within-Office Shrinkage Target", 
                                          min_value = 0.0,
                                          max_value = 1.0,
                                          step = 0.001,
                                          format ="%1.3f",
                                         value=settings['simulator']['Targets']['WIO'])
            
            settings['simulator']['Targets']['Utilization'] = Util_Target
            settings['simulator']['Targets']['Utilization_Client'] = Util_Client_Target
            settings['simulator']['Targets']['OOO'] = OOO_Target
            settings['simulator']['Targets']['WIO'] = WIO_Target
            
            
        
    # Set Default Border Training Date - TO-DO : Make it config-based
    border_training_date = datetime.datetime(2023, 7, 1)

    df_training = filtered_data [filtered_data ['Shift Date'] <= border_training_date]
    df_planning = filtered_data [filtered_data ['Shift Date'] > border_training_date]
    df_inference = pd.DataFrame(columns = filtered_data.columns)
    
    ## Info on the App
    with st.expander("A Capacity Planning Simulator for FBCO", expanded = False):
        st.write(readme["app"]["app_intro"]) # TO-DO
        st.write("")
    st.write("")
    
    # Launch Simulator
    if st.checkbox(
        "Launch Webapp Functionalities",
        value = False,
        help = readme['tooltips']['launch_forecast'] #TO-DO
    ):
        capacity_planner_tab, historical_data_tab = st.tabs(["📅 Capacity Planner",
                                                             "📒 Historical Data"])
        with capacity_planner_tab:
            
            lookbf_container = st.container()
            lookback_col, lookforward_col = lookbf_container.columns(2)
            
            lookback_months = lookback_col.selectbox(
                "Weeks Lookback",
                options = range(1,13),
                index = 3
            )
            
            lookforward_months = lookforward_col.selectbox(
                "Weeks Lookforward",
                options = range(1,13),
                index = 3
            )
            
            with st.expander("Headcount Forecasting Configuration",expanded=False):
                
                #Headcount Calculation
                hc_forecasting = st.radio(
                                    "Headcount Forecasting Method",
                                    ["3-Month Moving Average", "Persistence Model", "Expanding Mean Model"],
                                    captions = ["Use Moving Average to Generate Forecasts for Headcount",
                                                "Retain Previous Month Headcounts",
                                                "Get the average of ALL historical heacounts"],
                                    key = f"Headcount Method",
                                    horizontal= True)
                
                if hc_forecasting == "3-Month Moving Average":
                    hc_training, hc_test, hc_inference = moving_average_model(df_training,
                                                                        df_planning,
                                                                        'ActiveCount',
                                                                        12,
                                                                        forecast_size = lookforward_months,
                                                                        rounding_rule = np.ceil)
                    actual_hc_col = 'ActiveCount'
                    pred_hc_col = 'ActiveCount_MA12_rounded'
                    
                elif hc_forecasting == "Persistence Model":
                    hc_training, hc_test, hc_inference = persistence_model(df_training,
                                                                        df_planning,
                                                                        'ActiveCount',
                                                                        forecast_size = lookforward_months)
                    actual_hc_col = 'ActiveCount'
                    pred_hc_col = 'Persisted ActiveCount'
                    
                else:
                    hc_training, hc_test, hc_inference = expanding_window_model(df_training,
                                                                        df_planning,
                                                                        'ActiveCount',
                                                                        forecast_size = lookforward_months,
                                                                        rounding_rule = np.ceil)
                    actual_hc_col = 'ActiveCount'
                    pred_hc_col = 'Expanded mean_ActiveCount'
                
                
                hc_training['data_source'] = 'training'
                hc_test['data_source'] = 'testing'
                hc_inference['data_source'] = 'inference'
                    
                hc_inference['Required FTE'] = hc_test.iloc[-1]['Required FTE']
                hc_inference[actual_hc_col] = hc_inference[pred_hc_col]
                
                error_score = test_BaselineRegressor(hc_test, actual_hc_col, pred_hc_col)
                st.warning(f"Root Mean Square Error of {hc_forecasting} Forecast: ±{error_score:.2f} Headcount", icon="⚠️")
                
                
                hc_inference['Excess FTE'] = hc_inference[actual_hc_col] - hc_inference['Required FTE']
                st.caption("✏️ Editable Data")
                hc_inference_edited = st.data_editor(hc_inference[['Shift Date',
                                                                    pred_hc_col,
                                                                   'Required FTE', "Excess FTE", 'is_holiday']].rename(columns={pred_hc_col : 'Active Unique Agents per Week',
                                                                                                                                'is_holiday' : '# of Holidays'}).set_index('Shift Date').T,
                                                    key = f"hc_edited")
                
                hc_inference_edited = hc_inference_edited.T.reset_index().rename(columns={'Active Unique Agents per Week' : pred_hc_col,
                                                                                          '# of Holidays':'is_holiday'})

                hc_inference_edited['Required FTE'] = hc_inference_edited['Required FTE'].astype(float)
                hc_inference_edited["Excess FTE"] = hc_inference_edited['Excess FTE'].astype(float)
                hc_inference_edited['is_holiday'] = hc_inference_edited['is_holiday'].astype(bool)
                
                hc_training['data_source'] = 'training'
                hc_test['data_source'] = 'testing'
                hc_inference_edited['data_source'] = 'inference'
                
                submit_hc_forecasting = st.checkbox("Proceed to Next Step", help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.")
                
            if submit_hc_forecasting:
                with st.expander("Annual Leave Forecasting Configuration"):
                    AL_forecasting = st.radio(
                    "Annual Leave Forecasting Method",
                    ["Regression","3-Month Moving Average", "Persistence Model", "Expanding Mean Model"],
                    captions = ["Use Gradient Boosting Regression",
                                "Use Moving Average for Leaves",
                                "Retain Previous Month Leaves",
                                "Get the average of ALL Historical Leaves"],
                    key = f"Annual Leaves Method",
                    horizontal= True)
                    
                    forecasting_df = pd.concat([hc_training, hc_test, hc_inference_edited], ignore_index=True)
                    forecasting_df['Required FTE'] = forecasting_df['Required FTE'].astype(float)
                    forecasting_df['is_holiday'] = forecasting_df['is_holiday'].astype(float)
                    forecasting_df[pred_hc_col] = forecasting_df[pred_hc_col].astype(float)
                    forecasting_df['Excess FTE'] = forecasting_df[pred_hc_col] - forecasting_df['Required FTE']
                    
                    if AL_forecasting == "Regression":

                        forecasting_df['Annual Leaves Lag 1'] = forecasting_df[settings['modelSettings']['Leaves']['Annual']['Endog_column']].shift(1).fillna(0)
                        forecasting_df['Annual Leaves Lag 2'] = forecasting_df[settings['modelSettings']['Leaves']['Annual']['Endog_column']].shift(2).fillna(0)

                        Annual_Leaves_Model = train_XGBRegressor(forecasting_df[forecasting_df.data_source == 'training'], 
                                                                 settings, 
                                                                 addtl_columns = [pred_hc_col],
                                                                 forecasting_on="Leaves", 
                                                                 forecasting_what="Annual",
                                                                 **settings['modelSettings']["Leaves"]["Annual"]['Parameters'])
                        annual_test_forecasts, annual_testing_accuracy = test_ModelRegressor(Annual_Leaves_Model, 
                                                                                             settings, 
                                                                                             forecasting_df[forecasting_df.data_source == 'testing'], 
                                                                                             addtl_columns = [pred_hc_col],
                                                                                             forecasting_on="Leaves", forecasting_what="Annual", metric = "rmse")
                        st.warning(f"Root Mean Square Error of Regression Model on Test Set: ±{annual_testing_accuracy:.2f} Leaves", icon="⚠️")
                        
                        lag_leave1 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-1]['annual_leave_value']
                        lag_leave2 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-2]['annual_leave_value']
                        AL_inference = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop = True)
                        for ind,row in AL_inference.iterrows():
                            row['Annual Leaves Lag 1'] = lag_leave1
                            row['Annual Leaves Lag 2'] = lag_leave2

                            row_df = pd.DataFrame(row).T
                            row_df['Required FTE'] = row_df['Required FTE'].astype(float)
                            row_df["Excess FTE"] = row_df['Excess FTE'].astype(float)
                            row_df['is_holiday'] = row_df['is_holiday'].astype(float)
                            row_df[pred_hc_col] = row_df[pred_hc_col].astype(float)
                            row_df['Annual Leaves Lag 1'] = row_df['Annual Leaves Lag 1'].astype(float)
                            row_df['Annual Leaves Lag 2'] = row_df['Annual Leaves Lag 2'].astype(float)
                            AL_forecast = infer_ModelRegressor(Annual_Leaves_Model, 
                                                                settings, 
                                                                row_df, 
                                                                addtl_columns = [pred_hc_col],
                                                                forecasting_on="Leaves", 
                                                                forecasting_what="Annual", 
                                                                metric = "rmse")
                            AL_inference.loc[ind,'annual_leave_value'] = round_to_nearest_half(AL_forecast[0])
                            AL_inference.loc[ind,'Annual Leaves Lag 1'] = lag_leave1
                            AL_inference.loc[ind,'Annual Leaves Lag 2'] = lag_leave2
                            
                            lag_leave2 = lag_leave1
                            lag_leave1 = round_to_nearest_half(AL_forecast[0])
                        
                        actual_AL_col = 'annual_leave_value'
                        pred_AL_col = 'annual_leave_value'
                        AL_training = forecasting_df[forecasting_df.data_source == 'training']
                        AL_test = forecasting_df[forecasting_df.data_source == 'testing']
                        
                    elif AL_forecasting == "3-Month Moving Average":
                        AL_training, AL_test, AL_inference = moving_average_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                        forecasting_df[forecasting_df.data_source == 'testing'],
                                                                        'annual_leave_value',
                                                                        12,
                                                                        forecast_size = lookforward_months,
                                                                        rounding_rule = np.ceil)
                        actual_AL_col = 'annual_leave_value'
                        pred_AL_col = 'annual_leave_value_MA12_rounded'
                        
                    elif AL_forecasting == "Persistence Method":
                        AL_training, AL_test, AL_inference = persistence_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                            forecasting_df[forecasting_df.data_source == 'testing'],
                                                                            'annual_leave_value',
                                                                            forecast_size = lookforward_months)
                        actual_AL_col = 'annual_leave_value'
                        pred_AL_col = 'Persisted annual_leave_value'
                    else:
                        AL_training, AL_test, AL_inference = expanding_window_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                    forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                    'annual_leave_value',
                                                                                    forecast_size = lookforward_months)
                        actual_AL_col = 'annual_leave_value'
                        pred_AL_col = 'Expanded mean_annual_leave_value'

                    if AL_forecasting != "Regression":
                        AL_inference = pd.concat([hc_inference_edited.reset_index(drop=True),
                                                  AL_inference[[pred_AL_col]].reset_index(drop=True)], 
                                                 axis = 'columns')
                        
                        AL_inference[actual_AL_col] =  AL_inference[pred_AL_col]
                        
                        
                        AL_inference[pred_hc_col] = AL_inference[pred_hc_col].astype(int)
                        AL_inference['Required FTE'] = AL_inference['Required FTE'].astype(int)
                        AL_inference['Excess FTE'] = AL_inference['Excess FTE'].astype(int)
                        AL_inference['is_holiday'] = AL_inference['is_holiday'].astype(int)

                        error_score = test_BaselineRegressor(AL_test, actual_AL_col, pred_AL_col)
                        st.warning(f"Root Mean Square Error of {AL_forecasting} Forecast: ±{error_score:.2f} Leaves", icon="⚠️")   
                    
                    st.caption("Current Data")
                    st.dataframe(AL_inference[['Shift Date',
                                                pred_hc_col,
                                                'Required FTE',
                                                'Excess FTE',
                                                'is_holiday']].rename(columns={pred_hc_col : 'Active Unique Agents per Week',
                                                                                'is_holiday' : '# of Holidays'}).set_index('Shift Date').T,
                    )
                    
                    st.caption("✏️ Editable Data")
                    AL_inference_edited = st.data_editor(AL_inference[['Shift Date',
                                                                       actual_AL_col]].rename(columns={actual_AL_col : 'Annual Leave Count'}).set_index('Shift Date').T,
                                                key = f"AL_edited")
                    
                    
                    old_leave_values = list(forecasting_df[forecasting_df.data_source != 'inference'][actual_AL_col])
                    forecasting_df['Annual Leave Count'] = old_leave_values + list(AL_inference_edited.T['Annual Leave Count'])
                    
                    submit_AL_forecasting = st.checkbox("Proceed to Next Step", 
                                                        help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.",
                                                        key = 'AL')
            if submit_hc_forecasting:
                if submit_AL_forecasting:
                    with st.expander("TOIL Leave Forecasting Configuration"):           
                        
                        #TOIL Leave Forecasting
                        TOIL_forecasting = st.radio(
                                            "TOI Leave Forecasting Method",
                                            ["Regression","3-Month Moving Average", "Persistence Model", "Expanding Mean Model"],
                                            captions = ["Use Gradient Boosting Regression",
                                                        "Use Moving Average for Leaves",
                                                        "Retain Previous Month Leaves",
                                                        "Get the average of ALL Historical Leaves"],
                                            key = f"TOI Leaves Method",
                                            horizontal= True)
                        
                        if TOIL_forecasting == "Regression":

                                forecasting_df['TOIL Leaves Lag 1'] = forecasting_df[settings['modelSettings']['Leaves']['TOIL']['Endog_column']].shift(1).fillna(0)
                                forecasting_df['TOIL Leaves Lag 2'] = forecasting_df[settings['modelSettings']['Leaves']['TOIL']['Endog_column']].shift(2).fillna(0)

                                TOI_Leaves_Model = train_XGBRegressor(forecasting_df[forecasting_df.data_source == 'training'],
                                                                    settings, 
                                                                    addtl_columns=[pred_hc_col],
                                                                    forecasting_on="Leaves",
                                                                    forecasting_what="TOIL",
                                                                        **settings['modelSettings']["Leaves"]["TOIL"]['Parameters'])
                                toil_test_forecasts, toil_testing_accuracy = test_ModelRegressor(TOI_Leaves_Model, 
                                                                                                settings, 
                                                                                                forecasting_df[forecasting_df.data_source == 'testing'], 
                                                                                                addtl_columns=[pred_hc_col],
                                                                                                forecasting_on="Leaves", forecasting_what="TOIL", 
                                                                                                metric = "rmse")
                                st.warning(f"Root Mean Square Error of Regression Model on Test Set: ±{toil_testing_accuracy:.2f} Leaves", icon="⚠️")
                                
                                lag_leave1 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-1]['toil_leave_value']
                                lag_leave2 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-2]['toil_leave_value']
                                TOIL_inference = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop = True)
                                for ind,row in TOIL_inference.iterrows():
                                    row['TOIL Leaves Lag 1'] = lag_leave1
                                    row['TOIL Leaves Lag 2'] = lag_leave2
                                    
                                    row_df = pd.DataFrame(row).T
                                    row_df[pred_hc_col] = row_df[pred_hc_col].astype(float)
                                    row_df['Required FTE'] = row_df['Required FTE'].astype(float)
                                    row_df["Excess FTE"] = row_df['Excess FTE'].astype(float)
                                    row_df['is_holiday'] = row_df['is_holiday'].astype(float)
                                    row_df['TOIL Leaves Lag 1'] = row_df['TOIL Leaves Lag 1'].astype(float)
                                    row_df['TOIL Leaves Lag 2'] = row_df['TOIL Leaves Lag 2'].astype(float)
                                    
                                    TOIL_forecast = infer_ModelRegressor(TOI_Leaves_Model, 
                                                                        settings, 
                                                                        row_df, 
                                                                        addtl_columns=[pred_hc_col],
                                                                        forecasting_on="Leaves", 
                                                                        forecasting_what="TOIL", 
                                                                        metric = "rmse")
                                    TOIL_inference.loc[ind,'toil_leave_value'] = round_to_nearest_half(TOIL_forecast[0])
                                    TOIL_inference.loc[ind,'TOIL Leaves Lag 1'] = lag_leave1
                                    TOIL_inference.loc[ind,'TOIL Leaves Lag 2'] = lag_leave2
                                    
                                    lag_leave2 = lag_leave1
                                    lag_leave1 = round_to_nearest_half(TOIL_forecast[0])
                                
                                actual_TOIL_col = 'toil_leave_value'
                                pred_TOIL_col = 'toil_leave_value'
                                TOIL_training = forecasting_df[forecasting_df.data_source == 'training']
                                TOIL_test = forecasting_df[forecasting_df.data_source == 'testing']
                                
                        elif TOIL_forecasting == "3-Month Moving Average":
                            TOIL_training, TOIL_test, TOIL_inference = moving_average_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                            forecasting_df[forecasting_df.data_source == 'testing'],
                                                                            'toil_leave_value',
                                                                            12,
                                                                            forecast_size = lookforward_months,
                                                                            rounding_rule = np.ceil)
                            actual_TOIL_col = 'toil_leave_value'
                            pred_TOIL_col = 'toil_leave_value_MA12_rounded'
                        elif TOIL_forecasting == "Persistence Method":
                            TOIL_training, TOIL_test, TOIL_inference = persistence_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                'toil_leave_value',
                                                                                forecast_size=lookforward_months)
                            actual_TOIL_col = 'toil_leave_value'
                            pred_TOIL_col = 'Persisted toil_leave_value'
                        else:
                            TOIL_training, TOIL_test, TOIL_inference = expanding_window_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                        forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                        'toil_leave_value',
                                                                                        forecast_size=lookforward_months)
                            actual_TOIL_col = 'toil_leave_value'
                            pred_TOIL_col = 'Expanded mean_toil_leave_value'

                        if TOIL_forecasting != "Regression":
                            
                            TOIL_inference = pd.concat([hc_inference_edited.reset_index(drop=True),
                                                        TOIL_inference[[pred_TOIL_col]].reset_index(drop=True)], 
                            axis = 'columns')

                            TOIL_inference[actual_TOIL_col] =  TOIL_inference[pred_TOIL_col]


                            TOIL_inference[pred_hc_col] = TOIL_inference[pred_hc_col].astype(int)
                            TOIL_inference['Required FTE'] = TOIL_inference['Required FTE'].astype(int)
                            TOIL_inference['Excess FTE'] = TOIL_inference['Excess FTE'].astype(int)
                            TOIL_inference['is_holiday'] = TOIL_inference['is_holiday'].astype(int)
                        
                            TOIL_inference[actual_TOIL_col] =  TOIL_inference[pred_TOIL_col]
                            
                            error_score = test_BaselineRegressor(TOIL_test, actual_TOIL_col, pred_TOIL_col)
                            st.warning(f"Root Mean Square Error of {TOIL_forecasting} Forecast: ±{error_score:.2f} Leaves", icon="⚠️")
                                
                        st.caption("Current Data")
                        st.dataframe(TOIL_inference[['Shift Date',
                                                    pred_hc_col,
                                                    'Required FTE',
                                                    'Excess FTE',
                                                    'is_holiday']].rename(columns={actual_TOIL_col : 'TOI Leave Count'}).set_index('Shift Date').T,)
    
                        st.caption("✏️ Editable Data")
                        TOIL_inference_edited = st.data_editor(TOIL_inference[['Shift Date',
                                                                                actual_TOIL_col]].rename(columns={actual_TOIL_col : 'TOI Leave Count'}).set_index('Shift Date').T,
                                                    key = f"TOIL_edited")
                        
                        old_leave_values = list(forecasting_df[forecasting_df.data_source != 'inference'][actual_TOIL_col])
                        forecasting_df['TOI Leave Count'] = old_leave_values + list(TOIL_inference_edited.T['TOI Leave Count'])
                        
                        submit_TOI_forecasting = st.checkbox("Proceed to Next Step",
                                                            help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.",
                                                            key = 'TOI')
                                
            if submit_hc_forecasting:
                if submit_AL_forecasting:
                    if submit_TOI_forecasting:
                        with st.expander("Billable Hours Forecasting Configuration"):
                            
                            BP_forecasting = st.radio(
                                "Billable Hours Forecasting Method",
                                ["Regression", "ARIMA", "3-Month Moving Average", "Persistence Model", "Expanding Mean Model"],
                                captions = [
                                            "Use Gradient Boosting Regression",
                                            "Use ARIMA techniques",
                                            "Use Moving Average for Leaves",
                                            "Retain Previous Month Leaves",
                                            "Get the average of ALL Historical Leaves"],
                                key = f"BP Leaves Method",
                                horizontal= True)        
                            
                            forecasting_df['LeaveHours'] = (forecasting_df['Annual Leave Count'] + 
                                                            forecasting_df['TOI Leave Count']) * (settings['CityWorkHours'][city]['internal'])
                    
                            forecasting_df['TotalLogHours'] = forecasting_df.apply(lambda x: sum([x[col] 
                                                                    for col in (
                                                                        settings['activityClass']['billable'] + 
                                                                        settings['activityClass']['nonbillable'] + 
                                                                        ['LeaveHours'])]) ,
                                                        axis = 1)
                            
                            forecasting_df['TotalLogHours'] = forecasting_df.apply(lambda x : x['TotalLogHours'] if x['TotalLogHours'] == x['TotalLogHours'] else x[pred_hc_col] * (settings['CityWorkHours'][city]['internal'] * 5),
                                                                                axis = 1)
                            forecasting_df['LeavePercentage'] = forecasting_df['LeaveHours']/forecasting_df['TotalLogHours']
                            forecasting_df['LeavePercentage'] = forecasting_df['LeavePercentage'].astype(float)
                            forecasting_df['BillableHours'] = forecasting_df.apply(lambda x:sum([x[col] for col in settings['activityClass']['billable']]),axis = 1)
                            forecasting_df['BillablePercentage'] = forecasting_df['BillableHours']/forecasting_df['TotalLogHours']
                            forecasting_df['BillablePercentage'] = forecasting_df['BillablePercentage'].astype(float)
                            for lag in range(1,3):
                                forecasting_df[f'BillablePercentage Lag {lag}'] = forecasting_df['BillablePercentage'].shift(lag).fillna(0)
                                forecasting_df[f'BillablePercentage Lag {lag}'] = forecasting_df[f'BillablePercentage Lag {lag}'].astype(float)
                            
                            if BP_forecasting == "Regression":

                                BillablePercentage_Model = train_XGBRegressor(forecasting_df[forecasting_df.data_source == 'training'], 
                                                                              settings, 
                                                                              forecasting_on="Billable",
                                                                              forecasting_what="NonARIMA",
                                                                              **settings['modelSettings']["Billable"]["NonARIMA"]['Parameters'])
                                BillablePercentage_test_forecasts, BillablePercentage_testing_accuracy = test_ModelRegressor(BillablePercentage_Model, settings, forecasting_df[forecasting_df.data_source == 'testing'], forecasting_on="Billable",forecasting_what="NonARIMA", metric = "rmse")
                                st.warning(f"Root Mean Square Error of Regression Model on Test Set: ±{100*BillablePercentage_testing_accuracy:.2f}%", icon="⚠️")
                                
                                lag_leave1 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-1]['BillablePercentage']
                                lag_leave2 = forecasting_df[forecasting_df.data_source == 'testing'].iloc[-2]['BillablePercentage']
                                BP_inference = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop = True)
                                for ind,row in BP_inference.iterrows():
                                    row['BillablePercentage Lag 1'] = lag_leave1
                                    row['BillablePercentage Lag 2'] = lag_leave2

                                    row_df = pd.DataFrame(row).T
                                    row_df['ActiveCount'] = row_df['ActiveCount'].astype(float)
                                    row_df['Required FTE'] = row_df['Required FTE'].astype(float)
                                    row_df["Excess FTE"] = row_df['Excess FTE'].astype(float)
                                    row_df['is_holiday'] = row_df['is_holiday'].astype(float)
                                    row_df['LeavePercentage'] = row_df['LeavePercentage'].astype(float)
                                    row_df['BillablePercentage'] = row_df['BillablePercentage'].astype(float)
                                    
                                    row_df['BillablePercentage Lag 1'] = row_df['BillablePercentage Lag 1'].astype(float)
                                    row_df['BillablePercentage Lag 2'] = row_df['BillablePercentage Lag 2'].astype(float)
                                    BP_forecast = infer_ModelRegressor(BillablePercentage_Model, 
                                                                        settings, 
                                                                        row_df, 
                                                                        forecasting_on="Billable", 
                                                                        forecasting_what="NonARIMA", 
                                                                        metric = "rmse")
                                    BP_inference.loc[ind,'BillablePercentage'] = passthrough(BP_forecast[0])
                                    BP_inference.loc[ind,'BillablePercentage Lag 1'] = lag_leave1
                                    BP_inference.loc[ind,'BillablePercentage Lag 2'] = lag_leave2
                                    
                                    lag_leave2 = lag_leave1
                                    lag_leave1 = passthrough(BP_forecast[0])
                                
                                actual_BP_col = 'BillablePercentage'
                                pred_BP_col = 'BillablePercentage'
                                BP_training = forecasting_df[forecasting_df.data_source == 'training']
                                BP_test = forecasting_df[forecasting_df.data_source == 'testing']
                                
                            elif BP_forecasting == "ARIMA":
                                BillablePercentage_ARIMAModel = train_ARIMA(forecasting_df[forecasting_df.data_source == 'training'],
                                    settings,
                                    'Billable',
                                    'ARIMA',
                                    **settings['modelSettings']["Billable"]["ARIMA"]['Parameters'])
                                BillablePercentage_test_forecasts, BillablePercentage_testing_accuracy = test_ARIMARegressor(BillablePercentage_ARIMAModel, settings, forecasting_df[forecasting_df.data_source == 'testing'], forecasting_on="Billable",forecasting_what="ARIMA", metric = "rmse")
                                BP_test = forecasting_df[forecasting_df.data_source == 'testing'][['Shift Date', 'LeavePercentage','BillablePercentage']].copy()
                                
                                st.warning(f"Root Mean Square Error of ARIMA on Test Set: ±{100*BillablePercentage_testing_accuracy:.2f}%", icon="⚠️")
                                BillablePercentage_ARIMAModel = train_ARIMA(forecasting_df[forecasting_df.data_source != 'inference'],
                                    settings,
                                    'Billable',
                                    'ARIMA',
                                    **settings['modelSettings']["Billable"]["ARIMA"]['Parameters'])
                                BP_inference_forecasts = infer_ARIMARegressor(BillablePercentage_ARIMAModel, settings, forecasting_df[forecasting_df.data_source == 'inference'], forecasting_on="Billable",forecasting_what="ARIMA", metric = "rmse")
                                BP_inference = forecasting_df[forecasting_df.data_source == 'inference'][['Shift Date','LeavePercentage']].copy()
                                print(BP_inference_forecasts)
                                BP_inference['BillablePercentage'] = list(BP_inference_forecasts)
                                
                                actual_BP_col = 'BillablePercentage'
                                pred_BP_col = 'BillablePercentage'
                            
                            elif BP_forecasting == "3-Month Moving Average":
                                BP_training, BP_test, BP_inference = moving_average_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                'BillablePercentage',
                                                                                12,
                                                                                forecast_size = lookforward_months,
                                                                                rounding_rule = passthrough)
                                actual_BP_col = 'BillablePercentage'
                                pred_BP_col = 'BillablePercentage_MA12_rounded'
                            elif BP_forecasting == "Persistence Method":
                                BP_training, BP_test, BP_inference = persistence_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                    forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                    'BillablePercentage',
                                                                                    forecast_size=lookforward_months)
                                actual_BP_col = 'BillablePercentage'
                                pred_BP_col = 'Persisted BillablePercentage'
                            else:
                                BP_training, BP_test, BP_inference = expanding_window_model(forecasting_df[forecasting_df.data_source == 'training'],
                                                                                            forecasting_df[forecasting_df.data_source == 'testing'],
                                                                                            'BillablePercentage',
                                                                                            rounding_rule=passthrough,
                                                                                            forecast_size=lookforward_months)
                                actual_BP_col = 'BillablePercentage'
                                pred_BP_col = 'Expanded mean_BillablePercentage'
                                
                            
                            BP_inference = pd.concat([forecasting_df[forecasting_df.data_source == 'inference'][[
                                                                    'Shift Date',
                                                                    pred_hc_col,
                                                                    'Required FTE',
                                                                    'Excess FTE',
                                                                    'is_holiday',
                                                                    'Annual Leave Count',
                                                                    'TOI Leave Count',
                                                                    'LeavePercentage']].reset_index(drop=True),
                                                        BP_inference[[pred_BP_col]].reset_index(drop=True)], 
                                                axis = 'columns')
                                                    
                                                    
                            BP_inference[actual_BP_col] =  BP_inference[pred_BP_col]
                                
                            if BP_forecasting not in ['Regression','ARIMA']:    
                                error_score = test_BaselineRegressor(BP_test, actual_BP_col, pred_BP_col)
                                st.warning(f"Root Mean Square Error of {BP_forecasting} Forecast: ±{100*error_score:.2f}%", icon="⚠️")
                                
                            st.caption("Editable Headcount Data (Resulting Annual Leave Forecasts)")
                            BP_inference['NonBillablePercentage'] = 1 - (BP_inference['BillablePercentage'])

                            st.caption("Current Data")
                            st.dataframe(BP_inference[['Shift Date',
                                                        pred_hc_col,
                                                        'Required FTE',
                                                        'Excess FTE',
                                                        'is_holiday',
                                                        'Annual Leave Count',
                                                        'TOI Leave Count',
                                                        'NonBillablePercentage']].rename(columns={
                                                                                pred_hc_col : 'Active Unique Agents per Week',
                                                                                'is_holiday' : '# of Holidays',
                                                                                'LeavePercentage' : 'Leave Allocation Proportion (vs Total Hours)',
                                                                                'NonBillablePercentage' : "Forecasted OOO Shrinkage"}).set_index('Shift Date').T,)
                            
                            st.caption("✏️ Editable Data")
                            BP_inference_edited = st.data_editor(BP_inference[['Shift Date',
                                                                                actual_BP_col]].rename(columns={'BillablePercentage' : 'Billable Activity Allocation Proportion (vs Total Hours)'}).set_index('Shift Date').T,
                                        key = f"BP_edited")
                                    
                                    
                            log_data = BP_inference[['Shift Date',
                                                    pred_hc_col,
                                                    'Required FTE',
                                                    'Excess FTE',
                                                    'is_holiday',
                                                    'Annual Leave Count',
                                                    'TOI Leave Count',
                                                    'LeavePercentage']].rename(columns={
                                                                            pred_hc_col : 'Active Unique Agents per Week',
                                                                            'is_holiday' : '# of Holidays',
                                                                            'LeavePercentage' : 'Leave Allocation Proportion (vs Total Hours)'}).set_index('Shift Date')
                                                    
                            BP_inference_edited = BP_inference_edited.T.rename(columns = {'Billable Activity Allocation Proportion (vs Total Hours)' : 'BillablePercentage'})
                            log_data['BillablePercentage'] = list(BP_inference_edited.BillablePercentage)
                            old_bp_inference_values = list(forecasting_df[forecasting_df.data_source != 'inference'].BillablePercentage)
                            forecasting_df['BillablePercentage'] = old_bp_inference_values + list(BP_inference_edited.BillablePercentage)
                            forecasting_df['NonBillablePercentage'] = 1 - (forecasting_df['BillablePercentage'] + forecasting_df['LeavePercentage'])
                            submit_BP_forecasting = st.checkbox("Proceed to Next Step",
                                                            help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.",
                                                            key = 'BP')
            
            if submit_hc_forecasting:
                if submit_AL_forecasting:
                    if submit_TOI_forecasting:
                        if submit_BP_forecasting:
                            with st.expander("NonBillable Activity Allocation Configuration"):
                                                        #TOIL Leave Forecasting
                                Distrib_Method= st.radio(
                                            "Activity Time Allocation Method",
                                            ["Target Configuration","Per Agent"],
                                            captions = ["Use a Weight System",
                                                        "Allocate Exact Daily Activity Hours per Agent"],
                                                key = f"Allocation Method",
                                                horizontal= True)
                                if Distrib_Method == "Target Configuration":
                                    inference_df = forecasting_df[forecasting_df.data_source == 'inference']
                                    DISTRIBUTION_SETTINGS = settings['activityClass']['Targets'][city]
                                    
                                    st.caption("Current NonBillable Activity Allocation")
                                    nbdf = pd.DataFrame(columns = ["Headcount","Total Hours", "Total NonBillable Hours", 'LeaveCount','Leave Hours', 'Break Hours', 'Meal Hours', 'Non-Meta Training Hours', 'Absenteeism Count'])
                                    for i,r in inference_df.iterrows():
                                        HC = r[pred_hc_col]
                                        TotHours = HC * settings['CityWorkHours'][city]['internal'] * 5
                                        TotBillable = TotHours * r['BillablePercentage']
                                        TotNonBillable = TotHours * (1 - r['BillablePercentage'])
                                        TotLeaveH = r['LeaveHours']
                                        TotLeaveC = r['Annual Leave Count'] + r['TOI Leave Count']
                                        row_values = pd.DataFrame([HC,TotHours, TotNonBillable, TotLeaveC, TotLeaveH, 2.5 * HC, 0, 0.5 * HC], # (TotHours - TotBillable - TotLeaveH - (2.5*HC))
                                                                index = ["Headcount","Total Hours", "Total NonBillable Hours" ,'LeaveCount','Leave Hours', 'Break Hours', 'Meal Hours', 'Non-Meta Training Hours'])
                                        nbdf = pd.concat([nbdf, row_values.T], axis = 0, ignore_index = True)
                                    nbdf.index = BP_inference_edited.index.tolist()  
                                    nbdf['Planned Absenteeism Hours'] =  nbdf['Total NonBillable Hours'] - (nbdf['Leave Hours'] + nbdf['Break Hours'] + nbdf['Meal Hours'] + nbdf['Non-Meta Training Hours'])
                                    nbdf['Absenteeism Count'] = np.round(nbdf['Planned Absenteeism Hours']/settings['CityWorkHours'][city]['internal'])
                                    st.dataframe(nbdf[['Total Hours','Total NonBillable Hours',
                                                      "LeaveCount", "Leave Hours","Absenteeism Count","Planned Absenteeism Hours",
                                                      'Break Hours', 'Meal Hours', 'Non-Meta Training Hours']])
                                    
                                    st.caption("✏️ Additional Leaves / Hours", help = "Input Additional or Reduction Values.")
                                    addtl_nbdf = pd.DataFrame(columns = ['LeaveCount', 'Absenteeism Count','Non-Meta Training Hours'],
                                                              index = nbdf.index).fillna(0.0)
                                    addtl_nbdf_edited = st.data_editor(addtl_nbdf)
                                    
                                    nbdf_edited = nbdf.add(addtl_nbdf_edited, fill_value = 0)
                                    nbdf_edited['Leave Hours'] = nbdf_edited['LeaveCount'] * settings['CityWorkHours'][city]['internal']
                                    nbdf_edited['Planned Absenteeism Hours'] = nbdf_edited['Absenteeism Count'] * settings['CityWorkHours'][city]['internal']
                                    nbdf_edited['Total NonBillable Hours'] = nbdf_edited[['Leave Hours','Planned Absenteeism Hours',
                                                                                          'Break Hours','Meal Hours',
                                                                                          'Non-Meta Training Hours']].sum(axis = 1)
                                    nbdf_edited['NonBillable Percentage'] = nbdf_edited['Total NonBillable Hours']/nbdf_edited['Total Hours']
                                    nbdf_edited['Billable Percentage'] = 1 - nbdf_edited['NonBillable Percentage']
                                    nbdf_edited['Total Billable Hours'] = nbdf_edited['Total Hours'] * nbdf_edited['Billable Percentage']
                                    st.caption("Resulting NonBillable Activities Allocation")
                                    st.dataframe(nbdf_edited[['Total Hours','Total NonBillable Hours',
                                                            "LeaveCount", "Leave Hours","Absenteeism Count","Planned Absenteeism Hours",
                                                            'Break Hours', 'Meal Hours', 'Non-Meta Training Hours']])
                                    
                                    st.caption("Resulting Billable vs Nonbillable Activity Allocation")
                                    st.dataframe(nbdf_edited[['Total Hours',
                                                              'Billable Percentage', 'Total Billable Hours',
                                                              'NonBillable Percentage', 'Total NonBillable Hours']])
                                    
                                elif Distrib_Method == "Per Agent":
                                    window_size = 12
                                    testing_df = forecasting_df[forecasting_df.data_source != 'inference']
                                    inference_df = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop=True)
                                    DISTRIBUTION_SETTINGS = settings['activityClass']['Targets'][city]
                                    
                                    st.caption("Current NonBillable Activity Allocation")
                                    nbdf = pd.DataFrame(columns = ["Headcount","Total Hours", "Total NonBillable Hours", 'LeaveCount','Leave Hours', 'Break Hours', 'Meal Hours', 'Non-Meta Training Hours', 'Absenteeism Count'])
                                    for i,r in inference_df.iterrows():
                                        HC = r[pred_hc_col]
                                        TotHours = HC * settings['CityWorkHours'][city]['internal'] * 5
                                        TotBillable = TotHours * r['BillablePercentage']
                                        TotNonBillable = TotHours * (1 - r['BillablePercentage'])
                                        TotLeaveH = r['LeaveHours']
                                        TotLeaveC = r['Annual Leave Count'] + r['TOI Leave Count']
                                        row_values = pd.DataFrame([HC,TotHours, TotNonBillable, TotLeaveC, TotLeaveH, 2.5 * HC, 0, 0.5 * HC], # (TotHours - TotBillable - TotLeaveH - (2.5*HC))
                                                                index = ["Headcount","Total Hours", "Total NonBillable Hours" ,'LeaveCount','Leave Hours', 'Break Hours', 'Meal Hours', 'Non-Meta Training Hours'])
                                        nbdf = pd.concat([nbdf, row_values.T], axis = 0, ignore_index = True)
                                    nbdf.index = BP_inference_edited.index.tolist()  
                                    nbdf['Planned Absenteeism Hours'] =  nbdf['Total NonBillable Hours'] - (nbdf['Leave Hours'] + nbdf['Break Hours'] + nbdf['Meal Hours'] + nbdf['Non-Meta Training Hours'])
                                    nbdf['Absenteeism Count'] = np.round(nbdf['Planned Absenteeism Hours']/settings['CityWorkHours'][city]['internal'])
                                    st.dataframe(nbdf[['Headcount','Total Hours','Total NonBillable Hours',
                                                      "LeaveCount", "Leave Hours","Absenteeism Count","Planned Absenteeism Hours",
                                                      'Break Hours', 'Meal Hours', 'Non-Meta Training Hours']])   
                                    
                                    st.caption("✏️ Non-Billable Activity Hours or Days per Agent per Day")
                                    nbdf_per_agent = nbdf.div(nbdf['Headcount']*5, axis = 0)
                                    nbdf_per_agent = st.data_editor(nbdf_per_agent[['LeaveCount',
                                                                                    'Absenteeism Count',
                                                                                    'Break Hours',
                                                                                    'Meal Hours',
                                                                                    'Non-Meta Training Hours']]) 
                                    
                                    nbdf_per_agent['Headcount'] =  nbdf['Headcount'] 
                                    
                                    
                                    nbdf_edited = nbdf_per_agent[nbdf_per_agent.columns.difference(['Headcount'])].multiply(nbdf_per_agent['Headcount']*5, axis = 0)
                                    nbdf_edited['Headcount'] = nbdf['Headcount']
                                    nbdf_edited['Total Hours'] = nbdf['Total Hours']
                                    nbdf_edited['Leave Hours'] = nbdf_edited['LeaveCount'] * settings['CityWorkHours'][city]['internal']
                                    nbdf_edited['Planned Absenteeism Hours'] = nbdf_edited['Absenteeism Count'] * settings['CityWorkHours'][city]['internal']
                                    nbdf_edited['Total NonBillable Hours'] = nbdf_edited[['Leave Hours','Planned Absenteeism Hours',
                                                                            'Break Hours','Meal Hours',
                                                                            'Non-Meta Training Hours']].sum(axis = 1)
                                    nbdf_edited['NonBillable Percentage'] = nbdf_edited['Total NonBillable Hours']/nbdf_edited['Total Hours']
                                    nbdf_edited['Billable Percentage'] = 1 - nbdf_edited['NonBillable Percentage']
                                    nbdf_edited['Total Billable Hours'] = nbdf_edited['Total Hours'] * nbdf_edited['Billable Percentage']
                                    st.caption("Resulting NonBillable Activities Allocation")
                                    st.dataframe(nbdf_edited[['Total Hours','Total NonBillable Hours',
                                                        "LeaveCount", "Leave Hours","Absenteeism Count","Planned Absenteeism Hours",
                                                        'Break Hours', 'Meal Hours', 'Non-Meta Training Hours']])
                                    
                                    st.caption("Resulting Billable vs Nonbillable Activity Allocation")
                                    st.dataframe(nbdf_edited[['Total Hours',
                                                              'Billable Percentage', 'Total Billable Hours',
                                                              'NonBillable Percentage', 'Total NonBillable Hours']])
                                    
                                submit_nonbillable_hours = st.checkbox("Proceed Next Step",
                                                            help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.",
                                                            key = 'DV')
            
                            
            if submit_hc_forecasting:
                if submit_AL_forecasting:
                    if submit_TOI_forecasting:
                        if submit_BP_forecasting:
                            if submit_nonbillable_hours:

                                with st.expander("Billable Activity Allocation Configuration"):
                                                            #TOIL Leave Forecasting
                                    Distrib_Method= st.radio(
                                                "Activity Time Allocation Method",
                                                ["Target Configuration", "Moving Average","Per Agent"],
                                                captions = ["Use a Weight System",
                                                            "Use 3-Months Moving Average",
                                                            "Allocate Exact Daily Activity Hours per Agent"],
                                                    key = f"Billable Allocation Method",
                                                    horizontal= True)
                                    if Distrib_Method == "Target Configuration":
                                        inference_df = forecasting_df[forecasting_df.data_source == 'inference']
                                        DISTRIBUTION_SETTINGS = settings['activityClass']['Targets'][city]
                                        
                                        st.caption("✏️ Billable Activity Weights")
                                        billable_alloc_df = pd.DataFrame(index = DISTRIBUTION_SETTINGS['billable_types'],
                                                                        data = DISTRIBUTION_SETTINGS['billable_values'])
                                        edited_billable_alloc = st.data_editor(billable_alloc_df.T).T
                                        
                                        bdf = pd.DataFrame(columns = DISTRIBUTION_SETTINGS['billable_types'])
                                        for i,r in nbdf_edited.iterrows():
                                            HC = r['Headcount']
                                            TotBillable = r['Total Billable Hours']
                                            activityHours_row = edited_billable_alloc.div(edited_billable_alloc.sum(axis=0), axis=1) * TotBillable
                                            bdf = pd.concat([bdf, activityHours_row.T], axis = 0)   
                                        
                                        bdf.index = nbdf_edited.index.tolist()      
                                        bdf['Total Billable Hours'] = nbdf_edited['Total Billable Hours']                      
                                        
                                        st.caption("Resulting Billable Activity Allocation")
                                        st.dataframe(bdf)

                                    elif Distrib_Method == "Moving Average":
                                        window_size = 12
                                        testing_df = forecasting_df[forecasting_df.data_source != 'inference']
                                        inference_df = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop=True)
                                        DISTRIBUTION_SETTINGS = settings['activityClass']['Targets'][city]
                                        
                                        testing_df['total_billable_hours'] = testing_df[DISTRIBUTION_SETTINGS['billable_types']].sum(axis=1)
                                        testing_df.sort_values('Shift Date',ascending = False, inplace = True)
                                        
                                        weight_df = pd.DataFrame(columns = DISTRIBUTION_SETTINGS['billable_types'])
                                        bdf = pd.DataFrame(columns = DISTRIBUTION_SETTINGS['billable_types'])
                                        
                                        for i,r in nbdf_edited.reset_index().iterrows():
                                            if i != 0:
                                                testing_df = pd.concat([pd.DataFrame(bdf.loc[i-1]).T, testing_df]).reset_index(drop=True)
                                                testing_df['total_billable_hours'] = testing_df[DISTRIBUTION_SETTINGS['billable_types']].sum(axis=1)
                                            for data_column in bdf.columns:
                                                data_window_col = testing_df.iloc[0 : window_size][data_column]
                                                data_window_totcol = testing_df['total_billable_hours'].iloc[0 :  window_size]
                                                col_weight = sum(data_window_col/data_window_totcol)/window_size
                                                weight_df.loc[i,data_column] = col_weight
                                                
                                                HC = r['Headcount']
                                                TotBillable = r['Total Billable Hours']
                                                bdf.loc[i,data_column] = TotBillable*col_weight
                                        bdf.index = nbdf_edited.index.tolist()  
                                        weight_df.index = nbdf_edited.index.tolist()  
                                        
                                        st.caption("Billable Activity Weights")
                                        st.dataframe(weight_df)        
                                        
                                        st.caption("Resulting Billable Allocation")
                                        bdf['Total Billable Hours'] = nbdf_edited['Total Billable Hours'] 
                                        st.dataframe(bdf)
                                        
                                    elif Distrib_Method == "Per Agent":
                                        window_size = 12
                                        testing_df = forecasting_df[forecasting_df.data_source != 'inference']
                                        inference_df = forecasting_df[forecasting_df.data_source == 'inference'].reset_index(drop=True)
                                        DISTRIBUTION_SETTINGS = settings['activityClass']['Targets'][city]
                                        
                                        testing_df['total_billable_hours'] = testing_df[DISTRIBUTION_SETTINGS['billable_types']].sum(axis=1)
                                        testing_df.sort_values('Shift Date',ascending = False, inplace = True)
                                        
                                        billable_alloc_df = pd.DataFrame(index = DISTRIBUTION_SETTINGS['billable_types'],
                                                                        data = DISTRIBUTION_SETTINGS['billable_values'])
                                        
                                        bdf = pd.DataFrame(columns = DISTRIBUTION_SETTINGS['billable_types'])
                                        for i,r in nbdf_edited.iterrows():
                                            HC = r['Headcount']
                                            TotBillable = r['Total Billable Hours']
                                            activityHours_row = billable_alloc_df.div(billable_alloc_df.sum(axis=0), axis=1) * TotBillable
                                            bdf = pd.concat([bdf, activityHours_row.T], axis = 0)   
                                        bdf.index = nbdf_edited.index.tolist()      
                                        bdf['Headcount'] = nbdf_edited['Headcount']
                                        bdf['Total Billable Hours'] = nbdf_edited['Total Billable Hours']  
                                        
                                        # Editable Table
                                        st.caption("✏️ Billable Activity Hours per Agent")
                                        bdf_per_agent = bdf[bdf.columns.difference(['Headcount','Total Billable Hours'])].div(bdf['Headcount']*5, axis = 0)
                                        bdf_per_agent = st.data_editor(bdf_per_agent) 
                                        
                                        bdf_per_agent['Headcount'] =  bdf['Headcount'] 
                                        
                                        bdf =  bdf_per_agent[bdf_per_agent.columns.difference(['Headcount'])].multiply(bdf_per_agent['Headcount']*5, axis = 0)
                                        
                                        st.caption("Resulting Billable Allocation")
                                        bdf['Total Billable Hours'] = nbdf_edited['Total Billable Hours'] 
                                        st.dataframe(bdf)
                                        
                                    submit_capacity_plan = st.checkbox("Proceed to Data Visualization",
                                                                help = "Forward Edited Table to Next Step. Any further edits after inital submission requires retoggling of checkbox.",
                                                                key = 'BDV')
                                
            if submit_hc_forecasting:
                if submit_AL_forecasting:
                    if submit_TOI_forecasting:
                        if submit_BP_forecasting:
                            if submit_nonbillable_hours:
                                if submit_capacity_plan:
                                    st.header(f"Capacity Planning")
                                    st.caption(f"Total Changed wrt last {lookback_months} weeks")
                                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                                    
                                    latest_data = forecasting_df[forecasting_df.data_source != 'inference'].sort_values('Shift Date', ascending = False)
                                    lookback_data = latest_data.iloc[:lookback_months]
                                    
                                    forecast_data = forecasting_df[forecasting_df.data_source == 'inference']
                                    
                                    metric_col1.metric(f"Avg. Unique Active Agents", 
                                                    f"{np.mean(forecast_data[pred_hc_col].astype(float)):.2f}", 
                                                    f"{np.mean(forecast_data[pred_hc_col].astype(float)) - np.mean(lookback_data['ActiveCount'].astype(float)):.2f}"
                                                    )
                                    
                                    forecasted_avg_shrinkage = 100*np.mean((forecast_data['Annual Leave Count'].astype(float) + forecast_data['TOI Leave Count'].astype(float))/(5*forecast_data[pred_hc_col]))  
                                    historical_avg_shrinkage = 100*np.mean((lookback_data['Annual Leave Count'].astype(float) + lookback_data['TOI Leave Count'].astype(float))/(5*lookback_data[pred_hc_col]))
                                    metric_col2.metric("Leaves Shrinkage %",
                                                    f"{forecasted_avg_shrinkage:.2f}%",
                                                    f"{forecasted_avg_shrinkage - historical_avg_shrinkage:.2f}%")
                                    
                                    forecasted_leave_coverage = 100*np.mean(np.mean(forecast_data['Annual Leave Count'].astype(float) + forecast_data['TOI Leave Count'].astype(float))/(forecast_data['Excess FTE'].astype(float) * 5))
                                    lookback_leave_coverage = 100*np.mean(np.mean(lookback_data['Annual Leave Count'].astype(float) + lookback_data['TOI Leave Count'].astype(float))/(lookback_data['Excess FTE'].astype(float) * 5))
                                    metric_col3.metric("Excess FTE Leave Coverage", 
                                                    f"{forecasted_leave_coverage:.2f}%",
                                                    f"{forecasted_leave_coverage - lookback_leave_coverage:.2f}%",
                                                    )
                                    with st.expander("Capacity Plan Data Summary", expanded=True):
                                        st.caption("Basic Log Information")
                                        st.dataframe(log_data)
                                        
                                        st.caption("Billable Activity Hours Allocation")
                                        st.dataframe(bdf)
                                        
                                        st.caption("NonBillable Activity Hours Allocation")
                                        st.dataframe(nbdf)
                                        
                                        finalize = st.checkbox('Finalize Data',
                                                            help = 'Submit Data for Visualization',
                                                            key = 'finalize')
                                    
                                    if finalize:
                                    
                                        waterfall_tab, forecasting_tab = st.tabs(['📊 Waterfall Chart', "📉 Forecasting Plots"])
                                        with waterfall_tab:
                                            date_to_review = st.selectbox("Select Planning Week to Review",
                                                                        options = log_data.index.tolist(),
                                                                        index=0)                              
                                            generate_waterfall = st.checkbox("Plot Waterfall",
                                                                            key="WaterFall")
                                            
                                            if generate_waterfall:
                                                fcp_simulator = ForecastCapPlan(city, date_to_review, log_data, bdf, nbdf, settings)
                                                
                                                URI_1 = fcp_simulator.UtilizationInternal 
                                                URC_1 = fcp_simulator.UtilizationClient
                                                WIO_1 = fcp_simulator.WIO
                                                OOO_1 = fcp_simulator.OOO

                                                st.caption("Metrics vs Targets")
                                                col1,col2,col3,col4 = st.columns(4)

                                                col1.metric("Utilization Rate (Internal)", f"{100*URI_1:.2f}%", 
                                                            help = "Projected Productive Hours / Projected Billable Hours",
                                                            delta = f"{100*(settings['simulator']['Targets']['Utilization'] - URI_1):.2f}%")
                                                col2.metric("Utilization Rate (Client)", f"{100*URC_1:.2f}%", 
                                                            help = f"Projected Productive Hours / Target Billable Hours @ ({settings['simulator']['Targets']['OOO']*100}% OOO Shrinkage & {settings['simulator']['Targets']['Utilization']*100}% Utilization)",
                                                            delta = f"{100*(settings['simulator']['Targets']['Utilization_Client'] - URC_1):.2f}%")
                                                col3.metric("WIO Shrinkage", f"{100*WIO_1:.2f}%", 
                                                            help = "(Projected Billable Hours - Projected Productive Hours Logged) / Projected Billable Hours",
                                                            delta = f"{100*(settings['simulator']['Targets']['WIO'] - WIO_1):.2f}%")
                                                col4.metric("OOO Shrinkage", f"{100*OOO_1:.2f}%", 
                                                            help = "Projected Nonbillable Hours / Projected Total Hours",
                                                            delta = f"{100*(settings['simulator']['Targets']['OOO'] - OOO_1):.2f}%")

                                                st.plotly_chart(fcp_simulator.generate_waterfall(), use_container_width = True)
                                                
                                                with st.expander("Recommendations", expanded = True):
                                                    # Initialize actions lists for below and above scenarios
                                                    actions_below = []
                                                    actions_above = []
                                                    
                                                    # Check scenarios and store actions accordingly
                                                    if URI_1 < settings['simulator']['Targets']['Utilization']:
                                                        actions_below.append("Internal Utilization Rate is below target. Adjust Activity Allocation to optimize resource utilization.")
                                                        actions_below.append("Internal Utilization Rate is below target. Allocating more hours towards other non-project activities may enhance employee satisfaction and performance.")
                                                    else:
                                                        actions_above.append("Internal Utilization Rate is either above or equal to target. Consider revisiting project allocations to maintain employee engagement while meeting targets.")

                                                    if URC_1 < settings['simulator']['Targets']['Utilization_Client']:
                                                        actions_below.extend([
                                                            "Client Utilization Rate is below target. Increase Hour Allocation Towards Productive Hours to meet client demands.",
                                                            "Client Utilization Rate is below target. Consider Increasing Headcount to manage workload effectively.",
                                                            "Client Utilization Rate is below target. Encourage or Enforce Reduction in Leave Count to ensure sufficient workforce."
                                                        ])
                                                    else:
                                                        actions_above.extend([
                                                            "Client Utilization Rate is either above or equal to target. Slightly adjust hour allocation towards non-client-related tasks to diversify employee experience.",
                                                            "Client Utilization Rate is either above or equal to target. Evaluate current workforce capacity for future scaling rather than immediate additions.",
                                                            "Client Utilization Rate is either above or equal to target. Encourage a balanced approach to leave counts (potential increase in leave count) for employee well-being."
                                                        ])
                                                        actions_above.append("Client Utilization Rate is either above or equal to target. Current client workload is manageable, consider diversifying tasks for skill enhancement.")

                                                    if WIO_1 > settings['simulator']['Targets']['WIO']:
                                                        actions_above.append("WIO Shrinkage is above target. Consider optimizing schedules to reduce within-office non-productive hours.")
                                                    else:
                                                        actions_below.append("WIO Shrinkage is either below or equal to target. Current allocation seems balanced.")
                                                        actions_below.append("WIO Shrinkage is either below or equal to target. Adjust Activity Allocation towards Non-Productive Billable Hours to maximize workforce allocation.")
                                                       
                                                        
                                                    if OOO_1 < settings['simulator']['Targets']['OOO']:
                                                        actions_above.extend([
                                                            "OOO Shrinkage is above target. Consider reviewing leave and training allocations for better balance.",
                                                        ])
                                                        
                                                    else:
                                                        actions_below.append("OOO Shrinkage is below target. Current allocations seem sufficient.")
                                                        actions_below.extend([
                                                            "OOO Shrinkage is either below equal to target. Increase Leave/Absenteeism Allocation to accommodate absences.",
                                                            "OOO Shrinkage is either below equal to target. Increase Allocation Towards Non-Meta Training for skill enhancement."
                                                        ])
                                                    
                                                    # Display actions for below and above scenarios
                                                    st.warning("⚠️ Recommended Actions for Below Target Scenarios:")
                                                    for action in actions_below:
                                                        st.write(f"- {action}")

                                                    st.info("✅ Recommended Adjustments for Above Target Scenarios:")
                                                    for action in actions_above:
                                                        st.write(f"- {action}")
                                                                                                                                                        
                                        with forecasting_tab:
                                            graph_space_col, graph_select_col = st.columns([4,2], gap = "medium")
                                            
                                            with graph_select_col:
                                                Plot_to_Viz = st.radio(
                                                            "Select Graph to Visualize",
                                                            ["FTE Allocation vs Internal Utilization",
                                                            "FTE Allocation vs Client Utilization",
                                                            "FTE Allocation vs OOO Shrinkage",
                                                            "FTE Allocation vs WIO Shrinkage",
                                                            "Billable Hours vs Target Billable Hours",
                                                            "Productive Hours vs Target Productive Hours",
                                                            "Leave Allocation vs Excess FTE Coverage"],
                                                            captions = ["",
                                                                        "",
                                                                        "",
                                                                        "",
                                                                        "",
                                                                        ""
                                                                        ],
                                                            key = f"P2V Method",
                                                            horizontal= False)
                                            
                                            with graph_space_col:
                                                noninference_df = forecasting_df[forecasting_df.data_source != 'inference']
                                                
                                                # Non-Inference
                                                bdf_cols = bdf[bdf.columns.difference(["Total Billable Hours"])].columns.tolist()
                                                util_noninference_df = noninference_df[["Shift Date",
                                                                                        "ActiveCount",
                                                                                        pred_hc_col,
                                                                                        "Required FTE",
                                                                                        "Excess FTE",
                                                                                        "data_source",
                                                                                        "Annual Leave Count",
                                                                                        "TOI Leave Count",
                                                                                        "LeaveHours",
                                                                                        "TotalLogHours"] + 
                                                                                        bdf_cols +
                                                                                        ['meal Hours', 'break Hours', 'non-fb-training Hours']
                                                ]
                                                                                        
                                                util_noninference_df['Productive Hours'] = util_noninference_df['available Hours']
                                                util_noninference_df['Billable Hours'] = util_noninference_df[bdf_cols].sum(axis = 1)
                                                util_noninference_df['NonBillable Hours'] = util_noninference_df[['LeaveHours'] +
                                                                                                                ['meal Hours', 
                                                                                                                'break Hours', 
                                                                                                                'non-fb-training Hours']].sum(axis = 1)
                                                util_noninference_df['Target Billable Hours'] = util_noninference_df['Required FTE'] * (settings['CityWorkHours'][city]['client'] * 5) * (1-settings['simulator']['Targets']['OOO']) * settings['simulator']['Targets']['Utilization']
                                                util_noninference_df['Target Productive Hours'] = util_noninference_df['Target Billable Hours'] * settings['simulator']['Targets']['Utilization']
                                                util_noninference_df['Internal Utilization %'] = 100 * util_noninference_df['Productive Hours']/util_noninference_df['Billable Hours']
                                                util_noninference_df['Client Utilization %'] = 100 * util_noninference_df['Productive Hours']/(util_noninference_df['Required FTE'] * settings['CityWorkHours'][city]['client'] * 5 * (1-settings['simulator']['Targets']['OOO']))
                                                util_noninference_df['WIO Shrinkage'] = 100 * ((util_noninference_df['Billable Hours'] - util_noninference_df['Productive Hours'])/(util_noninference_df['Billable Hours'])) 
                                                util_noninference_df['OOO Shrinkage'] = 100 * (util_noninference_df['NonBillable Hours']/util_noninference_df['TotalLogHours'])
                                                util_noninference_df['Excess FTE Coverage'] = 100 * ((util_noninference_df['Annual Leave Count'] + util_noninference_df['TOI Leave Count'])/(5 * (util_noninference_df['Excess FTE'])))
                                                
                                                util_noninference_df = util_noninference_df.iloc[-lookback_months:]
                                                
                                                
                                                # Inference Data
                                                util_inference_df = pd.concat([log_data, bdf, nbdf],axis = 1)
                                                
                                                util_inference_df['Productive Hours'] = util_inference_df['available Hours']
                                                util_inference_df['Billable Hours'] = util_inference_df[bdf_cols].sum(axis = 1)
                                                util_inference_df['NonBillable Hours'] = util_inference_df[['Leave Hours',
                                                                                                            'Planned Absenteeism Hours'] +
                                                                                                            ['Meal Hours', 
                                                                                                            'Break Hours', 
                                                                                                            'Non-Meta Training Hours']].sum(axis = 1)
                                                util_inference_df['Target Billable Hours'] = util_inference_df['Required FTE'] * (settings['CityWorkHours'][city]['client'] * 5) * (1-settings['simulator']['Targets']['OOO'])
                                                util_inference_df['Target Productive Hours'] = util_inference_df['Target Billable Hours'] * settings['simulator']['Targets']['Utilization']
                                                util_inference_df['Internal Utilization %'] = 100 * util_inference_df['Productive Hours']/util_inference_df['Billable Hours']
                                                util_inference_df['Client Utilization %'] = 100 * util_inference_df['Productive Hours']/(util_inference_df['Required FTE'] * settings['CityWorkHours'][city]['client'] * 5 * (1-settings['simulator']['Targets']['OOO']))
                                                util_inference_df['WIO Shrinkage'] = 100 * ((util_inference_df['Billable Hours'] - util_inference_df['Productive Hours'])/(util_inference_df['Billable Hours'])) 
                                                util_inference_df['OOO Shrinkage'] = 100 * (util_inference_df['NonBillable Hours']/(util_inference_df['Billable Hours'] + util_inference_df['NonBillable Hours']))
                                                util_inference_df['Excess FTE Coverage'] = 100 * ((util_inference_df['Annual Leave Count'] + util_inference_df['TOI Leave Count'] + util_inference_df['Absenteeism Count'])/(5 * (util_inference_df['Excess FTE'])))
                                                
                                                if Plot_to_Viz == "FTE Allocation vs Internal Utilization":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    
                                                    fig.add_trace(go.Bar(name = "Target Headcount",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Required FTE"]) +list(util_inference_df['Required FTE'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Unique Active Agents",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['ActiveCount']) + list(util_inference_df['Active Unique Agents per Week'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "Internal Utilization %",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["Internal Utilization %"]) + list(util_inference_df['Internal Utilization %'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.add_hline(name = "Internal Utilization Target",
                                                                  y=settings['simulator']['Targets']['Utilization'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)

                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.4,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[80, 120], secondary_y=True)
                                                    fig.update_layout(
                                                                    barmode='group',
                                                                    bargroupgap=0.0,
                                                                    bargap = 0.30,
                                                                    yaxis_title="Number of Agents",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="Internal Utilization %", 
                                                                    secondary_y=True)
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "FTE Allocation vs Client Utilization":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Target Headcount",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Required FTE"]) +list(util_inference_df['Required FTE'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Unique Active Agents",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['ActiveCount']) + list(util_inference_df['Active Unique Agents per Week'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "Client Utilization %",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["Client Utilization %"]) + list(util_inference_df['Client Utilization %'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.add_hline(name = "Client Utilization Target",
                                                                  y=settings['simulator']['Targets']['Utilization_Client'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[80,200], secondary_y=True)
                                                    fig.update_layout(
                                                                    yaxis_title="Number of Agents",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="Client Utilization %", 
                                                                    secondary_y=True)
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "Billable Hours vs Target Billable Hours":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Billable Hours",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Billable Hours"]) +list(util_inference_df['Billable Hours'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Target Billable Hours",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['Target Billable Hours']) + list(util_inference_df['Target Billable Hours'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "Client Utilization %",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["Client Utilization %"]) + list(util_inference_df['Client Utilization %'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.add_hline(name = "Client Utilization Target",
                                                                  y=settings['simulator']['Targets']['Utilization_Client'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[80,200], secondary_y=True)
                                                    fig.update_layout(
                                                                    yaxis_title="Billable Hours",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="Client Utilization %", 
                                                                    secondary_y=True)
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "Productive Hours vs Target Productive Hours":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Productive Hours",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Productive Hours"]) +list(util_inference_df['Productive Hours'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Target Productive Hours",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['Target Productive Hours']) + list(util_inference_df['Target Productive Hours'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "Client Utilization %",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["Client Utilization %"]) + list(util_inference_df['Client Utilization %'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.add_hline(name = "Client Utilization Target",
                                                                  y=settings['simulator']['Targets']['Utilization_Client'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[80,200], secondary_y=True)
                                                    fig.update_layout(
                                                                    yaxis_title="Productive Hours",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="Client Utilization %", 
                                                                    secondary_y=True)
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "FTE Allocation vs OOO Shrinkage":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Target Headcount",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Required FTE"]) +list(util_inference_df['Required FTE'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Unique Active Agents",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['ActiveCount']) + list(util_inference_df['Active Unique Agents per Week'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "OOO Shrinkage",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["OOO Shrinkage"]) + list(util_inference_df['OOO Shrinkage'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.add_hline(name = "OOO Shrinkage Target",
                                                                  y=settings['simulator']['Targets']['OOO'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)
                                                    
                                                    fig.update_yaxes(range=[0,50], secondary_y=True)
                                                    
                                                    fig.update_layout(
                                                                    yaxis_title="Number of Agents",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="% OOO Shrinkage", 
                                                                    secondary_y=True)
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "FTE Allocation vs WIO Shrinkage":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Target Headcount",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df["Required FTE"]) +list(util_inference_df['Required FTE'])))
                                                    
                                                    fig.add_trace(go.Bar(name = "Unique Active Agents",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df['ActiveCount']) + list(util_inference_df['Active Unique Agents per Week'])
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "WIO Shrinkage",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["WIO Shrinkage"]) + list(util_inference_df['WIO Shrinkage'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.add_hline(name = "WIO Shrinkage Target",
                                                                  y=settings['simulator']['Targets']['WIO'] * 100, 
                                                                  line_width=3, 
                                                                  line_dash="dash", 
                                                                  line_color="green",
                                                                  secondary_y = True)
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[0,50], secondary_y=True)
                                                    
                                                    fig.update_layout(
                                                                    yaxis_title="Number of Agents",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="% WIO Shrinkage", 
                                                                    secondary_y=True)
                                                    
                                                
                                                    st.plotly_chart(fig)
                                                    
                                                elif Plot_to_Viz == "Leave Allocation vs Excess FTE Coverage":
                                                    
                                                    from plotly.subplots import make_subplots

                                                    # Create figure with secondary y-axis
                                                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                    fig.add_trace(go.Bar(name = "Excess FTE vs Unique Agents (Week)",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = (list(5*util_noninference_df["Excess FTE"]) +list(5*util_inference_df['Excess FTE']))
                                                                        )
                                                                  )
                                                    
                                                    fig.add_trace(go.Bar(name = "Leaves Allocation",
                                                                        x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                        y = list(util_noninference_df[['Annual Leave Count','TOI Leave Count']].sum(axis = 1)) + list(util_inference_df[['Annual Leave Count','TOI Leave Count','Absenteeism Count']].sum(axis = 1))
                                                                        ))
                                                    
                                                    fig.data[0].marker.color = tuple(['#0081FB']*(len(util_noninference_df) + len(util_inference_df)))
                                                    fig.data[1].marker.color = tuple(['#FDB51B']*len(util_noninference_df) + ['#FFECC1']*len(util_inference_df))
                                                    
                                                    fig.add_trace(go.Scatter(name = "Excess FTE Coverage %",
                                                                            x = list(util_noninference_df['Shift Date']) + list(util_inference_df.index),
                                                                            y = list(util_noninference_df["Excess FTE Coverage"]) + list(util_inference_df['Excess FTE Coverage'])),
                                                                            secondary_y = True
                                                                )
                                                    
                                                    fig.update_layout(legend=dict(
                                                                        orientation="h",
                                                                        yanchor="bottom",
                                                                        y=-0.25,
                                                                        xanchor="right",
                                                                        x=1
                                                                    ))
                                                    
                                                    fig.update_yaxes(range=[0,
                                                                            max(util_inference_df['Excess FTE Coverage']) + 10],
                                                                    secondary_y=True)
                                                    
                                                    fig.update_layout(
                                                                    yaxis_title="Number of Agents",
                                                                    font=dict(
                                                                        size=10,
                                                                    )
                                                    )
                                                                    
                                                    fig.update_yaxes(
                                                                    title_text="% Excess FTE Accounted For by Leaves", 
                                                                    secondary_y=True)
                                                
                                                    st.plotly_chart(fig)
                                            
                                                
                                                
                
            
        with historical_data_tab:
            historical_date = st.selectbox("Select an Initial Historical Date to Load",
                                        list(filtered_data["Shift Date"].unique()),
                                        index = 12,
                                        help = "Select the starting date to load for planning or use as a reference for planning of future months",
                                        )
            
            # Data To Filter
            historical_data = filtered_data.copy()
            historical_data = historical_data.set_index("Shift Date")
            
            # Data To Review
            review_series = historical_data.loc[historical_date]
            print(review_series.index)
            
            capcol1, capcol2, capcol3 = st.columns(3)
            
            # Basic Information
            capcol1.caption("Basic Log Information")
            capcol1.dataframe(pd.DataFrame(review_series[['City','Market','Role',
                                                        'ActiveCount', 'NFires','NHires',
                                                        'Required FTE', 'Excess FTE']]).rename(index = {'ActiveCount' : 'Current Headcount',
                                                                                                            'NFires' : 'Attrition',
                                                                                                            'NHires' : 'New Hires'}),
                                                        use_container_width=False)
            
            # Holidays and Leaves
            capcol2.caption("Holidays and Leaves")
            capcol2.dataframe(pd.DataFrame(review_series[['is_holiday',
                                                        'annual_leave_value',
                                                        'toil_leave_value',
                                                        'Total Annual Leave Hours',
                                                        'Total TOI Leave Hours']]).rename(index = {'is_holiday' : 'Number of Holidays',
                                                                                                    'annual_leave_value' : '# of Annual Leaves',
                                                                                                    'toil_leave_value' : '# of TOI Leaves'}),
                                                        use_container_width=False)
            
            # Activity Type Allocation
            capcol3.caption("Total Hours Logged")
            capcol3.dataframe(pd.DataFrame(review_series[['Productive Hours',
                                                        'Billable Hours',
                                                        'Nonbillable Hours',
                                                        'Total Logged Hours'
                                                        ]]),
                            use_container_width=False)
            
            # Activity Breakdown
            activity_df = review_series[['available Hours',
                                        'fb_training Hours',
                                        'coaching Hours', 
                                        'team_meeting Hours',
                                        'onboarding Hours',
                                        'wellness_support Hours',
                                        'non-fb-training Hours', 
                                        'meal Hours',
                                        'break Hours',
                                        ]]
            
            st.caption("Activity Breakdown")
            st.dataframe(pd.DataFrame(activity_df).T,
                                            use_container_width=False,
                                            hide_index=True)
            
            st.caption("Average Agent Activity per Day",
                    help = "Week Activity Allocation / (Headcount x 5 days)")
            st.dataframe(pd.DataFrame(activity_df/(5*review_series['ActiveCount'])).T,
                                                        use_container_width=False,
                                                        hide_index=True)


            simulator = HistoricalCapPlan(review_series, settings)
            
            URI_1 = simulator.UtilizationInternal 
            URC_1 = simulator.UtilizationClient
            WIO_1 = simulator.WIO
            OOO_1 = simulator.OOO

            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Utilization Rate (Internal)", f"{100*URI_1:.2f}%", help = "Actual Productive Hours Logged / Actual Billable Hours Logged")
            col2.metric("Utilization Rate (Client)", f"{100*URC_1:.2f}%", help = f"Actual Productive Hours Logged / Target Billable Hours @ ({settings['simulator']['Targets']['OOO']*100}% OOO Shrinkage & {settings['simulator']['Targets']['Utilization']*100}% Utilization)")
            col3.metric("WIO Shrinkage", f"{100*WIO_1:.2f}%", help = "(Actual Billable Hours Logged - Actual Productive Hours Logged) / Actual Billable Hours Logged")
            col4.metric("OOO Shrinkage", f"{100*OOO_1:.2f}%", help = "Actual Nonbillable Hours Logged / Actual Total Hours Logged")

            st.plotly_chart(simulator.generate_waterfall(), use_container_width = True)
        
        

        
        
    
    