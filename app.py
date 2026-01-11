from fastapi import FastAPI
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
import math
from joblib import load
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from prophet import Prophet
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from scipy import stats

# Initialize the FastAPI app
app = FastAPI()

def inv_boxcox(y_transformed, lambda_value):
    if lambda_value == 0:
        return np.exp(y_transformed)
    else:
        return (y_transformed * lambda_value + 1) ** (1 / lambda_value)

def calculation_logic(input1: str, input2: int):
    # Example logic: modify as needed
    # output1 = input1 * 2  # Example: multiply input1 by 2
    # output2 = input2 + 5  # Example: add 5 to input2
    
    data_1=pd.read_csv('McKesson_Large_Demand_Forecasting_Dataset.csv')
    data_2=pd.read_csv('daily_dataset_2023.csv')
    data_1.drop(columns=['product_id','region'], inplace=True)
    data_1['month_day'] = data_1['date'].str[5:10]
    data_2['month_day'] = data_2['date'].str[5:10]
    merged_df = pd.merge(data_1, data_2[['month_day', 'season']], 
                        on='month_day', 
                        how='left')
    merged_df['month_And_day_of_month'] = merged_df['month_day']
    merged_df['month_And_day'] = merged_df['date'].str[0:7]

    merged_df['date']=pd.to_datetime(merged_df['date'])
    print(merged_df.columns)
    merged_df['month_day']=merged_df['month_day'].str[3:5].astype(int)
    merged_df['day_of_week'] = merged_df['date'].dt.weekday        # 0=Monday, 6=Sunday
    merged_df['week_of_month'] = merged_df['date'].apply(lambda d: (d.day-1)//7 + 1)
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['lag_1'] = merged_df['units_sold'].shift(1)   
    merged_df['lag_7'] = merged_df['units_sold'].shift(7)
    merged_df['roll_mean_7'] = merged_df['units_sold'].shift(1).rolling(7).mean()
    merged_df['roll_std_7']  = merged_df['units_sold'].shift(1).rolling(7).std()
    merged_df['roll_sum_30'] = merged_df['units_sold'].shift(1).rolling(30).sum()
    # print(merged_df)
    merged_df.dropna(inplace=True)
    # print(merged_df['roll_sum_30'].head(100))
    merged_df_label_encoded=merged_df.copy()
    merged_df_label_encoded['medicine_season'] = merged_df_label_encoded['product_name'].astype(str) + "_" + merged_df_label_encoded['season'].astype(str)
    merged_df_label_encoded['medicine_flu'] = merged_df_label_encoded['product_name'].astype(str) + "_" + merged_df_label_encoded['flu_alert_level'].astype(str)
    for col in ['product_name','category','flu_alert_level','season','medicine_season','medicine_flu']:
        if col in merged_df_label_encoded.columns:
            le = LabelEncoder()
            merged_df_label_encoded[col] = le.fit_transform(merged_df_label_encoded[col].astype(str))
            print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # print(merged_df_label_encoded['month'].tail(100))
    # print(merged_df_label_encoded['roll_sum_30'].head(30))
    # print(merged_df_label_encoded['week_of_month'].tail(100))
    selected_columns=['date', 'product_name', 'category', 'units_sold', 'unit_price',
       'market_trend_index', 'is_holiday', 'flu_alert_level', 'economic_index',
       'month_day', 'season', 'day_of_week', 'week_of_month', 'month',
       'is_weekend', 'lag_1', 'lag_7', 'roll_mean_7', 'roll_std_7',
       'roll_sum_30','medicine_season','medicine_flu','month_And_day'];
    medicine_averagedf = None
    print("calculating medicine averages for :", input1)
    new_dataset = merged_df_label_encoded.loc[merged_df_label_encoded['product_name'] == 0, selected_columns]
    
    average_df = new_dataset.groupby(['month_And_day', 'week_of_month']).agg(
    {  'date': 'first', 'product_name': 'mean', 'category': 'mean', 'units_sold': 'mean', 'unit_price': 'mean',
       'market_trend_index': 'mean', 'is_holiday': 'mean', 'flu_alert_level': 'mean', 'economic_index': 'mean',
       'month_day': 'mean', 'season': 'mean', 'day_of_week': 'mean',  'month': 'mean',
       'is_weekend': 'mean', 'lag_1': 'mean', 'lag_7': 'mean', 'roll_mean_7': 'mean', 'roll_std_7': 'mean',
       'roll_sum_30': 'mean','medicine_season': 'mean','medicine_flu': 'mean'})
    # print(average_df.head(20))
    # print(average_df.dtypes)
   
    result_df = average_df.reset_index()
    medicine_averagedf = result_df.copy()   
    today = date.today()
    print(f"Today's date: {today}")

    # Calculate the date after 6 months
    # relativedelta automatically handles edge cases (e.g., adding months to Jan 31)
    future_date = today + relativedelta(months=+input2)
    difference = future_date - today
    no_of_weeks=difference.days // 7
    print(f"Date after 6 months: {future_date}")
    print(f"Difference in days: {difference.days} days")
    print(f"Difference in weeks: {no_of_weeks} weeks")
    date_0=today
    season_values = [None] * 7
    flu_index_values = [None] * 7
    week_of_the_month = [None] * 7
    month_of_the_year = [None] * 7
    week_of_the_year = [None] * 7 
    lag_1_values = [None] * 7
    lag_7_values = [None] * 7  
    roll_mean_7_values = [None] * 7
    roll_std_7_values = [None] * 7
    formatted_datetime_str = today.strftime("%Y-%m-%d")
    lag_1=None
    lag_7=None
    roll_mean_7=None
    roll_std_7=None
    print(medicine_averagedf.columns)
    print(medicine_averagedf['week_of_month'])
    total_count=0
    lambda_value=0.30454259734348804
    price=0
    # print(merged_df_label_encoded.columns)
    # print(merged_df_label_encoded['month_And_day_of_month'])
    
    if input1=="Amoxicillin_500mg":
        price=1.25
    elif input1=="Atorvastatin_20mg":
        price=0.85
    elif input1=="Insulin_Glargine":
        price=2.50
    elif input1=="Surgical_Gloves_Box":
        price=1.75
    elif input1=="Surgical_Masks_Box":
        price=1.10


    for i in range(no_of_weeks):
        for j in range(7):
            date_0 += relativedelta(days=+1)
            print(date_0)
            formatted_datetime_str = date_0.strftime("%Y-%m-%d")
            condition = merged_df_label_encoded['month_And_day_of_month'] == formatted_datetime_str[5:10]
            first_value = merged_df_label_encoded[condition]['season'].head(1).iloc[0]
            # month = merged_df_label_encoded[condition]['month'].iloc[0]
            flu_index_value= merged_df_label_encoded[condition]['flu_alert_level'].head(1).iloc[0]
            current_week_num = date_0.isocalendar()[1]
            # first_day_of_month = date_0.replace(day=1)
            # first_day_weekday = first_day_of_month.weekday()
            # day_of_month = date_0.day
            week_of_month= (int(formatted_datetime_str[8:10])-1)//7 + 1
            print(f"Week of the month: {week_of_month}")
            print(f"Month: {formatted_datetime_str[5:7]}")
            condition_lag = ((medicine_averagedf['week_of_month'] == float(week_of_month)) & (medicine_averagedf['month'] == float(formatted_datetime_str[5:7])))
            print(medicine_averagedf)
            lag_1=float(medicine_averagedf[condition_lag]['lag_1'].head(1))
            lag_7=float(medicine_averagedf[condition_lag]['lag_7'].head(1))
            roll_mean_7=float(medicine_averagedf[condition_lag]['roll_mean_7'].head(1))
            roll_std_7=float(medicine_averagedf[condition_lag]['roll_std_7'].head(1))
            flu_index_values[j]=flu_index_value
            season_values[j]=first_value
            week_of_the_month[j]=week_of_month
            week_of_the_year[j]=current_week_num
            month_of_the_year[j]=int(formatted_datetime_str[5:7])
            lag_1_values[j]=lag_1
            lag_7_values[j]=lag_7
            roll_mean_7_values[j]=roll_mean_7
            roll_std_7_values[j]=roll_std_7
            # month_of_the_year[j]=month
        print(season_values)
        print(flu_index_values)
        print(week_of_the_month)
        print(month_of_the_year)
        print(week_of_the_year)
        flu_index_average = float(sum(flu_index_values) / len(flu_index_values))
        season_average = float(sum(season_values) / len(season_values))
        week_of_the_year_max = float(max(set(week_of_the_year), key=week_of_the_year.count))
        month_of_the_year_max= float(max(set(month_of_the_year), key=month_of_the_year.count))
        week_of_the_month_max = float(max(set(week_of_the_month), key=week_of_the_month.count))
        lag_1_average = float(sum(lag_1_values) / len(lag_1_values))
        lag_7_average = float(sum(lag_7_values) / len(lag_7_values))
        roll_mean_7_average = float(sum(roll_mean_7_values) / len(roll_mean_7_values))
        roll_std_7_average = float(sum(roll_std_7_values) / len(roll_std_7_values))
        print("Averages:")
        print(f"Season Average: {season_average}")
        print(f"Flu Index Average: {flu_index_average}") 
        print(f"Week of the Year max: {week_of_the_year_max}")  
        print(f"Month of the Year max: {month_of_the_year_max}")  
        print(f"Week of the Month max: {week_of_the_month_max}")
        print(lag_1_average)
        print(lag_7_average)
        print(roll_mean_7_average)
        print(roll_std_7_average)
        pred=0
        if input1=="Amoxicillin_500mg":
            model = load("rf_model_amoxicillin.joblib")
            prediction = model.predict([[flu_index_average, season_average, week_of_the_month_max,
                                   month_of_the_year_max,
                                   lag_1_average, lag_7_average,
                                   roll_mean_7_average, roll_std_7_average ]])
            pred = inv_boxcox(prediction[0], lambda_value)
            print(f"Prediction for week {i+1}: {pred}")
        elif input1=="Atorvastatin_20mg":
            model = load("rf_model_atorvastatin.joblib")
            prediction = model.predict([[flu_index_average, season_average, week_of_the_month_max,
                                   month_of_the_year_max,
                                   lag_1_average, lag_7_average,
                                   roll_mean_7_average, roll_std_7_average ]])
            pred = inv_boxcox(prediction[0], lambda_value)
            print(f"Prediction for week {i+1}: {pred}")
        elif input1=="Insulin_Glargine":
            model = load("lgb_model_insulin_glargine.joblib")
            prediction = model.predict([[flu_index_average, season_average, week_of_the_month_max,
                                   month_of_the_year_max,
                                   lag_1_average, lag_7_average,
                                   roll_mean_7_average, roll_std_7_average ]])
            pred = np.expm1(prediction[0])
            print(f"Prediction for week {i+1}: {pred}")
        elif input1=="Surgical_Gloves_Box":
            model = load("lgb_model_surgical_gloves_box.joblib") 
            prediction = model.predict([[flu_index_average, season_average, week_of_the_month_max,
                                   month_of_the_year_max,
                                   lag_1_average, lag_7_average,
                                   roll_mean_7_average, roll_std_7_average ]])
            pred = np.expm1(prediction[0])
            print(f"Prediction for week {i+1}: {pred}")
        elif input1=="Surgical_Masks_Box":
            model = load("xg_model_surgical_masks_box.joblib")  
            prediction = model.predict([[flu_index_average, season_average, week_of_the_month_max,
                                   month_of_the_year_max,
                                   lag_1_average, lag_7_average,
                                   roll_mean_7_average, roll_std_7_average ]])
            pred = inv_boxcox(prediction[0], lambda_value)
            print(f"Prediction for week {i+1}: {pred}")     
        output1=pred*7
        total_count=total_count+output1
    return total_count, total_count*price
# Define the GET route
@app.get("/process")
async def process_data(input1: str, input2: int):
    # Perform some operations (you can replace this with your logic)
    output1, output2 = calculation_logic(input1, input2)
    
    
    # Return the outputs as a dictionary (which FastAPI will return as JSON)
    return {"output1": output1, "output2": output2}