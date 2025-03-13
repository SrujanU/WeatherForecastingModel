import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Training Data
train_file_path = "/Users/sathvikkatta/Documents/Cheveron Datafest 25/training_data.csv"
train_df = pd.read_csv(train_file_path)

# Feature Engineering for Training Data
train_df['temp_diff'] = train_df['air_temp'] - train_df['ground_temp']
train_df['pressure_change'] = train_df['pressure'].diff()
train_df['severe_weather_flag'] = (train_df['windspeed'] > 0.15).astype(int)
train_df['rolling_avg_windspeed'] = train_df['windspeed'].rolling(window=6, min_periods=1).mean()
train_df['rolling_max_windspeed'] = train_df['windspeed'].rolling(window=6, min_periods=1).max()
train_df['windspeed_lag1'] = train_df['windspeed'].shift(1)
train_df['pressure_lag1'] = train_df['pressure'].shift(1)
train_df['wind_pressure_interaction'] = train_df['windspeed'] * train_df['pressure']
train_df['event_duration'] = train_df['severe_weather_flag'].rolling(window=6, min_periods=1).sum()
train_df['event_intensity'] = train_df['windspeed'].rolling(window=6, min_periods=1).max()

# Define features and target
X_retrain = train_df[["pressure", "air_temp", "ground_temp", 'temp_diff', 'pressure_change', 
                      'severe_weather_flag', 'rolling_avg_windspeed', 'rolling_max_windspeed', 
                      'windspeed_lag1', 'pressure_lag1', 'wind_pressure_interaction', 
                      'event_duration', 'event_intensity']]
y_retrain = train_df["windspeed"]

# Splitting Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_retrain, y_retrain, test_size=0.2, random_state=42)

# Train Random Forest Model for Wind Speed Prediction
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest Model MSE: {mse:.10f}")

# Processing Event Files
event_predictions = []
for event_num in range(1, 11):  # We have event_1.csv to event_10.csv
    event_file_path = f"/Users/sathvikkatta/Documents/Cheveron Datafest 25/event_" + str(event_num) + ".csv"
    event_df = pd.read_csv(event_file_path)

    # Filtering GANopolis Data
    ganopolis_df = event_df[event_df['city'] == 'GANopolis'].sort_values(by=['hour'])

    # Weather-Related Features
    ganopolis_df['temp_diff'] = ganopolis_df['air_temp'] - ganopolis_df['ground_temp']
    ganopolis_df['pressure_change'] = ganopolis_df['pressure'].diff()  # Change in pressure over time
    ganopolis_df['severe_weather_flag'] = (ganopolis_df['windspeed'] > 0.15).astype(int)

    # Rolling Statistics
    ganopolis_df['rolling_avg_windspeed'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).mean()
    ganopolis_df['rolling_max_windspeed'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).max()

    # Lag Features
    ganopolis_df['windspeed_lag1'] = ganopolis_df['windspeed'].shift(1)  # Windspeed 1 hour ago
    ganopolis_df['pressure_lag1'] = ganopolis_df['pressure'].shift(1)    # Pressure 1 hour ago

    # Interaction Features
    ganopolis_df['wind_pressure_interaction'] = ganopolis_df['windspeed'] * ganopolis_df['pressure']

    ganopolis_df['event_duration'] = ganopolis_df['severe_weather_flag'].rolling(window=6, min_periods=1).sum()
    ganopolis_df['event_intensity'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).max()

    #print(ganopolis_df)
    
    # Predicting Next 5 Days of Wind Speed (120 hours) using updated model
    wind_speed_predictions = rf_model.predict(ganopolis_df[["pressure", "air_temp", "ground_temp", 'temp_diff', 'pressure_change', 
                    'severe_weather_flag', 'rolling_avg_windspeed', 'rolling_max_windspeed', 
                    'windspeed_lag1', 'pressure_lag1', 'wind_pressure_interaction', 
                    'event_duration', 'event_intensity']])
    
    def calculate_damage(wind_speed_predictions):

        weighted_wind_damage = np.sum(wind_speed_predictions ** 2)

        # Apply wind duration threshold (only count severe winds exponentially)
        severe_winds = wind_speed_predictions[wind_speed_predictions > 0.2]  # Only strong winds
        severe_damage = np.sum(np.exp(severe_winds))  # Exponential damage scaling

        # Compute total damage
        total_damage = weighted_wind_damage + severe_damage

        return total_damage

    total_damage = calculate_damage(wind_speed_predictions)
    print(f"Refined Total Damage: {total_damage:.2f}")

    # Calculating Damage and Pricing
    #total_damage = np.sum(wind_speed_predictions)  # Simplified damage estimation
    #print(total_damage)
    optimal_price = 250 + (total_damage)/(2)
    profit = (10000 - 20*(optimal_price)) * (optimal_price - total_damage)


    # Saving Predictions
    output_dict = {
        "event_number": event_num,
        "optimal_price": optimal_price,
        "profit": profit,
        **{f"hour_{i+1}": wind_speed_predictions[i] for i in range(len(wind_speed_predictions))},
    }
    event_predictions.append(output_dict)

# Creating Final Submission CSV
submission_df = pd.DataFrame(event_predictions)
submission_file_path = "/Users/sathvikkatta/Documents/Cheveron Datafest 25/submission.csv"
submission_df.to_csv(submission_file_path, index=False)

submission_df.to_csv(submission_file_path, index=False, float_format="%.4f")


print(submission_df)

