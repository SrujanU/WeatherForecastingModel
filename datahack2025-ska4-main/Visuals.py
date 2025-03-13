import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import seaborn as sns
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Sample Data Loading
train_file_path = "training_data.csv"
train_df = pd.read_csv(train_file_path)

# Feature Engineering
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

X_retrain = train_df[["pressure", "air_temp", "ground_temp", 'temp_diff', 'pressure_change', 
                      'severe_weather_flag', 'rolling_avg_windspeed', 'rolling_max_windspeed', 
                      'windspeed_lag1', 'pressure_lag1', 'wind_pressure_interaction', 
                      'event_duration', 'event_intensity']]
y_retrain = train_df["windspeed"]

# Splitting Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_retrain, y_retrain, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions for Wind Speed (Sample for one event, extendable for 10)
event_predictions = []
for event_num in range(1, 11):  # We have event_1.csv to event_10.csv
    event_file_path = f"event_" + str(event_num) + ".csv"
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

    # Predicting Wind Speed using updated model
    wind_speed_predictions = rf_model.predict(ganopolis_df[["pressure", "air_temp", "ground_temp", 'temp_diff', 'pressure_change', 
                    'severe_weather_flag', 'rolling_avg_windspeed', 'rolling_max_windspeed', 
                    'windspeed_lag1', 'pressure_lag1', 'wind_pressure_interaction', 
                    'event_duration', 'event_intensity']])

# Reshape predictions (5 days, 24 hours per day)
reshaped_predictions = np.reshape(wind_speed_predictions, (5, 24))

# Function to Convert RGB Tuple to Plotly RGB String
def rgb_to_str(rgb_tuple):
    return f'rgb({int(rgb_tuple[0] * 255)}, {int(rgb_tuple[1] * 255)}, {int(rgb_tuple[2] * 255)})'

# Create a Plotly Figure
# Create a Plotly Figure with dashed lines between each day
def create_wind_speed_figure(predictions):
    fig = go.Figure()
    cmap = sns.color_palette("viridis", 5)
    
    for day in range(5):
        fig.add_trace(go.Scatter(
            x=list(range(1 + day * 24, 25 + day * 24)),
            y=predictions[day],
            mode='lines+markers',
            name=f"Day {day + 1}",
            line=dict(color=rgb_to_str(cmap[day]), width=3),  # Convert RGB to valid string
            marker=dict(size=6)
        ))
        # Add a vertical dashed line after every 24-hour mark to split days
        fig.add_vline(
            x=24 + day * 24, 
            line=dict(dash="dash", color='black', width=2),
            annotation_text=f"Day {day + 1}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title="Wind Speed Forecast",
        xaxis_title="Hour",
        yaxis_title="Wind Speed (m/s)",
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        annotations=[
            # Optional: Add a label to mark days explicitly in the graph
            dict(
                x=12, y=max(predictions.flatten()), 
                xref="x", yref="y", 
                text="Day 1", showarrow=False, 
                font=dict(size=12, color="white"), 
                align="center"
            ),
            dict(
                x=36, y=max(predictions.flatten()), 
                xref="x", yref="y", 
                text="Day 2", showarrow=False, 
                font=dict(size=12, color="white"), 
                align="center"
            ),
            dict(
                x=60, y=max(predictions.flatten()), 
                xref="x", yref="y", 
                text="Day 3", showarrow=False, 
                font=dict(size=12, color="white"), 
                align="center"
            ),
            dict(
                x=84, y=max(predictions.flatten()), 
                xref="x", yref="y", 
                text="Day 4", showarrow=False, 
                font=dict(size=12, color="white"), 
                align="center"
            ),
            dict(
                x=108, y=max(predictions.flatten()), 
                xref="x", yref="y", 
                text="Day 5", showarrow=False, 
                font=dict(size=12, color="white"), 
                align="center"
            ),
        ]
    )
    
    return fig

# Initialize Dash App
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1("Wind Speed Forecasting and Analysis", style={'textAlign': 'center', 'color': '#f5a623'}),
    
    # Dropdown for selecting event number
    html.Div([
        html.Label("Select Event", style={'color': '#ffffff'}),
        dcc.Dropdown(
            id='event-dropdown',
            options=[{'label': f'Event {i}', 'value': i} for i in range(1, 11)],
            value=1,  # Default event
            style={'width': '50%', 'margin': '20px auto'}
        ),
    ]),
    
    # Graph for wind speed forecasting
    dcc.Graph(
        id='wind-speed-forecast',
        figure=create_wind_speed_figure(reshaped_predictions)
    ),
])

# Callback to update graph based on event selection
@app.callback(
    Output('wind-speed-forecast', 'figure'),
    Input('event-dropdown', 'value')
)
def update_graph(event_number):
    # Re-load and filter the selected event's data
    event_file_path = f"event_{event_number}.csv"
    event_df = pd.read_csv(event_file_path)
    ganopolis_df = event_df[event_df['city'] == 'GANopolis'].sort_values(by=['hour'])

    # Weather-Related Features and Predictions
    ganopolis_df['temp_diff'] = ganopolis_df['air_temp'] - ganopolis_df['ground_temp']
    ganopolis_df['pressure_change'] = ganopolis_df['pressure'].diff()  
    ganopolis_df['severe_weather_flag'] = (ganopolis_df['windspeed'] > 0.15).astype(int)
    ganopolis_df['rolling_avg_windspeed'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).mean()
    ganopolis_df['rolling_max_windspeed'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).max()
    ganopolis_df['windspeed_lag1'] = ganopolis_df['windspeed'].shift(1)
    ganopolis_df['pressure_lag1'] = ganopolis_df['pressure'].shift(1)
    ganopolis_df['wind_pressure_interaction'] = ganopolis_df['windspeed'] * ganopolis_df['pressure']
    ganopolis_df['event_duration'] = ganopolis_df['severe_weather_flag'].rolling(window=6, min_periods=1).sum()
    ganopolis_df['event_intensity'] = ganopolis_df['windspeed'].rolling(window=6, min_periods=1).max()

    # Predicting Wind Speed using updated model
    wind_speed_predictions = rf_model.predict(ganopolis_df[["pressure", "air_temp", "ground_temp", 'temp_diff', 'pressure_change', 
                    'severe_weather_flag', 'rolling_avg_windspeed', 'rolling_max_windspeed', 
                    'windspeed_lag1', 'pressure_lag1', 'wind_pressure_interaction', 
                    'event_duration', 'event_intensity']])

    # Reshape predictions (5 days, 24 hours per day)
    reshaped_predictions = np.reshape(wind_speed_predictions, (5, 24))

    return create_wind_speed_figure(reshaped_predictions)

if __name__ == '__main__':
    app.run_server(debug=True)
