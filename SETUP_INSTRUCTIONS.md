# Traffic Prediction Web Application - Setup Instructions

## Overview
This is a machine learning-powered web application that predicts traffic volume based on weather conditions, time, date, and other factors. It uses a Neural Network (MLPRegressor) trained on historical traffic data.

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Navigate to the Project Directory
```bash
cd "Traffic_Prediction-main\Traffic_Prediction-main\mp"
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv traffic_env

# Activate virtual environment
# On Windows:
traffic_env\Scripts\activate
# On macOS/Linux:
source traffic_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000` or `http://127.0.0.1:5000`

## How to Use the Application

1. **Open your web browser** and go to `http://localhost:5000`

2. **Fill in the prediction form:**
   - **Date**: Select the date for prediction
   - **Day**: Choose the day of the week
   - **Time**: Enter the time (24-hour format)
   - **Temperature**: Enter temperature in Kelvin (e.g., 288 for ~15°C)
   - **Is Holiday**: Select if it's a holiday
   - **Climate Condition**: Choose weather type (Clear, Clouds, Rain, etc.)
   - **Weather Description**: Select detailed weather description

3. **Click "Predict"** to get the traffic volume prediction

4. **View Results**: The application will show:
   - Predicted traffic volume (vehicles per hour)
   - Summary of input parameters
   - Weather-based background animation

## Features

- **Machine Learning Prediction**: Uses MLPRegressor neural network
- **Interactive UI**: Weather-based dynamic backgrounds
- **Real-time Predictions**: Instant traffic volume forecasting
- **Historical Data**: Trained on comprehensive traffic dataset

## Temperature Conversion
The model expects temperature in Kelvin:
- 0°C = 273.15K
- 15°C = 288.15K  
- 25°C = 298.15K

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: If port 5000 is busy, modify `app.py`:
   ```python
   app.run(debug=True, port=5001)
   ```

3. **File Not Found Errors**: Ensure you're running from the `mp` directory

4. **Model Training Takes Time**: The first run may take a few minutes to train the model

### Performance Notes:
- Model trains on 10,000 sample records for faster startup
- Prediction accuracy depends on input data quality
- Weather conditions significantly impact predictions

## Project Structure
```
mp/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/               # Static files (CSS, images, data)
│   ├── Train.csv         # Training dataset
│   ├── *.gif, *.jpg      # Weather background images
│   └── *.png             # Chart images
└── templates/            # HTML templates
    ├── index.html        # Main input form
    └── output.html       # Results page
```

## Dataset Information
- **Records**: 33,000+ traffic observations
- **Features**: Weather, time, temperature, holidays
- **Target**: Traffic volume (vehicles per hour)
- **Time Range**: 2012-2018 historical data

## Next Steps
- Experiment with different weather conditions
- Try various times of day to see traffic patterns
- Compare holiday vs non-holiday predictions
- Test extreme weather scenarios

For issues or questions, check the console output for detailed error messages.
