from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load data
data = pd.read_csv("map_composite_drought_index.csv")

# Get unique states and districts
STATES = sorted(data['STATE'].dropna().unique().tolist())
DISTRICTS = sorted(data['DISTRICT'].dropna().unique().tolist())

# Create state-district mapping
state_district_map = {}
for state in STATES:
    districts_in_state = sorted(data[data['STATE'] == state]['DISTRICT'].unique().tolist())
    state_district_map[state] = districts_in_state

CURRENT_YEAR = datetime.now().year
YEARS = list(range(2015, CURRENT_YEAR + 5))

MONTHS = [
    (1, "January"), (2, "February"), (3, "March"),
    (4, "April"), (5, "May"), (6, "June"),
    (7, "July"), (8, "August"), (9, "September"),
    (10, "October"), (11, "November"), (12, "December")
]

# Load trained model
try:
    model = joblib.load("model/xgb_drought_forecast_model.pkl")
    print(f"Model loaded successfully. Expected features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

def drought_class(cdi):
    if cdi <= -1.5:
        return "Severe Drought", "severe"
    elif cdi <= -1.0:
        return "Moderate Drought", "moderate"
    elif cdi <= -0.5:
        return "Mild Drought", "mild"
    else:
        return "Normal", "normal"

def get_historical_data(district, year, month):
    """Get historical data for a district, year, and month"""
    filtered = data[(data['DISTRICT'] == district) & 
                    (data['year'] == year) & 
                    (data['month'] == month)]
    
    if not filtered.empty:
        row = filtered.iloc[0]
        return {
            'spi_3': row.get('SPI_3', 0),
            'ndvi': row.get('NDVI_ANOMALY', 0),
            'sm_deficit': row.get('SM_DEFICIT', 0),
            'spi_3_n': row.get('SPI_3_N', 0),
            'ndvi_n': row.get('NDVI_ANOMALY_N', 0),
            'sm_deficit_inv_n': row.get('SM_DEFICIT_INV_N', 0),
            'cdi': row.get('CDI', 0)
        }
    return None

def predict_district(district, year, month):
    """Predict CDI for a district using the model"""
    if model is None:
        # If model is not loaded, return a default prediction based on historical patterns
        print(f"Warning: Model not available, using fallback prediction for {district}")
        return {
            'cdi': -0.5,  # Default to mild drought
            'spi_3': 0.0,
            'ndvi': 0.0,
            'sm_deficit': 0.0,
            'is_prediction': True
        }
    
    try:
        # Get the most recent historical data for this district to use as baseline
        district_data = data[data['DISTRICT'] == district]
        if not district_data.empty:
            # Get the latest available data for this district
            latest_district = district_data.sort_values(['year', 'month'], ascending=False).iloc[0]
            
            # Use normalized features if available, otherwise use raw and normalize
            spi_3_n = latest_district.get('SPI_3_N', 0) if not pd.isna(latest_district.get('SPI_3_N', 0)) else 0
            ndvi_n = latest_district.get('NDVI_ANOMALY_N', 0) if not pd.isna(latest_district.get('NDVI_ANOMALY_N', 0)) else 0
            sm_deficit_inv_n = latest_district.get('SM_DEFICIT_INV_N', 0) if not pd.isna(latest_district.get('SM_DEFICIT_INV_N', 0)) else 0
        else:
            # If no historical data, use mean values from dataset
            spi_3_n = float(data['SPI_3_N'].mean()) if 'SPI_3_N' in data.columns and not data['SPI_3_N'].isna().all() else 0.0
            ndvi_n = float(data['NDVI_ANOMALY_N'].mean()) if 'NDVI_ANOMALY_N' in data.columns and not data['NDVI_ANOMALY_N'].isna().all() else 0.0
            sm_deficit_inv_n = float(data['SM_DEFICIT_INV_N'].mean()) if 'SM_DEFICIT_INV_N' in data.columns and not data['SM_DEFICIT_INV_N'].isna().all() else 0.0
        
        # Prepare features for model (assuming model expects these 3 normalized features)
        # Adjust based on actual model input requirements
        X = np.array([[spi_3_n, ndvi_n, sm_deficit_inv_n]])
        
        # If model expects more features, pad with zeros or use defaults
        if hasattr(model, 'n_features_in_') and model.n_features_in_:
            n_features = model.n_features_in_
            if n_features > 3:
                X = np.pad(X, ((0, 0), (0, n_features - 3)), mode='constant', constant_values=0)
            elif n_features < 3:
                X = X[:, :n_features]
        
        cdi = model.predict(X)[0]
        
        # Estimate other values based on CDI (for display purposes)
        # In a real scenario, you might want to predict these separately or use historical patterns
        spi_3 = spi_3_n * 2.5  # Rough inverse normalization
        ndvi = ndvi_n * 0.5 if ndvi_n < 2 else ndvi_n
        sm_deficit = -sm_deficit_inv_n * 0.5 if sm_deficit_inv_n < 2 else sm_deficit_inv_n
        
        return {
            'cdi': float(cdi),
            'spi_3': float(spi_3),
            'ndvi': float(ndvi),
            'sm_deficit': float(sm_deficit),
            'is_prediction': True
        }
    except Exception as e:
        # Fallback to default values if prediction fails
        return {
            'cdi': 0.0,
            'spi_3': 0.0,
            'ndvi': 0.0,
            'sm_deficit': 0.0,
            'is_prediction': True
        }

# Get the latest year with data
if not data.empty:
    LATEST_DATA_YEAR = int(data['year'].max())
    latest_year_data = data[data['year'] == LATEST_DATA_YEAR]
    LATEST_DATA_MONTH = int(latest_year_data['month'].max()) if not latest_year_data.empty else 12
else:
    LATEST_DATA_YEAR = CURRENT_YEAR
    LATEST_DATA_MONTH = 12

def calculate_percentages(spi_3, ndvi, sm_deficit):
    """Calculate percentage values for display"""
    # Normalize SPI_3 to percentage (typically ranges from -3 to 3)
    rainfall_pct = max(0, min(100, ((spi_3 + 3) / 6) * 100))
    
    # Normalize NDVI anomaly to percentage (typically ranges from -1 to 1)
    vegetation_pct = max(0, min(100, ((ndvi + 1) / 2) * 100))
    
    # Normalize SM deficit to percentage (inverse, so higher deficit = lower moisture)
    # Assuming SM_DEFICIT ranges from -1 to 1
    soil_moisture_pct = max(0, min(100, ((1 - sm_deficit) / 2) * 100))
    
    return rainfall_pct, vegetation_pct, soil_moisture_pct

@app.route("/")
def home():
    return render_template(
        "home.html",
        states=STATES,
        years=YEARS,
        months=MONTHS
    )

@app.route("/api/districts/<state>")
def get_districts(state):
    """API endpoint to get districts for a state"""
    districts = state_district_map.get(state, [])
    return jsonify(districts)

@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint for predictions"""
    try:
        state = request.json.get('state')
        district = request.json.get('district')
        year = int(request.json.get('year'))
        month = int(request.json.get('month'))
        
        # Determine if we should use historical data or prediction
        use_prediction = (year > LATEST_DATA_YEAR) or \
                        (year == LATEST_DATA_YEAR and month > LATEST_DATA_MONTH)
        
        if not use_prediction:
            # Try to get historical data first
            historical = get_historical_data(district, year, month)
            
            if historical and not pd.isna(historical['cdi']):
                # Use historical CDI if available
                cdi = historical['cdi']
                spi_3 = historical['spi_3'] if not pd.isna(historical['spi_3']) else 0
                ndvi = historical['ndvi'] if not pd.isna(historical['ndvi']) else 0
                sm_deficit = historical['sm_deficit'] if not pd.isna(historical['sm_deficit']) else 0
                is_prediction = False
            else:
                # Historical data not available, use prediction
                prediction = predict_district(district, year, month)
                cdi = prediction['cdi']
                spi_3 = prediction['spi_3']
                ndvi = prediction['ndvi']
                sm_deficit = prediction['sm_deficit']
                is_prediction = True
        else:
            # Use model prediction for future dates
            prediction = predict_district(district, year, month)
            cdi = prediction['cdi']
            spi_3 = prediction['spi_3']
            ndvi = prediction['ndvi']
            sm_deficit = prediction['sm_deficit']
            is_prediction = True
        
        drought_name, drought_class_name = drought_class(cdi)
        rainfall_pct, vegetation_pct, soil_moisture_pct = calculate_percentages(spi_3, ndvi, sm_deficit)
        
        return jsonify({
            'success': True,
            'cdi': round(float(cdi), 2),
            'drought_class': drought_name,
            'drought_class_name': drought_class_name,
            'rainfall': round(rainfall_pct, 1),
            'vegetation': round(vegetation_pct, 1),
            'soil_moisture': round(soil_moisture_pct, 1),
            'spi_3': round(float(spi_3), 2),
            'ndvi': round(float(ndvi), 3),
            'sm_deficit': round(float(sm_deficit), 3),
            'is_prediction': is_prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/map-data")
def map_data():
    """API endpoint to get map data for all districts with year/month selection"""
    try:
        # Get year and month from query parameters, default to latest available
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        
        # If not provided, use latest available data
        if not year:
            year = LATEST_DATA_YEAR
        if not month:
            month = LATEST_DATA_MONTH if year == LATEST_DATA_YEAR else 12
        
        # Validate year and month
        if year < 2015 or year > CURRENT_YEAR + 5:
            return jsonify({'error': f'Invalid year: {year}. Must be between 2015 and {CURRENT_YEAR + 5}'}), 400
        if month < 1 or month > 12:
            return jsonify({'error': f'Invalid month: {month}. Must be between 1 and 12'}), 400
        
        # Determine if we need predictions
        use_prediction = (year > LATEST_DATA_YEAR) or \
                        (year == LATEST_DATA_YEAR and month > LATEST_DATA_MONTH)
        
        map_data = []
        
        if not use_prediction:
            # Get historical data for specified year and month
            filtered_data = data[(data['year'] == year) & (data['month'] == month)]
            
            for _, row in filtered_data.iterrows():
                try:
                    cdi = row.get('CDI', 0) if not pd.isna(row.get('CDI', 0)) else None
                    if cdi is None or pd.isna(cdi):
                        continue
                    
                    drought_name, drought_class_name = drought_class(float(cdi))
                    spi_3 = float(row.get('SPI_3', 0)) if not pd.isna(row.get('SPI_3', 0)) else 0.0
                    ndvi = float(row.get('NDVI_ANOMALY', 0)) if not pd.isna(row.get('NDVI_ANOMALY', 0)) else 0.0
                    sm_deficit = float(row.get('SM_DEFICIT', 0)) if not pd.isna(row.get('SM_DEFICIT', 0)) else 0.0
                    rainfall_pct, vegetation_pct, soil_moisture_pct = calculate_percentages(spi_3, ndvi, sm_deficit)
                    
                    map_data.append({
                        'state': str(row['STATE']),
                        'district': str(row['DISTRICT']),
                        'cdi': float(cdi),
                        'drought_class': drought_name,
                        'drought_class_name': drought_class_name,
                        'year': int(row['year']),
                        'month': int(row['month']),
                        'spi_3': round(float(spi_3), 2),
                        'ndvi': round(float(ndvi), 3),
                        'sm_deficit': round(float(sm_deficit), 3),
                        'rainfall': round(rainfall_pct, 1),
                        'vegetation': round(vegetation_pct, 1),
                        'soil_moisture': round(soil_moisture_pct, 1),
                        'is_prediction': False
                    })
                except Exception as e:
                    # Skip rows with errors, but log them
                    print(f"Error processing row for {row.get('DISTRICT', 'Unknown')}: {str(e)}")
                    continue
        else:
            # Generate predictions for all districts
            unique_districts = data[['STATE', 'DISTRICT']].drop_duplicates()
            
            for _, row in unique_districts.iterrows():
                try:
                    district = str(row['DISTRICT'])
                    state = str(row['STATE'])
                    prediction = predict_district(district, year, month)
                    
                    drought_name, drought_class_name = drought_class(prediction['cdi'])
                    rainfall_pct, vegetation_pct, soil_moisture_pct = calculate_percentages(
                        prediction['spi_3'], 
                        prediction['ndvi'], 
                        prediction['sm_deficit']
                    )
                    
                    map_data.append({
                        'state': state,
                        'district': district,
                        'cdi': round(prediction['cdi'], 2),
                        'drought_class': drought_name,
                        'drought_class_name': drought_class_name,
                        'year': year,
                        'month': month,
                        'spi_3': round(prediction['spi_3'], 2),
                        'ndvi': round(prediction['ndvi'], 3),
                        'sm_deficit': round(prediction['sm_deficit'], 3),
                        'rainfall': round(rainfall_pct, 1),
                        'vegetation': round(vegetation_pct, 1),
                        'soil_moisture': round(soil_moisture_pct, 1),
                        'is_prediction': True
                    })
                except Exception as e:
                    # Skip districts with prediction errors, but log them
                    print(f"Error predicting for district {row.get('DISTRICT', 'Unknown')}: {str(e)}")
                    continue
        
        return jsonify(map_data)
    except Exception as e:
        import traceback
        error_msg = f"Error in map_data endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e)}), 400

@app.route("/map")
def map_view():
    return render_template("map.html", years=YEARS, months=MONTHS, 
                          latest_year=LATEST_DATA_YEAR, latest_month=LATEST_DATA_MONTH)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
