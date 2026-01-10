# Drought Monitoring System

An AI-powered agricultural drought monitoring and prediction system for India. This web application provides district-level drought predictions using machine learning and satellite data.

## Features

- 🌍 **District-wise Predictions**: Select any state and district in India to get drought predictions
- 📊 **Multi-indicator Analysis**: View rainfall (SPI), vegetation (NDVI), and soil moisture metrics
- 🗺️ **Interactive Map**: Explore drought conditions across India with an interactive heatmap
- 🎯 **Early Warning System**: Get 1-2 month advance predictions for drought conditions
- 📱 **Responsive Design**: Beautiful, modern UI that works on all devices
- 🔄 **Real-time Updates**: Access current and historical drought data

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: XGBoost
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Maps**: Leaflet.js
- **Icons**: Font Awesome

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd temp_droug
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data files are present**
   - `map_composite_drought_index.csv` - Main dataset with historical drought data
   - `model/xgb_drought_forecast_model.pkl` - Trained ML model for predictions

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

## Usage

### Making Predictions

1. Go to the home page
2. Select a **State** from the dropdown
3. Select a **District** (filtered by state)
4. Choose **Year** and **Month**
5. Click "Predict Drought Condition"
6. View results including:
   - Drought class (Normal, Mild, Moderate, Severe)
   - CDI (Composite Drought Index) value
   - Rainfall percentage and SPI-3 value
   - Vegetation percentage and NDVI anomaly
   - Soil moisture percentage and deficit

### Exploring the Map

1. Navigate to the **Map** page
2. Select **Year** and **Month** to view different time periods
   - Historical years (2015-2021): Shows real data from the dataset
   - Future years: Shows AI model predictions (marked with "PREDICTED" badge)
3. View district-wise drought conditions across India in heatmap format
4. Use filters to:
   - Filter by state
   - Filter by drought condition (Normal, Mild, Moderate, Severe)
5. Click on any district to see detailed information including:
   - CDI value and drought class
   - Rainfall percentage and SPI-3 value
   - Vegetation percentage and NDVI anomaly
   - Soil moisture percentage and deficit

## API Endpoints

- `GET /` - Home page
- `GET /map` - Interactive map page
- `GET /about` - About page
- `GET /team` - Team page
- `GET /contact` - Contact page
- `GET /api/districts/<state>` - Get districts for a state
- `POST /api/predict` - Get drought prediction
- `GET /api/map-data?year=YYYY&month=M` - Get map data for all districts for a specific year/month

## Project Structure

```
temp_droug/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── map_composite_drought_index.csv # Historical drought dataset
├── model/
│   └── xgb_drought_forecast_model.pkl  # Trained model
├── static/
│   ├── css/
│   │   └── style.css          # Stylesheet
│   └── js/
│       └── map.js             # Map JavaScript (if needed)
└── templates/
    ├── base.html              # Base template
    ├── home.html              # Home/prediction page
    ├── map.html               # Map visualization page
    ├── about.html             # About page
    ├── team.html              # Team page
    └── contact.html           # Contact page
```

## Drought Classification

- **Normal**: CDI > -0.5 (No drought risk)
- **Mild Drought**: -1.0 < CDI ≤ -0.5 (Slight water stress)
- **Moderate Drought**: -1.5 < CDI ≤ -1.0 (Moderate water stress)
- **Severe Drought**: CDI ≤ -1.5 (Critical water stress)

## Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is open source and available for educational and research purposes.

## Contact

- Email: drought.ai.team@gmail.com
- GitHub: github.com/drought-ai

## Acknowledgments

- Built for agricultural innovation and climate resilience
- Uses satellite data from various sources
- Powered by machine learning and AI

