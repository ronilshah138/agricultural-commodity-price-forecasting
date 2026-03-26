# AgroDash: The Digital Agronomist - Project Summary

## Overview
**AgroDash** is a full-stack web application designed to act as a "Digital Agronomist," providing precision data science applied to organic agriculture. Its core functionality revolves around predicting pricing trends for various agricultural commodities across different states and districts, acting as an advanced tracking and prediction dashboard for farmers, traders, and agricultural stakeholders.

## Technical Stack
- **Frontend**: React (v19), Vite, React Router DOM, Recharts (for data visualization), Tailwind CSS, Lucide React (icons).
- **Backend**: FastAPI (Python), Uvicorn.
- **Machine Learning**: XGBoost Regressor (with GPU/CPU support), Scikit-Learn, Pandas.
- **Database**: SQLite (via SQLAlchemy), Pydantic for data validation.

## Core Architecture & Features
### 1. Machine Learning Pipeline
- **Predictive Modeling**: Uses **XGBoost Regressor** to predict the `modal_price` of commodities.
- **Features Used**: Commodity code, state code, district code, market code, along with time-series lags (`lag_1`, `lag_2`, `lag_3`) and a rolling mean (`rolling_mean_3`).
- **Metrics Tracking**: Evaluates models using MAE, MSE, RMSE, and $R^2$ scores, storing both global metrics and per-commodity metrics to track the model's accuracy on specific items.

### 2. Backend Services (FastAPI)
The backend is highly modularized with specific routers handling different API endpoints:
- **`commodities`**: Endpoints for retrieving available crops and states.
- **`predictions`**: Endpoints for requesting price predictions based on model outputs.
- **`metrics`` & `model`**: Endpoints outlining model health, training states, and $R^2$ performance metrics.
- **`data`**: Internal endpoints for serving preprocessed state, district, and market lists.

### 3. Frontend Dashboard UI
The frontend strictly adheres to a premium **"Digital Agronomist"** design system (Material 3 Tonal Range):
- **Aesthetics**: Glassmorphism, intentional asymmetry, and depth via tonal layering rather than rigid grids or heavy shadows.
- **Color Palette**: Highlights natural tones such as "Fertile Soil" (Deep Green `#00450d`), "Golden Harvest" (Yellow/Brown accents), and off-white bases to reduce eye strain.
- **Visualizations**: Extensive use of `Recharts`. Historical data is displayed with reliable solid lines, while predictive data is showcased with subtle dashed yellow lines, allowing users to intuitively differentiate past realities from future forecasts.

## Running the Application
A unified `run_all.sh` script automates the launching process, concurrently spinning up the FastAPI backend on port 8000 and the Vite development server on port 5173. 
