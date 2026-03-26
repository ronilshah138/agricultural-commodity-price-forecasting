export const COMMODITIES = ["Wheat", "Rice", "Maize", "Soybean", "Sugarcane"];
export const REGIONS = ["North India", "South India", "East India", "West India", "Central India"];

export const MOCK_TRENDS = [
  { date: "2024-01-01", actual: 2450, predicted: 2440 },
  { date: "2024-02-01", actual: 2520, predicted: 2510 },
  { date: "2024-03-01", actual: 2480, predicted: 2490 },
  { date: "2024-04-01", actual: 2600, predicted: 2580 },
  { date: "2024-05-01", actual: 2650, predicted: 2630 },
  { date: "2024-06-01", actual: 2720, predicted: 2700 },
  { date: "2024-07-01", actual: null, predicted: 2780 },
  { date: "2024-08-01", actual: null, predicted: 2850 },
  { date: "2024-09-01", actual: null, predicted: 2820 },
];

export const MOCK_METRICS = {
  mae: 42.5,
  mse: 2150.8,
  rmse: 46.3,
  r2: 0.94,
};

export const MOCK_COMPARISON = [
  { metric: "Training Time", arima: "0.2s", xgboost: "1.5s" },
  { metric: "Prediction Delay", arima: "0.01s", xgboost: "0.05s" },
  { metric: "Accuracy (R²)", arima: "0.88", xgboost: "0.94" },
  { metric: "Robustness", arima: "Medium", xgboost: "High" },
  { metric: "Data Complexity", arima: "Linear", xgboost: "Non-Linear" },
];

export const MOCK_DATA_RECORDS = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
  commodity: COMMODITIES[i % COMMODITIES.length],
  region: REGIONS[i % REGIONS.length],
  price: Math.floor(Math.random() * 500) + 2000,
  predictedPrice: Math.floor(Math.random() * 500) + 2000,
  confidence: (Math.random() * 0.1 + 0.85).toFixed(2),
}));
