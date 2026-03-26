import React, { useState, useEffect } from 'react';
import ComparisonTable from '../components/ComparisonTable';
import { api } from '../api';

export default function Comparison() {
  const [data, setData] = useState([]);

  useEffect(() => {
    api.getComparison().then(raw => {
      if (raw.length < 2) return;
      const arima = raw[0];
      const xgboost = raw[1];
      
      const pivoted = [
        { metric: "MAE (Error)", arima: arima.mae, xgboost: xgboost.mae },
        { metric: "RMSE", arima: arima.rmse, xgboost: xgboost.rmse },
        { metric: "R² Accuracy", arima: arima.r2, xgboost: xgboost.r2 },
        { metric: "Training Time", arima: arima.training_time, xgboost: xgboost.training_time },
      ];
      setData(pivoted);
    });
  }, []);

  return (
    <div className="flex flex-col gap-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header>
        <h2 className="text-4xl font-bold tracking-tight mb-2">Model Comparison</h2>
        <p className="text-on-surface-variant font-medium">Contrasting classical ARIMA against hybrid ML approaches.</p>
      </header>
      
      <section>
        <ComparisonTable data={data} />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="card p-8">
          <h4 className="text-lg font-bold mb-4">ARIMA Performance</h4>
          <p className="text-on-surface-variant text-sm leading-relaxed mb-4">
            Excels in capturing purely seasonal components with low data volume. However, it fails to account for exogenous shocks like fuel price spikes or sudden policy changes.
          </p>
          <div className="flex items-center gap-2 text-primary font-bold">
            <div className="w-2 h-2 bg-primary rounded-full"></div>
            <span>Best for Short-term Linear Trends</span>
          </div>
        </div>
        <div className="card p-8">
          <h4 className="text-lg font-bold mb-4">XGBoost (AgroDash Default)</h4>
          <p className="text-on-surface-variant text-sm leading-relaxed mb-4">
            Our optimized gradient boosting model integrates weather data, soil moisture levels, and market sentiment to provide a robust non-linear forecast.
          </p>
          <div className="flex items-center gap-2 text-primary font-bold">
            <div className="w-2 h-2 bg-primary rounded-full"></div>
            <span>Active Validation: High Accuracy</span>
          </div>
        </div>
      </section>
    </div>
  );
}
