import React, { useState, useEffect } from 'react';
import StatCard from '../components/StatCard';
import { api } from '../api';

export default function Metrics() {
  const [metrics, setMetrics] = useState({ mae: 0, mse: 0, rmse: 0, r2_score: 0, commodity: "Global" });
  const [selectedComm, setSelectedComm] = useState("Global");
  const [commodities, setCommodities] = useState([]);

  useEffect(() => {
    api.getCommodities().then(list => setCommodities(["Global", ...list]));
  }, []);

  useEffect(() => {
    const fetchMetrics = async () => {
      const data = await api.getMetrics(selectedComm === "Global" ? null : selectedComm);
      setMetrics(data);
    };
    fetchMetrics();
  }, [selectedComm]);

  return (
    <div className="flex flex-col gap-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-4xl font-bold tracking-tight mb-2">Model Performance</h2>
          <p className="text-on-surface-variant font-medium">Validation metrics for our ensemble prediction engine.</p>
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-xs font-bold text-on-surface-variant uppercase tracking-tighter">Filter by Commodity</label>
          <select 
            className="input-field min-w-48 font-bold"
            value={selectedComm}
            onChange={(e) => setSelectedComm(e.target.value)}
          >
            {commodities.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <StatCard label="MAE" value={metrics.mae} unit="₹" subtext="Mean Absolute Error" />
        <StatCard label="MSE" value={metrics.mse.toFixed(2)} subtext="Mean Squared Error" />
        <StatCard label="RMSE" value={metrics.rmse.toFixed(2)} unit="₹" subtext="Root Mean Squared Error" />
        <StatCard label="R² Score" value={metrics.r2_score.toFixed(4)} subtext="Coefficient of Determination" />
      </div>

      <div className="card p-12 flex flex-col gap-8">
        <h3 className="text-2xl font-bold">{(metrics.commodity || "Global")} Performance Breakdown</h3>
        <p className="text-on-surface-variant leading-relaxed max-w-3xl">
          The **R² score of {metrics.r2_score.toFixed(4)}** for **{(metrics.commodity || "Global")}** indicates how much of the price variability is captured by our XGBoost model. 
          {selectedComm === "Global" 
            ? " Global metrics represent the aggregate accuracy across all tracked commodities."
            : ` For ${selectedComm}, the MAE of ₹${metrics.mae} suggests the model's average deviation from the actual market price.`
          }
        </p>
        
        <div className="h-64 section-bg rounded-xl flex items-center justify-center border-dashed border-2 border-primary/20">
          <p className="text-on-surface-variant italic">Showing active validation for {(metrics.commodity || "Global")}</p>
        </div>
      </div>
    </div>
  );
}
