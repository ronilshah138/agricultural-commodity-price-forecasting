import React, { useState, useEffect } from 'react';
import PriceChart from '../components/PriceChart';
import { api } from '../api';

export default function Trends() {
  const [commodities, setCommodities] = useState([]);
  const [states, setStates] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [markets, setMarkets] = useState([]);
  
  const [selectedComm, setSelectedComm] = useState("Rice");
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [selectedMarket, setSelectedMarket] = useState("");
  const [selectedDate, setSelectedDate] = useState("2023-07-28");
  const [combinedData, setCombinedData] = useState([]);

  useEffect(() => {
    api.getCommodities().then(comms => {
      setCommodities(comms);
      if (comms.includes("Tomato")) setSelectedComm("Tomato");
      else if (comms.length > 0) setSelectedComm(comms[0]);
    });
  }, []);

  useEffect(() => {
    if (selectedComm) {
      api.getStates(selectedComm).then(data => {
        setStates(data);
        if (data.length > 0) setSelectedState(data[0]);
        else setSelectedState("");
      });
    }
  }, [selectedComm]);

  useEffect(() => {
    if (selectedComm && selectedState) {
      api.getDistricts(selectedComm, selectedState).then(data => {
        setDistricts(data);
        if (data.length > 0) setSelectedDistrict(data[0]);
        else setSelectedDistrict("");
      });
    }
  }, [selectedComm, selectedState]);

  useEffect(() => {
    if (selectedComm && selectedState && selectedDistrict) {
      api.getMarkets(selectedComm, selectedState, selectedDistrict).then(data => {
        setMarkets(data);
        if (data.length > 0) setSelectedMarket(data[0]);
        else setSelectedMarket("");
      });
    }
  }, [selectedComm, selectedState, selectedDistrict]);

  useEffect(() => {
    const fetchTrends = async () => {
      if (!selectedState || !selectedDate) return;
      
      const baseDate = new Date(selectedDate);
      const end = selectedDate;
      
      const startObj = new Date(baseDate);
      startObj.setDate(startObj.getDate() - 90);
      const start = startObj.toISOString().split('T')[0];
      
      const predictEndObj = new Date(baseDate);
      predictEndObj.setDate(predictEndObj.getDate() + 7);
      const predictEnd = predictEndObj.toISOString().split('T')[0];
      
      const [historical, predictions] = await Promise.all([
        api.getPrices(selectedComm, selectedState, selectedDistrict, selectedMarket, start, end),
        api.getPredictions(selectedComm, selectedState, selectedDistrict, selectedMarket, end, predictEnd)
      ]);

      // Combine historical and predictions for the chart
      const formattedPredictions = predictions.map(p => ({
        ...p,
        modal_price: null, // Don't show solid line for predictions
      }));

      setCombinedData([...historical, ...formattedPredictions]);
    };
    
    fetchTrends();
  }, [selectedComm, selectedState, selectedDistrict, selectedMarket, selectedDate]);

  return (
    <div className="flex flex-col gap-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
        <div>
          <h2 className="text-4xl font-bold tracking-tight mb-2">Historical & Predicted Trends</h2>
          <p className="text-on-surface-variant font-medium">90-day trajectory analysis with 7-day forecasting.</p>
        </div>
        <div className="flex flex-wrap gap-4 w-full md:w-auto">
          <select 
            className="input-field flex-grow md:min-w-40 font-bold"
            value={selectedComm}
            onChange={(e) => setSelectedComm(e.target.value)}
          >
            {commodities.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <select 
            className="input-field flex-grow md:min-w-40 font-bold"
            value={selectedState}
            onChange={(e) => setSelectedState(e.target.value)}
          >
            {states.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select 
            className="input-field flex-grow md:min-w-40 font-bold"
            value={selectedDistrict}
            onChange={(e) => setSelectedDistrict(e.target.value)}
            disabled={districts.length === 0}
          >
            <option value="">Select District</option>
            {districts.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
          <select 
            className="input-field flex-grow md:min-w-40 font-bold"
            value={selectedMarket}
            onChange={(e) => setSelectedMarket(e.target.value)}
            disabled={markets.length === 0}
          >
            <option value="">Select Market</option>
            {markets.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <input 
            type="date"
            className="input-field flex-grow md:min-w-40 font-bold"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
          />
        </div>
      </header>

      <div className="grid grid-cols-1 gap-12">
        <PriceChart data={combinedData} title={`${selectedComm} - ${selectedMarket || selectedDistrict || selectedState} (7-Day Forecast)`} />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="card p-8 section-bg">
            <h4 className="text-lg font-bold mb-4">Market Drivers</h4>
            <ul className="flex flex-col gap-4 text-on-surface-variant text-sm">
              <li className="flex justify-between">
                <span>Seasonal Demand</span>
                <span className="font-bold text-primary">+12% Impact</span>
              </li>
              <li className="flex justify-between">
                <span>Fuel Costs</span>
                <span className="font-bold text-error">+4% Impact</span>
              </li>
              <li className="flex justify-between">
                <span>Rainfall (Expected)</span>
                <span className="font-bold text-primary">-2% Impact</span>
              </li>
            </ul>
          </div>
          <div className="card p-8 flex flex-col justify-center gap-2">
            <p className="text-primary font-bold text-xl">XGBoost Signal: Strong Momentum</p>
            <p className="text-on-surface-variant leading-relaxed text-sm">
              The model indicates persistent price growth for {selectedComm} in {selectedState} based on thinning historical buffers and rising harvesting costs.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
