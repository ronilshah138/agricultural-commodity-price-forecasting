const API_BASE = "http://localhost:8000";

export const api = {
  getStates: async (commodity) => {
    let url = `${API_BASE}/commodities/states`;
    if (commodity) url += `?commodity=${encodeURIComponent(commodity)}`;
    const res = await fetch(url);
    return res.json();
  },

  getCommodities: async () => {
    const res = await fetch(`${API_BASE}/commodities/`);
    return res.json();
  },

  getDistricts: async (commodity, state) => {
    let url = `${API_BASE}/commodities/districts?`;
    if (commodity) url += `commodity=${encodeURIComponent(commodity)}&`;
    if (state) url += `state=${encodeURIComponent(state)}`;
    const res = await fetch(url);
    return res.json();
  },

  getMarkets: async (commodity, state, district) => {
    let url = `${API_BASE}/commodities/markets?`;
    if (commodity) url += `commodity=${encodeURIComponent(commodity)}&`;
    if (state) url += `state=${encodeURIComponent(state)}&`;
    if (district) url += `district=${encodeURIComponent(district)}`;
    const res = await fetch(url);
    return res.json();
  },
  
  getPrices: async (commodity, region, district, market, start, end) => {
    let url = `${API_BASE}/commodities/${commodity}/prices?`;
    if (region) url += `region=${region}&`;
    if (district) url += `district=${district}&`;
    if (market) url += `market=${market}&`;
    if (start) url += `start_date=${start}&`;
    if (end) url += `end_date=${end}&`;
    const res = await fetch(url);
    return res.json();
  },
  
  getPredictions: async (commodity, region, district, market, start, end) => {
    const res = await fetch(`${API_BASE}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        commodity,
        region,
        district,
        market,
        start_date: start,
        end_date: end
      }),
    });
    return res.json();
  },
  
  getMetrics: async (commodity = null) => {
    let url = `${API_BASE}/metrics/`;
    if (commodity) url += `?commodity=${encodeURIComponent(commodity)}`;
    const res = await fetch(url);
    return res.json();
  },
  
  getComparison: async () => {
    const res = await fetch(`${API_BASE}/metrics/comparison`);
    return res.json();
  },
  
  getDataRecords: async (page = 1, limit = 10, commodity = "", state = "", date = "") => {
    let url = `${API_BASE}/data/?page=${page}&limit=${limit}`;
    if (commodity) url += `&commodity=${commodity}`;
    if (state) url += `&state=${state}`;
    if (date) url += `&date=${date}`;
    const res = await fetch(url);
    return res.json();
  },
  
  getModelStatus: async () => {
    const res = await fetch(`${API_BASE}/model/status`);
    return res.json();
  }
};
