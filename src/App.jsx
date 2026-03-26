import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Trends from './pages/Trends';
import Metrics from './pages/Metrics';
import Comparison from './pages/Comparison';
import DataRecords from './pages/DataRecords';

function App() {
  return (
    <div className="flex bg-surface min-h-screen">
      <Sidebar />
      <main className="flex-1 p-12 max-w-7xl mx-auto w-full">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/comparison" element={<Comparison />} />
          <Route path="/data" element={<DataRecords />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
