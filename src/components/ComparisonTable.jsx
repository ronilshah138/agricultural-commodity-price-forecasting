import React from 'react';

export default function ComparisonTable({ data }) {
  return (
    <div className="card shadow-sm overflow-hidden">
      <table className="w-full text-left">
        <thead className="bg-surface-container-low">
          <tr>
            <th className="px-8 py-6 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Model Parameter</th>
            <th className="px-8 py-6 text-xs font-bold text-primary uppercase tracking-wider">ARIMA (Classical)</th>
            <th className="px-8 py-6 text-xs font-bold text-tertiary uppercase tracking-wider">XGBoost (ML Hybrid)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-on-surface-variant/10">
          {data.map((row, idx) => (
            <tr key={idx} className="hover:bg-surface-container/30 transition-all">
              <td className="px-8 py-6 text-sm font-bold text-on-surface">{row.metric}</td>
              <td className="px-8 py-6 text-sm data-text font-medium text-on-surface-variant">{row.arima}</td>
              <td className="px-8 py-6 text-sm data-text font-bold text-on-surface">{row.xgboost}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
