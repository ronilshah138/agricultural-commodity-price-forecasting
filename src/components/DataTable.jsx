import React from 'react';

export default function DataTable({ data }) {
  return (
    <div className="card shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-surface-container-low">
            <tr>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Date</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Commodity</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">State</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">District</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Min Price</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Max Price</th>
              <th className="px-8 py-4 text-xs font-bold text-on-surface-variant uppercase tracking-wider">Modal Price</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-on-surface-variant/10">
            {data.map((row) => (
              <tr key={row.id} className="hover:bg-surface-container/50 transition-all group">
                <td className="px-8 py-4 text-sm data-text text-on-surface-variant font-medium">{row.date}</td>
                <td className="px-8 py-4 text-sm font-bold text-on-surface">{row.commodity}</td>
                <td className="px-8 py-4 text-sm text-on-surface-variant font-medium">{row.state}</td>
                <td className="px-8 py-4 text-sm text-on-surface-variant font-medium">{row.district}</td>
                <td className="px-8 py-4 text-sm data-text font-medium text-on-surface-variant">₹{row.min_price.toLocaleString()}</td>
                <td className="px-8 py-4 text-sm data-text font-medium text-on-surface-variant">₹{row.max_price.toLocaleString()}</td>
                <td className="px-8 py-4 text-sm data-text font-bold text-primary">₹{row.modal_price.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-8 py-4 bg-surface-container-low flex justify-between items-center">
        <p className="text-xs text-on-surface-variant font-medium">Showing 50 of 50 records</p>
        <div className="flex gap-2">
          <button className="px-4 py-2 rounded-lg text-xs font-bold bg-surface-container-lowest text-on-surface-variant hover:text-primary transition-all">Prev</button>
          <button className="px-4 py-2 rounded-lg text-xs font-bold bg-surface-container-lowest text-on-surface-variant hover:text-primary transition-all">Next</button>
        </div>
      </div>
    </div>
  );
}
