import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

export default function PriceChart({ data, title }) {
  return (
    <div className="card p-8 flex flex-col gap-8 shadow-sm h-[450px]">
      <div className="flex justify-between items-center">
        <h3 className="text-xl font-bold">{title}</h3>
        <div className="flex gap-4 text-sm font-medium">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-primary rounded-full"></div>
            <span>Historical</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-tertiary-fixed-dim rounded-full"></div>
            <span>Predicted</span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 w-full relative">
        {data && data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00450d" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#00450d" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="#c0c9bb" strokeOpacity={0.2} />
              <XAxis 
                dataKey="date" 
                axisLine={false} 
                tickLine={false} 
                tick={{ fill: '#41493e', fontSize: 12, fontWeight: 500 }}
                dy={10}
              />
              <YAxis 
                axisLine={false} 
                tickLine={false} 
                tick={{ fill: '#41493e', fontSize: 12, fontWeight: 500 }}
                dx={-10}
                tickFormatter={(val) => `₹${val}`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  borderRadius: '12px', 
                  border: 'none', 
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                  padding: '12px'
                }}
                labelStyle={{ fontWeight: 'bold', marginBottom: '4px' }}
                itemStyle={{ fontSize: '12px', fontWeight: '500' }}
              />
              <Area 
                type="monotone" 
                dataKey="modal_price" 
                stroke="#00450d" 
                strokeWidth={3} 
                fillOpacity={1} 
                fill="url(#colorPrice)" 
                activeDot={{ r: 6, strokeWidth: 0, fill: '#00450d' }}
                name="Actual Price"
              />
              <Line 
                type="monotone" 
                dataKey="predicted_price" 
                stroke="#ffba38" 
                strokeWidth={3} 
                strokeDasharray="5 5"
                dot={false}
                name="Predicted Price"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/50 rounded-lg border border-outline-variant/30 text-on-surface-variant text-sm font-medium">
            No 30-day continuous price records available for this target configuration. Adjust the region or date.
          </div>
        )}
      </div>
    </div>
  );
}
