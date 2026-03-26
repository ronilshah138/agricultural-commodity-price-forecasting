import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

export default function StatCard({ label, value, trend, unit = "", subtext = "" }) {
  return (
    <div className="card p-8 flex flex-col gap-4 shadow-sm hover:surface-container/10 transition-all border-none">
      <p className="text-on-surface-variant text-sm font-medium uppercase tracking-wider">{label}</p>
      <div className="flex items-baseline gap-2">
        <h3 className="text-4xl font-bold data-text">{value}{unit}</h3>
        {trend && (
          <span className={cn(
            "text-sm font-bold data-text",
            trend > 0 ? "text-primary" : "text-error"
          )}>
            {trend > 0 ? "+" : ""}{trend}%
          </span>
        )}
      </div>
      {subtext && <p className="text-xs text-on-surface-variant leading-relaxed">{subtext}</p>}
    </div>
  );
}
