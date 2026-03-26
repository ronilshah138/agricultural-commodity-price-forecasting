import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, TrendingUp, BarChart3, Binary, Table, Leaf } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const cn = (...inputs) => twMerge(clsx(inputs));

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/trends', label: 'Price Trends', icon: TrendingUp },
  { path: '/metrics', label: 'Performance', icon: BarChart3 },
  { path: '/comparison', label: 'Model Comparison', icon: Binary },
  { path: '/data', label: 'Price Records', icon: Table },
];

export default function Sidebar() {
  return (
    <aside className="w-72 min-h-screen section-bg p-8 flex flex-col gap-12 sticky top-0">
      <div className="flex items-center gap-3 px-2">
        <div className="bg-primary p-2 rounded-lg">
          <Leaf className="text-on-primary w-6 h-6" />
        </div>
        <h1 className="text-2xl font-bold tracking-tighter">AgroDash</h1>
      </div>

      <nav className="flex flex-col gap-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-4 px-4 py-3 rounded-lg transition-all font-medium",
                isActive 
                  ? "bg-surface-container-lowest text-primary shadow-sm" 
                  : "text-on-surface-variant hover:bg-surface-variant/30"
              )
            }
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto p-4 card bg-primary/5">
        <p className="text-sm font-bold text-primary mb-1">Digital Agronomist</p>
        <p className="text-xs text-on-surface-variant leading-relaxed">
          Premium analytical platform for agricultural commodity predictions.
        </p>
      </div>
    </aside>
  );
}
