import React, { useState, useEffect } from 'react';
import DataTable from '../components/DataTable';
import { api } from '../api';

export default function DataRecords() {
  const [data, setData] = useState({ records: [], total: 0 });
  const [page, setPage] = useState(1);

  useEffect(() => {
    api.getDataRecords(page, 15).then(setData);
  }, [page]);

  return (
    <div className="flex flex-col gap-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-4xl font-bold tracking-tight mb-2">Commodity Price Records</h2>
          <p className="text-on-surface-variant font-medium">Comprehensive historical ledger with model validation.</p>
        </div>
        <div className="flex gap-4 items-center">
           <span className="text-sm font-medium">Page {page} of {Math.ceil(data.total / 15)}</span>
           <button 
             className="btn-primary py-1 px-4 text-sm"
             onClick={() => setPage(p => Math.max(1, p - 1))}
             disabled={page === 1}
           >Prev</button>
           <button 
             className="btn-primary py-1 px-4 text-sm"
             onClick={() => setPage(p => p + 1)}
             disabled={page * 15 >= data.total}
           >Next</button>
        </div>
      </header>

      <section>
        <DataTable data={data.records} />
      </section>
    </div>
  );
}
