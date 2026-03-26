#!/bin/bash

# Navigate to the project directory
cd /Users/ronilshah/Projects/agri

# Ensure backend and frontend processes are killed on exit
trap 'kill %1; kill %2' SIGINT

echo "Starting FastAPI Backend..."
# Assuming venv is set up
source venv/bin/activate
# Start backend on default port 8000
python -m uvicorn backend.main:app --reload --port 8000 &

echo "Starting React Frontend..."
# Start Vite frontend
npm run dev -- --port 5173 &

echo "Both servers are running."
echo "Press Ctrl+C to stop."

# Wait for background processes to catch Ctrl+C
wait
