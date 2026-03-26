import sys
import os
sys.path.append(os.getcwd())
try:
    from backend.ml.predict import make_predictions
    res = make_predictions("Wheat", "Maharashtra", "", "", "2025-12-30", "2026-01-05")
    print("SUCCESS:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
