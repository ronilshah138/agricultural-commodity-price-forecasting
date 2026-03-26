import sys
import os
import traceback

sys.path.append(os.getcwd())
try:
    from backend.ml.predict import make_predictions
    res = make_predictions('Fish', 'NCT of Delhi', '', '', '2025-12-30', '2026-01-05')
    print("SUCCESS:", res)
except Exception as e:
    print("FATAL ERROR:")
    traceback.print_exc()
