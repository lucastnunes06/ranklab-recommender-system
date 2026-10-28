from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranklab.training import train
if __name__ == "__main__":
    results = train()
    print(json.dumps(results, ensure_ascii=False, indent=2))
