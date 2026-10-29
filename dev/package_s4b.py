"""Build dist/s4b_claude_prior.zip from my_model5 (S4b).

Needs my_model5/submission_config.json with your SPEND-CAPPED key:
    cp my_model5/submission_config.example.json my_model5/submission_config.json
    (then paste the key into it yourself)
The key is never printed. Never commit submission_config.json.

Run:  .venv/bin/python dev/package_s4b.py
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
FOLDER = KIT / "my_model5"
OUT = KIT / "dist" / "s4b_claude_prior.zip"
CONCURRENCY = 8   # parallel workers asked from the platform (8 threads gave 0 failures locally)

config_path = FOLDER / "submission_config.json"
if not config_path.exists():
    sys.exit("missing my_model5/submission_config.json (copy the .example file and add the key)")

key = json.loads(config_path.read_text())["api_keys"]["ANTHROPIC_API_KEY"]
print("key present:", "yes" if key.startswith("sk-ant-") else "NO - check the file")
if not key.startswith("sk-ant-"):
    sys.exit(1)

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(FOLDER / "model.py", "model.py")
    zf.write(config_path, "submission_config.json")
    zf.writestr("prediction_config.json", json.dumps({"concurrency": CONCURRENCY}))
print("saved", OUT)

subprocess.run([sys.executable, str(KIT / "paiec_baseline" / "check_submission_zip.py"), str(OUT)])
