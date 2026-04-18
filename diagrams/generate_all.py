"""Generate all figures for the DL course."""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

scripts = sorted(SCRIPTS_DIR.glob("lec*_figures.py"))

if not scripts:
    print("No figure scripts found.")
    sys.exit(0)

failed = []
for script in scripts:
    print(f"Running {script.name}...")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        failed.append(script.name)
    else:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

if failed:
    print(f"\nFailed scripts: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nAll figure scripts completed successfully.")
