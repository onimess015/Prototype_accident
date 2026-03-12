#!/usr/bin/env python
"""Monitor training progress and check for completion."""
import time
import subprocess
import sys
from pathlib import Path

model_file = Path(
    r"c:\Prototype(accident_predictor)\accident_prediction_project\models\accident_model.pkl"
)
original_modify_time = model_file.stat().st_mtime if model_file.exists() else 0

print(f"Monitoring training... (original change time: {original_modify_time})")
print(f"Will check every 30 seconds")
print("=" * 60)

for attempt in range(120):  # Check for up to 60 minutes
    try:
        # Count Python processes
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object Count",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Extract count from output
        for line in result.stdout.split("\n"):
            if line.strip().isdigit():
                count = int(line.strip())
                cur_time = time.strftime("%H:%M:%S")
                print(f"[{cur_time}] Python processes: {count}", end="")

                # Check model file
                if model_file.exists():
                    current_modify_time = model_file.stat().st_mtime
                    if current_modify_time > original_modify_time:
                        print(" --> MODEL UPDATED!", flush=True)
                    else:
                        print()
                else:
                    print()
                break

        if count == 0 and attempt > 5:  # Give it at least 30 seconds
            print("\n[COMPLETE] Training finished! Model file should be updated.")

            # Try to load and display results
            if model_file.exists():
                print(
                    f"\nModel file size: {model_file.stat().st_size / (1024*1024):.2f} MB"
                )
                print(f"Model file modified: {time.ctime(model_file.stat().st_mtime)}")
            break

    except Exception as e:
        print(f"[Error checking: {e}]")

    time.sleep(30)

print("\nMonitoring complete.")
