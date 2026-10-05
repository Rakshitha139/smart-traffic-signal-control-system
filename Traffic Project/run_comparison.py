"""
Batch runner to accumulate traffic data over multiple simulation runs
"""
import subprocess
import time
import os
import shutil
from datetime import datetime

NUM_RUNS = 10
RUN_DURATION = 300
OUTPUT_DIR = "traffic_data"

def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists("traffic_log.csv"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(OUTPUT_DIR, f"traffic_log_backup_{timestamp}.csv")
        shutil.copy2("traffic_log.csv", backup_path)
        print(f"📁 Backed up to: {backup_path}")

def run_simulation(run_number):
    print(f"\n{'='*50}")
    print(f"🔄 Running simulation #{run_number}...")
    print(f"{'='*50}")
    
    try:
        process = subprocess.Popen(
            ["python", "simulation_v2.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=RUN_DURATION + 30)
        if process.returncode == 0:
            print(f"✅ Simulation #{run_number} completed")
        else:
            print(f"⚠️ Simulation #{run_number} had issues")
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"⏰ Simulation #{run_number} timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(2)

def main():
    print("🚦 Traffic Data Accumulator")
    print(f"   Running {NUM_RUNS} simulations, {RUN_DURATION}s each")
    setup()
    
    for i in range(1, NUM_RUNS + 1):
        run_simulation(i)
        if i < NUM_RUNS:
            print("⏳ Waiting 5 seconds...")
            time.sleep(5)
    
    print("\n🎉 All simulations complete!")
    print("   Run: python train_model.py to train the model.")

if __name__ == "__main__":
    main()
