import sys
import subprocess
import os

def run_script(script_path: str):
    print(f"\n▶️ Running {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Script {script_path} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"✅ Script {script_path} completed successfully.")
    

if __name__ == "__main__":
    scripts = [
        os.path.join('scripts', 'import_to_db.py'),
        os.path.join('scripts', 'feature_engineering.py'),
        # os.path.join('scripts', 'preprocess.py'),
        os.path.join('scripts', 'run_detection_model.py'),
    ]
    for script in scripts:
        run_script(script)


    print("\n✅ Pipeline execution complete.")
