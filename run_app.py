import sys
import subprocess

if __name__ == "__main__":
    print("Launching AI Delivery Route Optimizer Streamlit Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)
