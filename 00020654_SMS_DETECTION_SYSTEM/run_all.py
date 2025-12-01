# run_all.py — execute notebooks/sms_spam_ensemble.ipynb in-place, then launch Streamlit
import sys, subprocess, os
from pathlib import Path

ROOT = Path(__file__).parent
NB = ROOT / "notebooks" / "sms_spam_ensemble.ipynb"

def sh(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, shell=os.name=="nt")

def main():
    # 1) ensure deps
    sh([sys.executable, "-m", "pip", "install", "-q", "nbclient", "nbconvert", "jupyter", "ipykernel"])
    # 2) execute notebook in-place
    sh(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NB)])
    print("\n✅ Notebook executed successfully.")
    # 3) launch streamlit
    app = ROOT / "app" / "streamlit_app.py"
    if app.exists():
        print("\n🚀 Launching Streamlit…")
        sh([sys.executable, "-m", "streamlit", "run", str(app)])

if __name__ == "__main__":
    main()
