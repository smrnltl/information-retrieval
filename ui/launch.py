"""
Unified Academic Tools Launcher
Combined Academic Search Engine + Document Classifier
"""
import subprocess
import sys
import webbrowser
import time
import os
from pathlib import Path

def launch_unified_app():
    """Launch the unified academic tools application"""
    print("Starting Unified Academic Tools...")
    print("Academic Search Engine + Document Classifier")
    print("Loading engines and models...")
    
    # Change to UI directory
    ui_dir = Path(__file__).parent
    os.chdir(ui_dir)

    port = os.environ.get("PORT", "8000")
    try:
        # Launch FastAPI with uvicorn
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", port,
            #"--reload"
        ])
        
        # Wait a moment for server to start
        print("Starting server...")
        time.sleep(5)
        
        # Open browser
        url = "http://localhost:8000"
        print(f"Opening browser at {url}")
        webbrowser.open(url)
        
        print("\nUnified Academic Tools is running!")
        print("Search academic publications")
        print("Classify documents (Politics, Business, Health)")
        print("Combined interface with modern design")
        print("Press Ctrl+C to stop the server")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        print("Server stopped")
    except Exception as e:
        print(f"Error launching application: {e}")
        print("Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    launch_unified_app()
