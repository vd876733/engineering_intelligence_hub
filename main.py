"""
Entry point for the Engineering Intelligence Hub.
This script launches the Streamlit web interface, which automatically
connects the AST parser, vector store, and Antigravity Agent.
"""

import os
import sys
import subprocess

def main():
    print("🚀 Launching Engineering Intelligence Hub...")
    
    # Path to the streamlit UI application
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ui_script = os.path.join(base_dir, "src", "ui", "app.py")
    
    if not os.path.exists(ui_script):
        print(f"Error: Could not find the UI script at {ui_script}")
        sys.exit(1)
        
    # Launch Streamlit using the current python executable (the venv)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_script])
    except KeyboardInterrupt:
        print("\nShutting down Engineering Intelligence Hub. Goodbye!")

if __name__ == "__main__":
    main()
