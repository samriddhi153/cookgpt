#!/usr/bin/env python3
"""
CookGPT Launcher - Starts the full stack application
Usage: python run.py
"""

import subprocess
import sys
import time
import webbrowser
import requests
from threading import Thread
import signal
import os

# Configuration
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
STREAMLIT_PORT = 8501
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

def print_banner():
    print("\n" + "="*60)
    print("🍳 CookGPT - AI Cooking Assistant")
    print("="*60 + "\n")

def check_dependencies():
    """Verify required packages are installed"""
    print("📦 Checking dependencies...")
    required = ["fastapi", "uvicorn", "streamlit", "langchain", "langgraph", "faiss", "sentence_transformers", "pandas", "requests", "dotenv", "groq"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing packages: {missing}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)

    print("✅ All dependencies installed\n")

def check_env_vars():
    """Verify environment variables"""
    print("🔐 Checking environment variables...")
    from dotenv import load_dotenv
    load_dotenv()

    required_vars = ["GROQ_API_KEY", "USDA_API_KEY"]
    optional_vars = ["HUGGINGFACE_API_KEY"]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"❌ Missing required env vars: {missing}")
        print("   Check your .env file")
        sys.exit(1)

    print(f"✅ Required vars present ({', '.join(required_vars)})")
    print(f"✅ Optional vars: {', '.join([v for v in optional_vars if os.getenv(v)])}\n")

def start_backend():
    """Start FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    print(f"   URL: {BACKEND_URL}")

    # Run uvicorn as subprocess
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)]

    # Start without blocking
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Optional: Print backend logs in a separate thread
    def log_thread():
        for line in process.stdout:
            print(f"[Backend] {line.strip()}")

    Thread(target=log_thread, daemon=True).start()

    # Wait for backend to be ready
    print("   Waiting for backend to initialize (RAG index build may take 1-2 min)...")
    max_attempts = 60  # 2 minutes
    for i in range(max_attempts):
        try:
            resp = requests.get(f"{BACKEND_URL}/docs", timeout=3)
            if resp.status_code == 200:
                print("✅ Backend is running!\n")
                return process
        except:
            time.sleep(2)
            print(f"   Attempt {i+1}/{max_attempts}...", end="\r")

    print("❌ Backend failed to start within timeout")
    process.kill()
    sys.exit(1)

def start_streamlit():
    """Start Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    print(f"   URL: http://localhost:{STREAMLIT_PORT}")

    cmd = [
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "localhost"
    ]

    process = subprocess.Popen(cmd)
    print("✅ Streamlit UI launched\n")

    # Auto-open browser
    time.sleep(2)
    url = f"http://localhost:{STREAMLIT_PORT}"
    print(f"🌐 Opening browser: {url}")
    webbrowser.open(url)

    return process

def main():
    print_banner()

    # Pre-flight checks
    check_dependencies()
    check_env_vars()

    print("="*60 + "\n")

    # Start Backend
    backend_proc = start_backend()

    # Start Streamlit
    streamlit_proc = start_streamlit()

    print("="*60)
    print("✨ CookGPT is now running!")
    print("="*60)
    print(f"\n   Backend API : {BACKEND_URL}")
    print(f"   API Docs    : {BACKEND_URL}/docs")
    print(f"   Streamlit UI: http://localhost:{STREAMLIT_PORT}")
    print("\n   Press Ctrl+C to stop all services\n")

    # Wait for interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        backend_proc.terminate()
        streamlit_proc.terminate()
        backend_proc.wait()
        streamlit_proc.wait()
        print("✅ All services stopped\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
