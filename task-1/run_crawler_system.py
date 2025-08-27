#!/usr/bin/env python3
"""
Complete Crawler System Launcher
Launches all crawler components including scheduler, dashboard, and monitoring
"""

import subprocess
import sys
import time
import os
import threading
from pathlib import Path
import signal
import json

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'jinja2', 'pydantic', 'schedule', 
        'requests', 'selenium', 'undetected_chromedriver'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with: pip install -r requirements_crawler.txt")
        return False
    
    return True

def check_data_files():
    """Check if data files exist"""
    required_files = ['publications.db', 'publications.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nRun your crawler first to generate the data files.")
        print("You can still use the system, but search functionality will be limited.")
    
    return len(missing_files) == 0

def start_api_server():
    """Start the main API server"""
    print("🚀 Starting Main API Server (port 8000)...")
    process = subprocess.Popen([
        sys.executable, "api_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(3)
    if process.poll() is None:
        print("✅ Main API Server started successfully")
        return process
    else:
        stdout, stderr = process.communicate()
        print(f"❌ Main API Server failed to start:")
        print(f"   STDOUT: {stdout.decode()}")
        print(f"   STDERR: {stderr.decode()}")
        return None

def start_web_server():
    """Start the web portal server"""
    print("🌐 Starting Web Portal (port 8080)...")
    process = subprocess.Popen([
        sys.executable, "web_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(3)
    if process.poll() is None:
        print("✅ Web Portal started successfully")
        return process
    else:
        stdout, stderr = process.communicate()
        print(f"❌ Web Portal failed to start:")
        print(f"   STDOUT: {stdout.decode()}")
        print(f"   STDERR: {stderr.decode()}")
        return None

def start_crawler_dashboard():
    """Start the crawler management dashboard"""
    print("🕷️  Starting Crawler Dashboard (port 8081)...")
    process = subprocess.Popen([
        sys.executable, "crawler_dashboard.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(3)
    if process.poll() is None:
        print("✅ Crawler Dashboard started successfully")
        return process
    else:
        stdout, stderr = process.communicate()
        print(f"❌ Crawler Dashboard failed to start:")
        print(f"   STDOUT: {stdout.decode()}")
        print(f"   STDERR: {stderr.decode()}")
        return None

def create_default_crawler_jobs():
    """Create default crawler jobs"""
    print("⚙️  Creating default crawler jobs...")
    try:
        import requests
        response = requests.post(
            "http://localhost:8000/crawler/jobs/create-defaults",
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            job_count = len(result.get('created_jobs', []))
            print(f"✅ Created {job_count} default crawler jobs")
            return True
        else:
            print(f"⚠️  Failed to create default jobs: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Could not create default jobs: {e}")
        return False

def show_system_status():
    """Show system status and URLs"""
    try:
        import requests
        
        # Check API server
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            api_status = "✅ Running" if response.status_code == 200 else "❌ Error"
        except:
            api_status = "❌ Not responding"
        
        # Check crawler scheduler
        try:
            response = requests.get("http://localhost:8000/crawler/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                scheduler_status = "✅ Running" if data.get('running') else "❌ Stopped"
                active_jobs = data.get('active_crawls', 0)
                total_jobs = data.get('total_jobs', 0)
            else:
                scheduler_status = "❌ Error"
                active_jobs = 0
                total_jobs = 0
        except:
            scheduler_status = "❌ Not responding"
            active_jobs = 0
            total_jobs = 0
        
        # Check search engine
        try:
            response = requests.get("http://localhost:8000/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                publications = data.get('total_publications', 0)
                search_status = f"✅ {publications} publications indexed"
            else:
                search_status = "❌ Error"
        except:
            search_status = "❌ Not responding"
        
        print("\n" + "=" * 70)
        print("🎓 ACADEMIC SEARCH ENGINE & CRAWLER SYSTEM")
        print("=" * 70)
        print(f"📊 API Server:          {api_status}")
        print(f"🔍 Search Engine:       {search_status}")
        print(f"🕷️  Crawler Scheduler:   {scheduler_status}")
        print(f"📋 Crawler Jobs:        {total_jobs} total, {active_jobs} active")
        print("=" * 70)
        print("🌐 Access Points:")
        print("   Search Portal:       http://localhost:8080")
        print("   Crawler Dashboard:   http://localhost:8081/crawler")
        print("   API Documentation:   http://localhost:8000/docs")
        print("   API Health Check:    http://localhost:8000/health")
        print("=" * 70)
        print("📖 Quick Start Guide:")
        print("   1. Search publications at http://localhost:8080")
        print("   2. Manage crawlers at http://localhost:8081/crawler")
        print("   3. View API docs at http://localhost:8000/docs")
        print("   4. Monitor system health via API endpoints")
        print("=" * 70)
        
    except Exception as e:
        print(f"⚠️  Could not retrieve full system status: {e}")

def main():
    """Main function to run the complete system"""
    print("🚀 STARTING COMPLETE ACADEMIC SEARCH & CRAWLER SYSTEM")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check data files (warning only)
    data_exists = check_data_files()
    
    # Start services
    processes = {}
    
    # Start main API server (includes scheduler)
    api_process = start_api_server()
    if api_process:
        processes['api'] = api_process
    else:
        print("❌ Cannot continue without API server")
        return 1
    
    # Start web portal
    web_process = start_web_server()
    if web_process:
        processes['web'] = web_process
    
    # Start crawler dashboard
    dashboard_process = start_crawler_dashboard()
    if dashboard_process:
        processes['dashboard'] = dashboard_process
    
    # Wait for services to fully start
    print("⏳ Waiting for services to initialize...")
    time.sleep(10)
    
    # Create default crawler jobs if no data exists
    if not data_exists:
        create_default_crawler_jobs()
    
    # Show system status
    show_system_status()
    
    print("\n💡 Tips:")
    print("   - The crawler scheduler runs automatically in the background")
    print("   - Use the dashboard to create and manage crawler jobs")
    print("   - Check logs in the console for real-time activity")
    print("   - Press Ctrl+C to stop all services")
    print()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down all services...")
        for name, process in processes.items():
            try:
                process.terminate()
                print(f"   Stopped {name} service")
            except:
                pass
        print("✅ All services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep main process alive
    try:
        while True:
            time.sleep(1)
            
            # Check if any process died
            dead_processes = []
            for name, process in processes.items():
                if process.poll() is not None:
                    dead_processes.append(name)
            
            if dead_processes:
                print(f"\n⚠️  Service(s) stopped unexpectedly: {', '.join(dead_processes)}")
                break
                
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    for name, process in processes.items():
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass
    
    print("✅ System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())