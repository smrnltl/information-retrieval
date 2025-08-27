#!/usr/bin/env python3
"""
Academic Search Engine System Launcher
This script helps you run the complete search engine system
"""

import subprocess
import sys
import time
import os
import threading
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'jinja2', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with: pip install -r requirements.txt")
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
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nRun your crawler first to generate the data files.")
        return False
    
    return True

def build_search_index():
    """Build the search index"""
    print("🔨 Building search index...")
    try:
        from search_engine import rebuild_index
        rebuild_index()
        print("✅ Search index built successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to build search index: {e}")
        return False

def run_api_server():
    """Run the API server in a separate process"""
    print("🚀 Starting API server...")
    try:
        # Run API server
        process = subprocess.Popen([
            sys.executable, "api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ API server started successfully on http://localhost:8000")
            print("   - API Documentation: http://localhost:8000/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ API server failed to start:")
            print(f"   STDOUT: {stdout.decode()}")
            print(f"   STDERR: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def run_web_server():
    """Run the web server in a separate process"""
    print("🌐 Starting web server...")
    try:
        # Run web server
        process = subprocess.Popen([
            sys.executable, "web_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ Web server started successfully on http://localhost:8080")
            print("   - Search Interface: http://localhost:8080")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Web server failed to start:")
            print(f"   STDOUT: {stdout.decode()}")
            print(f"   STDERR: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Failed to start web server: {e}")
        return None

def test_search_functionality():
    """Test basic search functionality"""
    print("🧪 Testing search functionality...")
    try:
        from search_engine import SearchEngine
        from query_processor import QueryParser
        
        # Initialize components
        engine = SearchEngine()
        parser = QueryParser()
        
        # Test search
        results = engine.search("education", limit=5)
        
        if results:
            print(f"✅ Search test passed! Found {len(results)} results")
            print(f"   Sample result: {results[0].publication.title[:50]}...")
            return True
        else:
            print("⚠️  Search test returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def display_system_info():
    """Display system information"""
    try:
        from search_engine import SearchEngine
        engine = SearchEngine()
        stats = engine.get_stats()
        
        print("\n📊 System Information:")
        print(f"   Total Publications: {stats['total_publications']}")
        print(f"   Indexed Terms: {stats['total_terms']}")
        print(f"   Index Size: {stats['index_size']} bytes")
        
    except Exception as e:
        print(f"⚠️  Could not retrieve system stats: {e}")

def main():
    """Main function to run the complete system"""
    print("=" * 60)
    print("🎓 Academic Search Engine System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check data files
    if not check_data_files():
        return 1
    
    # Build search index
    if not build_search_index():
        return 1
    
    # Test search functionality
    if not test_search_functionality():
        return 1
    
    # Display system info
    display_system_info()
    
    print("\n🚀 Starting servers...")
    
    # Start API server
    api_process = run_api_server()
    if not api_process:
        return 1
    
    # Start web server
    web_process = run_web_server()
    if not web_process:
        api_process.terminate()
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Academic Search Engine is now running!")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:8080")
    print("🔌 API Server: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    print("\n💡 Usage Tips:")
    print("   - Use the web interface for searching publications")
    print("   - Try advanced search with filters and boolean operators")
    print("   - Check the API docs for programmatic access")
    print("   - Press Ctrl+C to stop the system")
    print("\n🔍 Example searches to try:")
    print("   - 'higher education'")
    print("   - 'economics AND finance'")
    print("   - 'author:smith year:2020'")
    print("   - '\"machine learning\" OR \"artificial intelligence\"'")
    
    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
            
            if web_process.poll() is not None:
                print("❌ Web server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Terminate processes gracefully
        if api_process and api_process.poll() is None:
            api_process.terminate()
            print("   API server stopped")
        
        if web_process and web_process.poll() is None:
            web_process.terminate()
            print("   Web server stopped")
        
        print("✅ System shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())