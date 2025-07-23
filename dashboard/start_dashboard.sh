#!/bin/bash

# Advanced Trading Dashboard - Quick Start Script

set -e

echo "🚀 Starting Advanced Trading Dashboard"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if ! command -v redis-cli &> /dev/null; then
    echo "⚠️  Redis CLI not found. Please install Redis server."
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   macOS: brew install redis"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:alpine"
else
    if ! redis-cli ping &> /dev/null; then
        echo "⚠️  Redis server is not running. Starting Redis..."
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes
            sleep 2
        else
            echo "❌ Redis server not found. Please start Redis manually or use Docker."
            exit 1
        fi
    else
        echo "✅ Redis is running"
    fi
fi

# Start the dashboard
echo ""
echo "🎯 Starting Trading Dashboard..."
echo "   Access at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

python run_dashboard.py --debug