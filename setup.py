#!/usr/bin/env python3
"""
Setup script for the Low Latency Trading Platform
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "models",
        "backtest_results",
        "core",
        "strategies",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_environment():
    """Set up environment file"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("📝 Created .env file from .env.example")
            print("   ⚠️  Please edit .env with your configuration")
        else:
            print("⚠️  .env.example not found, please create .env manually")
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if os.path.exists('requirements.txt'):
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
            return False
    else:
        print("⚠️  requirements.txt not found")
        return False
    
    return True

def check_external_dependencies():
    """Check for external dependencies"""
    dependencies = {
        'redis-server': 'Redis server for caching',
        'docker': 'Docker for containerized services (optional)',
        'docker-compose': 'Docker Compose for multi-container setup (optional)'
    }
    
    print("🔍 Checking external dependencies...")
    for cmd, description in dependencies.items():
        if shutil.which(cmd):
            print(f"   ✅ {cmd} found")
        else:
            print(f"   ⚠️  {cmd} not found - {description}")

def create_init_files():
    """Create __init__.py files for proper package structure"""
    init_dirs = ['core', 'strategies', 'tests']
    
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"📄 Created {init_file}")

def setup_git_hooks():
    """Set up git hooks for development"""
    if os.path.exists('.git'):
        hooks_dir = Path('.git/hooks')
        
        # Create pre-commit hook
        pre_commit_hook = hooks_dir / 'pre-commit'
        if not pre_commit_hook.exists():
            hook_content = '''#!/bin/bash
# Run tests before commit
echo "Running tests..."
python -m pytest tests/ --quiet
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Run linting
echo "Running linting..."
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
'''
            pre_commit_hook.write_text(hook_content)
            pre_commit_hook.chmod(0o755)
            print("🔗 Created git pre-commit hook")

def main():
    """Main setup function"""
    print("🚀 Setting up Low Latency Trading Platform")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Create __init__.py files
    create_init_files()
    
    # Set up environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check external dependencies
    check_external_dependencies()
    
    # Set up git hooks
    setup_git_hooks()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Set up external services (Redis, PostgreSQL, etc.)")
    print("3. Configure your Alpaca API keys in .env")
    print("4. Run: python trading_platform.py")
    print("\n📚 Documentation:")
    print("- README.md for detailed instructions")
    print("- API_DOCUMENTATION.md for API reference")
    print("- Check the /docs directory for more guides")

if __name__ == "__main__":
    main()