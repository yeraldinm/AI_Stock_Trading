#!/usr/bin/env python3
"""
Dashboard Startup Script
"""
import os
import sys
import argparse
from app import app, socketio

def main():
    parser = argparse.ArgumentParser(description='Run Trading Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    
    args = parser.parse_args()
    
    if args.production:
        # Production settings
        app.config['DEBUG'] = False
        print(f"Starting Trading Dashboard in PRODUCTION mode on {args.host}:{args.port}")
        socketio.run(app, 
                    host=args.host, 
                    port=args.port, 
                    debug=False,
                    use_reloader=False)
    else:
        # Development settings
        app.config['DEBUG'] = args.debug
        print(f"Starting Trading Dashboard in DEVELOPMENT mode on {args.host}:{args.port}")
        socketio.run(app, 
                    host=args.host, 
                    port=args.port, 
                    debug=args.debug,
                    use_reloader=args.debug)

if __name__ == '__main__':
    main()