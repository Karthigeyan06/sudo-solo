#!/usr/bin/env python3
"""
SUDO-SOLO Automation System Launcher
Helps start all components of the automation pipeline
"""

import os
import sys
import subprocess
import time
import platform
import argparse
from pathlib import Path

class AutomationLauncher:
    """Launcher for all automation components"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.processes = {}
        self.running = False
        
    def print_banner(self):
        """Print startup banner"""
        banner = """
╔════════════════════════════════════════════════════════════════╗
║                  SUDO-SOLO AUTOMATION SYSTEM                   ║
║           Autonomous Solar Panel Maintenance Pipeline          ║
╚════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_requirements(self):
        """Check if all requirements are installed"""
        print("[1/4] Checking requirements...")
        
        try:
            import flask
            import tensorflow
            import transformers
            import torch
            print("  ✓ All requirements found")
            return True
        except ImportError as e:
            print(f"  ✗ Missing requirement: {e}")
            print("\n  Install with: pip install -r requirements_automation.txt")
            return False
    
    def check_config(self):
        """Check configuration"""
        print("[2/4] Checking configuration...")
        
        try:
            from pipeline_config import validate_config, CONTROL_CENTER_IP
            
            if not validate_config():
                print("  ✗ Configuration validation failed")
                return False
            
            print(f"  ✓ Configuration valid")
            print(f"    - Control Center IP: {CONTROL_CENTER_IP}")
            return True
        except Exception as e:
            print(f"  ✗ Configuration error: {e}")
            return False
    
    def check_ports(self):
        """Check if required ports are available"""
        print("[3/4] Checking ports...")
        
        import socket
        
        ports = {
            5000: "Control Center / Detection",
            5001: "Device Control",
            80: "Camera Server"
        }
        
        issues = []
        for port, service in ports.items():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                s.close()
                
                if result == 0:
                    print(f"  ⚠ Port {port} ({service}): In use - may conflict")
                    issues.append(port)
                else:
                    print(f"  ✓ Port {port} ({service}): Available")
            except:
                pass
        
        if issues:
            print(f"\n  Note: Ports {issues} appear to be in use")
            return input("  Continue anyway? (y/n): ").lower() == 'y'
        
        return True
    
    def check_model_file(self):
        """Check if model file exists"""
        print("[4/4] Checking model file...")
        
        model_path = os.path.join(self.project_root, 'solar_fault_model.h5')
        
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ Model found: {size:.1f} MB")
            return True
        else:
            print(f"  ✗ Model file not found: {model_path}")
            return False
    
    def start_control_center(self):
        """Start control center"""
        if self.running and 'control_center' in self.processes:
            print("  ✓ Control Center already running")
            return True
        
        print("\nStarting Control Center...")
        try:
            script = os.path.join(self.project_root, 'control_center.py')
            if sys.platform == 'win32':
                self.processes['control_center'] = subprocess.Popen(
                    [sys.executable, script],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.processes['control_center'] = subprocess.Popen(
                    [sys.executable, script],
                    cwd=self.project_root
                )
            
            time.sleep(2)
            print("  ✓ Control Center started (PID: {})".format(
                self.processes['control_center'].pid
            ))
            return True
        except Exception as e:
            print(f"  ✗ Failed to start Control Center: {e}")
            return False
    
    def start_detection_server(self):
        """Start detection server"""
        if self.running and 'detection' in self.processes:
            print("  ✓ Detection Server already running")
            return True
        
        print("Starting Detection Server...")
        try:
            script = os.path.join(self.project_root, 'detect.py')
            if sys.platform == 'win32':
                self.processes['detection'] = subprocess.Popen(
                    [sys.executable, script, '--server'],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.processes['detection'] = subprocess.Popen(
                    [sys.executable, script, '--server'],
                    cwd=self.project_root
                )
            
            time.sleep(2)
            print("  ✓ Detection Server started (PID: {})".format(
                self.processes['detection'].pid
            ))
            return True
        except Exception as e:
            print(f"  ✗ Failed to start Detection Server: {e}")
            return False
    
    def start_pipeline(self):
        """Start automation pipeline"""
        if self.running and 'pipeline' in self.processes:
            print("  ✓ Pipeline already running")
            return True
        
        print("Starting Automation Pipeline...")
        try:
            script = os.path.join(self.project_root, 'automation_pipeline.py')
            if sys.platform == 'win32':
                self.processes['pipeline'] = subprocess.Popen(
                    [sys.executable, script],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.processes['pipeline'] = subprocess.Popen(
                    [sys.executable, script],
                    cwd=self.project_root
                )
            
            time.sleep(2)
            print("  ✓ Automation Pipeline started (PID: {})".format(
                self.processes['pipeline'].pid
            ))
            return True
        except Exception as e:
            print(f"  ✗ Failed to start Automation Pipeline: {e}")
            return False
    
    def launch_all(self):
        """Launch all components"""
        print("\n" + "="*60)
        print("STARTING ALL COMPONENTS")
        print("="*60 + "\n")
        
        success = True
        success = self.start_control_center() and success
        print()
        success = self.start_detection_server() and success
        print()
        success = self.start_pipeline() and success
        
        if success:
            self.running = True
            print("\n" + "="*60)
            print("ALL COMPONENTS STARTED SUCCESSFULLY")
            print("="*60)
            print("\nDashboard URL: http://192.168.1.100:5000")
            print("\nPress Ctrl+C to stop all services.")
            print("="*60 + "\n")
            return True
        else:
            print("\n✗ Some components failed to start")
            return False
    
    def monitor(self):
        """Monitor running processes"""
        try:
            while True:
                time.sleep(1)
                
                # Check if any process died
                for name, proc in self.processes.items():
                    if proc and proc.poll() is not None:
                        print(f"\n✗ {name} process died (exit code: {proc.returncode})")
                        print("Run launcher again to restart\n")
                        return
        
        except KeyboardInterrupt:
            self.stop_all()
    
    def stop_all(self):
        """Stop all processes"""
        print("\n\nStopping all services...")
        
        for name, proc in self.processes.items():
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    print(f"  ✓ {name} stopped")
                except:
                    proc.kill()
                    print(f"  ✓ {name} killed")
        
        print("\nAll services stopped.")
    
    def run(self, mode='all'):
        """Run launcher"""
        self.print_banner()
        
        # Pre-launch checks
        if not self.check_requirements():
            return False
        
        if not self.check_config():
            return False
        
        if not self.check_ports():
            return False
        
        if not self.check_model_file():
            print("\nWarning: Model file missing. Get solar_fault_model.h5")
            return False
        
        print("\n" + "="*60)
        
        # Launch services
        if mode == 'all' or mode == 'full':
            if not self.launch_all():
                return False
        elif mode == 'control':
            self.start_control_center()
        elif mode == 'detection':
            self.start_detection_server()
        elif mode == 'pipeline':
            self.start_pipeline()
        
        # Monitor running processes
        if self.running:
            self.monitor()
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SUDO-SOLO Automation System Launcher'
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'full', 'control', 'detection', 'pipeline'],
        default='all',
        help='Which components to start (default: all)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip pre-launch checks'
    )
    
    args = parser.parse_args()
    
    launcher = AutomationLauncher()
    
    try:
        if not launcher.run(mode=args.mode):
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
        launcher.stop_all()
        sys.exit(0)

if __name__ == '__main__':
    main()
