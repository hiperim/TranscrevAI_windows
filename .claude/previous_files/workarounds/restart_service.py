"""
Workaround Script - Restart TranscrevAI Service
CORREÇÃO 1.3: Force reload de modelos reiniciando processo

Uso:
  python restart_service.py
"""

import sys
import os
import psutil
import time
import signal

print('=' * 70)
print('RESTART SERVICE - Force Model Reload')
print('=' * 70)
print()

# Find and kill existing main.py processes
killed = 0
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = proc.info.get('cmdline', [])
        if cmdline and 'main.py' in ' '.join(cmdline):
            print(f'Killing process PID {proc.info["pid"]}: {" ".join(cmdline)}')
            proc.terminate()
            killed += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if killed > 0:
    print(f'\n{killed} process(es) terminated. Waiting 2 seconds...')
    time.sleep(2)
else:
    print('No existing main.py processes found.')

print('\nService restart complete.')
print('Start server manually with: python main.py')
print('=' * 70)