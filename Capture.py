import subprocess
import os

def take_screenshot():
    file_path = 'files/screenshot/screenshot.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        subprocess.run(['spectacle', '-b', '-n', '-o', file_path], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(['grim', file_path], check=True)