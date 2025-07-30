import subprocess

# Run a simple Ubuntu command via WSL
result = subprocess.run(['wsl', 'ls', '/home'], capture_output=True, text=True)
print(result.stdout)