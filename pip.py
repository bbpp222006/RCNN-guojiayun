import subprocess

with open("requirements.txt", "w") as f:
    subprocess.run(["pip", "freeze"], stdout=f)