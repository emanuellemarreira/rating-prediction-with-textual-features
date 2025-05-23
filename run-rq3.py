import subprocess

scripts = [
    "code.RQ3.scripts.best_model_RFE"
]

print("Executando RQ3...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ3 finalizada.")
