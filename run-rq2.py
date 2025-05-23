import subprocess

scripts = [
    "code.RQ2.scripts.best_model_by_feature_group",
    "code.RQ2.scripts.ablation_study",
]

print("Executando RQ2...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ2 finalizada.")
