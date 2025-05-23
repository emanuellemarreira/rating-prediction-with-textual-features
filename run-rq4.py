import subprocess

scripts = [
    "code.RQ4.scripts.best_model_by_product_category"
]

print("Executando RQ4...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ4 finalizada.")
