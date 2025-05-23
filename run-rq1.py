import subprocess

scripts = [
    "code.RQ1.scripts.gradient_boosting",
    "code.RQ1.scripts.logistic_regression",
    "code.RQ1.scripts.random_forest",
    "code.RQ1.scripts.xgboost_rscv",
    "code.RQ1.scripts.svm",
    "code.RQ1.scripts.svm_tfidf"
]

print("Executando RQ1...")

for script in scripts:
    print(f"\nExecutando {script}...")
    subprocess.run(["python", "-m", script], check=True)

print("\nRQ1 finalizada.")
