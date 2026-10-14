
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/content/drive/MyDrive/crisis-fusion"
OUT = os.path.join(BASE, "figures", "report")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------

results = pd.DataFrame([
    ["Task 1a - Informativeness", 87.03, 86.83, 91.13, 90.98],
    ["Task 1b - Disaster Type", 86.57, 86.30, 90.29, 90.23],
    ["Task 2 - Humanitarian Information", 82.30, 82.25, 86.28, 85.76],
    ["Task 3 - Damage Severity", 64.27, 52.55, 63.14, 49.41],
], columns=[
    "Task",
    "Best Unimodal Accuracy",
    "Best Unimodal F1",
    "Score Fusion Accuracy",
    "Score Fusion F1"
])

# ---------------------------------------------------------
# FIGURE 1: ACCURACY COMPARISON
# ---------------------------------------------------------

x = range(len(results))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width/2 for i in x],
    results["Best Unimodal Accuracy"],
    width,
    label="Best Unimodal"
)

plt.bar(
    [i + width/2 for i in x],
    results["Score Fusion Accuracy"],
    width,
    label="Score Fusion"
)

plt.xticks(list(x), ["Task 1a", "Task 1b", "Task 2", "Task 3"])
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: Unimodal vs Score Fusion")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "accuracy_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------
# FIGURE 2: F1 COMPARISON
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width/2 for i in x],
    results["Best Unimodal F1"],
    width,
    label="Best Unimodal"
)

plt.bar(
    [i + width/2 for i in x],
    results["Score Fusion F1"],
    width,
    label="Score Fusion"
)

plt.xticks(list(x), ["Task 1a", "Task 1b", "Task 2", "Task 3"])
plt.ylabel("Weighted F1 (%)")
plt.title("Weighted F1 Comparison: Unimodal vs Score Fusion")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "f1_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------
# FIGURE 3: SCORE-FUSION IMPROVEMENT
# ---------------------------------------------------------

improvements = [
    91.13 - 87.03,
    90.29 - 86.57,
    86.28 - 82.30,
    63.14 - 64.27
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    ["Task 1a", "Task 1b", "Task 2", "Task 3"],
    improvements
)

plt.axhline(0, linewidth=1)
plt.ylabel("Accuracy Improvement (percentage points)")
plt.title("Score Fusion Accuracy Improvement")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "fusion_improvement.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------
# TASK 3 IMPROVEMENT EXPERIMENTS
# ---------------------------------------------------------

task3 = pd.read_csv(
    os.path.join(BASE, "task3_improvement_results.csv")
)

task3["Experiment"] = [
    "Baseline",
    "Class Weights",
    "Weights + Augmentation",
    "Weights + Augmentation + Oversampling"
]

plt.figure(figsize=(11, 6))

plt.bar(
    task3["Experiment"],
    task3["macro_f1"] * 100
)

plt.ylabel("Macro F1 (%)")
plt.title("Task 3 Improvement Experiments")
plt.ylim(0, 70)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "task3_improvements.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------
# SAVE CLEAN SUMMARY
# ---------------------------------------------------------

results.to_csv(
    os.path.join(OUT, "final_results_clean.csv"),
    index=False
)

print("=" * 70)
print("REPORT FIGURES GENERATED")
print("=" * 70)

for f in os.listdir(OUT):
    print(f)

print("\nSaved to:")
print(OUT)
