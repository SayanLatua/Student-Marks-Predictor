"""Creates a small SYNTHETIC student dataset (students.csv) for practice.
Replace it with a real dataset (e.g. Kaggle 'Student Performance') later."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 200
hours = rng.uniform(1, 10, n).round(1)
attendance = rng.uniform(50, 100, n).round(0)
prev = rng.uniform(35, 95, n).round(0)
marks = 4 * hours + 0.25 * attendance + 0.35 * prev + rng.normal(0, 4, n)
marks = np.clip(marks, 0, 100).round(1)
pd.DataFrame({"study_hours": hours, "attendance": attendance,
              "previous_score": prev, "final_marks": marks}).to_csv("students.csv", index=False)
print("students.csv created")
