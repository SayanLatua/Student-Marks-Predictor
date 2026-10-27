"""Project 2: Student Marks Predictor (Regression)

Part A: simple linear regression (study hours -> marks)
Part B: multiple linear regression (hours + attendance + previous score)
Plots the fitted line and a residual plot.

Run:  python make_dataset.py   (once)
      python student_marks.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("students.csv")
print(df.corr().round(2), "\n")  # which features matter most?

# ---------- Part A: simple regression ----------
X1, y = df[["study_hours"]], df["final_marks"]
X1_tr, X1_te, y_tr, y_te = train_test_split(X1, y, test_size=0.2, random_state=1)
simple = LinearRegression().fit(X1_tr, y_tr)
p1 = simple.predict(X1_te)
print(f"Simple    -> slope={simple.coef_[0]:.2f}, intercept={simple.intercept_:.2f}, "
      f"R2={r2_score(y_te, p1):.3f}, MAE={mean_absolute_error(y_te, p1):.2f}")

# ---------- Part B: multiple regression ----------
features = ["study_hours", "attendance", "previous_score"]
X_tr, X_te, y_tr2, y_te2 = train_test_split(df[features], y, test_size=0.2, random_state=1)
multi = LinearRegression().fit(X_tr, y_tr2)
p2 = multi.predict(X_te)
print(f"Multiple  -> R2={r2_score(y_te2, p2):.3f}, MAE={mean_absolute_error(y_te2, p2):.2f}")
for f, c in zip(features, multi.coef_):
    print(f"   {f}: {c:.2f}")

# ---------- Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(df.study_hours, df.final_marks, alpha=0.5)
xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 50).reshape(-1, 1)
ax[0].plot(xs, simple.predict(pd.DataFrame(xs, columns=["study_hours"])), "r")
ax[0].set(xlabel="Study hours", ylabel="Final marks", title="Simple regression fit")

residuals = y_te2 - p2
ax[1].scatter(p2, residuals, alpha=0.6)
ax[1].axhline(0, color="r", linestyle="--")
ax[1].set(xlabel="Predicted marks", ylabel="Residual", title="Residual plot (multiple)")
plt.tight_layout()
plt.savefig("student_marks_plots.png", dpi=120)

# ---------- Try a prediction ----------
new = pd.DataFrame([[6, 85, 70]], columns=features)
print(f"\nPredicted marks for 6 hrs, 85% attendance, previous 70: {multi.predict(new)[0]:.1f}")
