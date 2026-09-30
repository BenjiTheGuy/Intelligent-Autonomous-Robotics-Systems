from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Component A - PID Controller Analysis

# Loading the Experiment Data
baseline = pd.read_csv("results/pid_baseline_kp2.csv")
exp_1 = pd.read_csv("results/pid_kp3.csv")
exp_2 = pd.read_csv("results/pid_kp3_ki005.csv")
exp_3 = pd.read_csv("results/pid_kp3_ki005_kd04.csv")

experiments = {
    "Baseline (Kp=2.0, Ki=0, Kd=0.2)": baseline,
    "Experiment 1 (Kp=3.0, Ki=0, Kd=0.2)": exp_1,
    "Experiment 2 (Kp=3.0, Ki=0.05, Kd=0.2)": exp_2,
    "Experiment 3 (Kp=3.0, Ki=0.05, Kd=0.4)": exp_3,
}

# Performance Calculation on different criteria
def calculate_heading_overshoot(df):
    heading_error = df["Heading Error"].to_numpy()

    for i in range(1, len(heading_error)):
        if (heading_error[i - 1] * heading_error[i]) < 0:
            positive_values = heading_error[i:]

            positive_values = positive_values[positive_values > 0]

            if len(positive_values) > 0:
                return np.max(positive_values)

            return 0

    return 0

def calculate_reduction_time(df):
    time = df["Time"].to_numpy()

    error = np.abs(df["Position Error"].to_numpy())

    initial_error = error[0]

    target_90 = initial_error * 0.90
    target_10 = initial_error * 0.10

    reached_90 = np.where(error <= target_90)[0]
    reached_10 = np.where(error <= target_10)[0]

    if len(reached_90) == 0:
        return np.nan

    if len(reached_10) == 0:
        return np.nan

    return(time[reached_10[0]] - time[reached_90[0]])

def calculate_settling_time(df):
    time = df["Time"].to_numpy()

    error = np.abs(df["Position Error"].to_numpy())

    tolerance = error[0] * 0.02

    outside_band = np.where(error > tolerance)[0]

    if len(outside_band) == 0:
        return time[0]

    last_outside = outside_band[-1]

    if last_outside == len(time) - 1:
        return np.nan

    return time[last_outside + 1]

def calculate_metrics(df):
    initial_error = df["Position Error"].iloc[0]
    final_error = df["Position Error"].iloc[-1]

    error_reduction = ((abs(initial_error) - abs(final_error)) / (abs(initial_error)) * 100)

    return {
        "Initial Error (m)": initial_error,
        "Final Error (m)": final_error,
        "Error Reduction (%)": error_reduction,
        "Heading Overshoot (rad)": calculate_heading_overshoot(df),
        "90%-10% Error Time (s)": calculate_reduction_time(df),
        "Settling Time (s)": calculate_settling_time(df),
        "Final Heading Error (rad)": df["Heading Error"].iloc[-1],
        "Maximum PID Correction": df["PID Correction"].abs().max(),
    }

# Generating Performance Table
performance = []

for name, df in experiments.items():
    results = calculate_metrics(df)

    results["Experiment"] = name

    performance.append(results)

performance_table = pd.DataFrame(performance)

print("\n")
print("=" * 100)
print("PID PERFORMANCE ANALYSIS")
print("=" * 100)

print(performance_table.to_string(index=False))

# Position Error Plot
plt.figure(figsize=(10, 5.5))

for name, df in experiments.items():
    plt.plot(
        df["Time"],
        df["Position Error"],
        label=name,
        lw=1.5,
    )

plt.axhline(
    0,
    ls="--",
    lw=1
)

plt.title("PID Position Error Response")
plt.xlabel("Time (s)")
plt.ylabel("Position Error (m)")

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()

# Heading Error Plot
plt.figure(figsize=(10, 5.5))

for name, df in experiments.items():
    plt.plot(
        df["Time"],
        df["Heading Error"],
        label=name,
        lw=1.5
    )

plt.axhline(
    0,
    ls="--",
    lw=1
)

plt.title("PID Heading Error Response")
plt.xlabel("Time (s)")
plt.ylabel("Heading Error (rad)")

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()

# PID Correction Plot
plt.figure(figsize=(10, 5.5))

for name, df in experiments.items():
    plt.plot(
        df["Time"],
        df["PID Correction"],
        label=name,
        lw=1.5
    )

plt.axhline(
    0,
    ls="--",
    lw=1
)

plt.title("PID Correction Response")
plt.xlabel("Time (s)")
plt.ylabel("PID Correction")

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()