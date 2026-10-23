# Importing libraries to use in file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Settling Tolerances
POSITION_TOLERANCE = 0.01
HEADING_TOLERANCE = 0.01
###

# Overshoot Proxy represented as the peak absolute heading error
def calculate_overshoot(df):
    heading_error = np.abs(df["Heading Error"].to_numpy())

    return np.max(heading_error)

# Rise-Time Proxy which measures the time for the position error to reduce from 90% to 10% of its initial absolute value.
def calculate_rise_time(df):
    time = df["Time"].to_numpy()
    error = np.abs(df["Position Error"].to_numpy())

    initial_error = error[0]

    if initial_error == 0:
        return np.nan

    reached_90 = np.where(error <= 0.90 * initial_error)[0]
    reached_10 = np.where(error <= 0.10 * initial_error)[0]

    if len(reached_90) == 0 or len(reached_10) == 0:
        return np.nan

    t90 = time[reached_90[0]]
    t10 = time[reached_10[0]]

    return t10 - t90 if t10 >= t90 else np.nan

# Settling Time which measures the first time the error enters its tolerance band and stays there util the recording ends.
def calculate_settling_time(df, column, tolerance):
    time = df["Time"].to_numpy()
    error = np.abs(df[column].to_numpy())

    for i in range(len(time)):
        if np.all(error[i:] <= tolerance):
            return time[i]

    return np.nan

# Steady-State error which measure the mean signed error over the final 10% of recorded samples
def calculate_steady_state_error(df, column):
    error = df[column].to_numpy()
    window_start = int(0.90 * len(error))

    return np.mean(error[window_start:])

# Generating Performance Table
performance = []

for name, df in experiments.items():
    performance.append({
        "Experiment": name,
        "Overshoot Proxy (rad)": calculate_overshoot(df),
        "Rise-Time Proxy (s)": calculate_rise_time(df),
        "Position Settling Time (s)": calculate_settling_time(df, "Position Error", POSITION_TOLERANCE),
        "Heading Settling Time (s)": calculate_settling_time(df, "Heading Error", HEADING_TOLERANCE),
        "Position Steady-State Error (m)": calculate_steady_state_error(df, "Position Error"),
        "Heading Steady-State Error (rad)": calculate_steady_state_error(df, "Heading Error"),
    })

performance_table = pd.DataFrame(performance)

performance_table.to_csv("results/pid_performance_metrics.csv", index=False)

print("\nMetrics saved to results/pid_performance_metrics.csv")

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