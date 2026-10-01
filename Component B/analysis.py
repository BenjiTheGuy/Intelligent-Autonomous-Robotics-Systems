# Importing libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# File Paths
# Finding the Component B Project Directory
root = os.path.dirname(os.path.abspath(__file__))

# Input CSV
data_file = os.path.join(root, "data", "consensus_data.csv")

results_dir = os.path.join(root, "results")

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Loading Data
print("Loading Consensus Data...")

try:
    data = pd.read_csv(data_file)

except FileNotFoundError:
    print("ERROR: File Not Found.")
    print("Expected Location: ", data_file)

# Displaying Basic Information
print("\nConsensus Dataset:")
print(data)

print("\nDataset Information:")
print(data.info())

# Checking Required Columns
required_cols = [
    "time",
    "robot_1",
    "robot_2",
    "robot_3"
]

for column in required_cols:
    if column not in data.columns:
        print(f"ERROR: Required Column '{column}' is missing from the dataset")

        exit()

# Calculating Group Consensus
# Calculating the average value of all three robots at every recorded time step
data["group_average"] = data[
    ["robot_1", "robot_2", "robot_3"]
].mean(axis=1)

# Calculating Consensus Error
# This value is calculated as the average absolute difference between each robot's value and the group average

data["consensus_error"] = (
    (
        abs(data["robot_1"] - data["group_average"]) +
        abs(data["robot_2"] - data["group_average"]) +
        abs(data["robot_3"] - data["group_average"])
    )
    / 3
)

# Displaying Consensus Results
print("\nFinal Consensus Values:")

print(f"Robot 1: {data['robot_1'].iloc[-1]:.6f}")
print(f"Robot 2: {data['robot_2'].iloc[-1]:.6f}")
print(f"Robot 3: {data['robot_3'].iloc[-1]:.6f}")

print(f"Final Group Average: {data['group_average'].iloc[-1]:.6f}")
print(f"Final Consensus Error: {data['consensus_error'].iloc[-1]:.10f}")

# Consensus Convergence Graph
plt.figure(figsize=(10, 6))

plt.plot(
    data["time"],
    data["robot_1"],
    label="Robot 1"
)

plt.plot(
    data["time"],
    data["robot_2"],
    label="Robot 2"
)

plt.plot(
    data["time"],
    data["robot_3"],
    label="Robot 3"
)

plt.axhline(
    y=50,
    ls="--",
    label="Expected Consensus = 50"
)

plt.xlabel("Simulation Time (s)")
plt.ylabel("Consensus Value")

plt.title("Average Consensus Convergence of Three Robots")

plt.legend()

plt.grid(True)

plt.tight_layout()

# Save the Graph
convergence_graph = os.path.join(results_dir, "consensus_convergence.png")

plt.savefig(convergence_graph, dpi=300)

print(f"Consensus Convergence Graph Saved To: {convergence_graph}")

plt.show()

# Consensus Error Graph
plt.figure(figsize=(10, 6))

plt.plot(
    data["time"],
    data["consensus_error"],
    label="Consensus Error"
)

plt.xlabel("Simulation Time (s)")
plt.ylabel("Consensus Error")

plt.title("Consensus Error Over Time")

plt.legend()

plt.grid(True)

plt.tight_layout()

# Save the Graph
error_graph = os.path.join(results_dir, "consensus_error.png")

plt.savefig(error_graph, dpi=300)

print(f"Consensus Error Graph Saved To: {error_graph}")

plt.show()

# Saving the Analysed Data
analysed_data_file = os.path.join(results_dir, "consensus_analysis.csv")

data.to_csv(analysed_data_file, index=False)

print(f"Analysed Consensus Data Saved To: {analysed_data_file}")

# Final Summary
print("\n")
print("=" * 60)
print("CONSENSUS ANALYSIS COMPLETE")
print("=" * 60)

print(f"Initial Robot 1 Value: {data['robot_1'].iloc[0]:.4f}")
print(f"Initial Robot 2 Value: {data['robot_2'].iloc[0]:.4f}")
print(f"Initial Robot 3 Value: {data['robot_3'].iloc[0]:.4f}")

print(f"Final Group Average: {data['group_average'].iloc[-1]:.6f}")
print(f"Final Consensus Error: {data['consensus_error'].iloc[-1]:.10f}")

print("=" * 60)