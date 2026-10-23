INTELLIGENT AUTONOMOUS ROBOTIC SYSTEMS
======================================

PROJECT COMPONENTS
------------------
Component A:
Perception, Motion Planning and Control for an Autonomous Mobile Robot

Component B:
Multi-Agent Robotic Swarm System with Learning-Based Decision-Making

SOFTWARE
------------------
- Webots: R2025a
- Python: 3.14
- IDE: PyCharm
- Operating System: Windows 11

COMPONENT A
------------------
The simulation uses LiDAR and Camera-based Perception, Dijkstra's and A* path planning, and PID-based waypoint tracking.

To run the project:

1. Install the Python Dependencies:
    python -m pip install numpy matplotlib opencv-python pandas tabulate

2. Run the planning script from the Component A controller directory:
    cd "Component A\webots\controllers\component_a_controller"
    python planning.py

    This generates the occupancy-grid visualization and A* waypoints.

3. Open Component_A.wbt from Component A/webots/worlds/ in Webots and run the simulation with the configured component_a_controller. The controller records trajectory and PID response data in the Component A data folder.

4. After the simulation, run the trajectory and performance evaluation scripts from the controller directory:
    python plot_trajectory.py
    python performance_evaluation.py

    These will generate the planned vs actual trajectory plot and evaluate navigation performance.

5. To analyze the PID experiments, run analysis.py from the Component A project directory:
    cd "/../../.."
    python analysis.py

    This will compare the PID configurations, display position-error, heading-error and PID correction plots and will save the results to results/pid_performance_metrics.csv

COMPONENT B
------------------
The simulation uses three robots and combines consensus communication, PSO target allocation, A* path planning, and neural-network navigation.

To run the project:

1. Install the Python Dependencies:
    python -m pip install matplotlib numpy pandas scikit-learn joblib

2. Run the preparation scripts from the Component B project directory:
    python target_visualization.py
    python machine_learning.py
    python swarm.py

3. Open the Component B world in Webots and run the simulation with the configured controller. The controller requires the trained model at results/navigation_model.joblib and planned routes at data/astar_full_paths.json

4. After the simulation, run:
    python analysis.py

    This analyzes data/consensus_data.csv and saves consensus graphs and analyzed data in the results folder.

OTHER SCRIPTS
------------------
- path_planning.py: Tests A* path planning and saves a sample route plot.

EXPECTED RUNTIME AND HARDWARE
------------------
Python scripts generally take seconds to a few minutes.
Allow approximately 5-15 minutes for a full Webots run,depending on the computer and simulation settings. No specialized hardware is required; a desktop or laptop is sufficient.

At least 8GB RAM and a graphics-capable system are recommended.

DEPENDENCIES
------------------
Component A:
- NumPy
- Matplotlib
- OpenCV
- Pandas
- Tabulate

Component B:
- Matplotlib
- NumPy
- Pandas
- scikit-learn
- joblib

Along with Python standard-library modules. Webots provides the controller API.

STATUS
------
Component A and Component B code are fully completed. A report is provided alongside the project.