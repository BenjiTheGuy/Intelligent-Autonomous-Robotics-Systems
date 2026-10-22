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
Open the Component A world in Webots and run the assigned robot controller. Refer to the Component A project files for its specific setup and controls.

COMPONENT B
------------------
The simulation uses three robots and combines consensus communication, PSO target allocation, A* path planning, and neural-network navigation.

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
Python scripts generally take seconds to a few minutes. Allow approximately 5-15 minutes for a full Webots run, depending on the computer and simulation settings. No specialized hardware is required; a desktop or laptop is sufficient. At least 8GB RAM and a graphics-capable system are recommended.

DEPENDENCIES
------------------
Component B uses matplotlib, numpy, pandas, scikit-learn, and joblib, along with Python standard-library modules. Webots provides the controller API.

STATUS
------
Component B code is fully completed. Additional Details will follow which are due in the report. As well as the completion of Component A.