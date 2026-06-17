# **Vision-Language Navigation for Path Planning using LLMs and Reinforcement Learning**

This project implements a **vision-language navigation (VLN) and autonomous path planning pipeline** for the **Yahboom ROSMASTER X3 PLUS** robot on **ROS2 Humble**. It combines **SLAM-based localisation and mapping**, an **HMA-RRT\* path planner**, and a **YOLO-based visual perception module** to let the robot understand natural-language tasks (e.g. *"bring me a cup of water"*), locate the relevant object, plan a collision-free path to it, and navigate there.

## **Project Overview**
The pipeline includes:
1. **Perception**:
   - Arm-mounted camera sweep across multiple angles using YOLO object detection.
   - Real-time RGB + LiDAR sensor fusion for spatial awareness.

2. **Localisation and Mapping (SLAM)**:
   - Occupancy grid mapping using log-odds updates from LiDAR scans.
   - Continuous pose tracking and explored-area estimation.

3. **Path Planning**:
   - RRT\* based planner with dynamic region-based sampling, APF-guided expansion, and a hierarchical retreat (escape) mechanism.
   - A\* fallback planner using the same occupancy grid (`lidar_slam_planner.py`).
   - Obstacle inflation via binary dilation and obstacle-aware path smoothing.

4. **Navigation Execution**:
   - Virtual robot movement along the planned path, with live odometry and TF broadcasting for RViz visualisation.
   - Arm interaction to reach/grip the target object on arrival.

5. **Evaluation Metrics**:
   - **Navigation Error**: distance between final and goal position.
   - **SPL (Success weighted by Path Length)**: efficiency of the path taken relative to the optimal path.
   - **Planning Time, Nodes Generated, Waypoints**: planner performance statistics.

---

## **Project Structure**

```plaintext
.
├── main.py                          # Unified node: SLAM + RRT* navigation + YOLO task execution
├── slam_module.py                   # SLAM logic — occupancy mapping and pose tracking
├── rrt_virtual_mover.py             # HMA-RRT* path planner + virtual robot mover
├── lidar_slam_planner.py            # Occupancy map builder + A* path planner (fallback)
├── virtual_mover.py                 # Simulates robot movement along a planned path
├── arm_camera_scan.py                # Arm camera sweep + YOLO-based object scanning
├── launch/
│   └── x3_rtabmap_depth.launch.py   # ROS2 launch file for RTAB-Map depth pipeline
├── results/
│   ├── 2n8kARJN3HM_graph.json       # Navigation run output (graph data)
│   ├── 2n8kARJN3HM_path_report.txt  # Navigation run output (path report)
│   └── multi_scene_results/         # Aggregated results across multiple test scenes
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/SabudhFoundation/passion-project-aab.git
cd passion-project-aab
```

### **2. Install Dependencies**
This project runs on **ROS2 Humble** under **Ubuntu 22.04** (via WSL on Windows). Create a virtual environment and install the Python dependencies:
```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### **3. ROS2 and Hardware Setup**
Ensure the following are available:
- ROS2 Humble (`ros-humble-desktop`) installed and sourced.
- YDLiDAR ROS2 driver for LiDAR input.
- Yahboom ROSMASTER X3 PLUS camera/arm drivers (or simulated equivalents).

---

## **Execution Steps**

### **1. Launch the LiDAR Driver**
```bash
ros2 launch ydlidar_ros2_driver ydlidar.launch.py
```

### **2. Run the Unified Navigation Node**
```bash
python3 main.py
```

### **3. Trigger a Task**
Send a natural-language task (the robot will scan, localise, plan, navigate, and interact):
```bash
ros2 topic pub /task std_msgs/String "{data: 'cup of water'}" --once
```

Or send a raw XY goal directly, bypassing vision:
```bash
ros2 topic pub /goal geometry_msgs/Point "{x: 2.0, y: 1.5, z: 0.0}" --once
```

### **4. Visualise in RViz**
```
Fixed Frame : map
Add topics  : /map, /rrt_path, /virtual_pose, /rrt_tree (MarkerArray)
```

---

## **Results**
The results include:
1. **Navigation Reports**:
   - `2n8kARJN3HM_path_report.txt`, `2n8kARJN3HM_graph.json`

2. **Multi-Scene Evaluation**:
   - `multi_scene_results/` — aggregated metrics across multiple test environments.

3. **Metrics Comparison**:
   The final navigation runs were evaluated on:
   - **Navigation Error** (m)
   - **SPL (Success weighted Path Length)**
   - **Planning Time** (s)
   - **Waypoints / Nodes Generated**

---

## **Key Functions**

| File                      | Description                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| `main.py`                  | Unified node — runs the full task pipeline: arm scan, localise, estimate, plan, navigate, interact. |
| `slam_module.py`           | Builds the occupancy grid from LiDAR scans and tracks robot pose in real time. |
| `rrt_virtual_mover.py`     | HMA-RRT\* planner — dynamic sampling, APF expansion, hierarchical retreat, and virtual movement along the planned path. |
| `lidar_slam_planner.py`    | Builds the occupancy map and runs an A\* fallback planner over the same map.   |
| `virtual_mover.py`         | Moves the virtual robot along a path published by the planner, with live TF/odometry broadcasting. |
| `arm_camera_scan.py`       | Sweeps the arm-mounted camera across a configurable arc, capturing frames for YOLO-based object detection. |

---

## **Dependencies**
- Python 3.10+ (ROS2 Humble requirement)
- ROS2 Humble (`ros-humble-desktop`)
- Required libraries:
  - `numpy`
  - `scipy`
  - `rclpy`
  - `opencv-python`
  - `ultralytics` (YOLO)

Install Python dependencies using `pip install -r requirements.txt`.

---

## **Contributions**
This project was developed as part of a Data Science internship at **Sabudh Foundation**, under the mentorship of **Dr.&nbsp;Sukhjit Singh Sehra**.

Contributors:
- Amit Kumar Giri
- Abhaynoor Singh
- Birinder Singh Bhinder
