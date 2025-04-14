# 🛠️ IsaacLab Spoon Scene Setup

This guide explains how to build the environment and set up parameters for running the `interactive_scene.py` script with a custom spoon and bowl setup in IsaacLab.

---

## 📦 Build Environment

Ensure the **IsaacLab** package is installed and properly set up.

---

## ⚙️ Set Parameters

### 🔧 Edit `interactive_scene.py`

- **Line 48**  
  Move `spoon_franka.py` to:  /home/user-name/{path}/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
- **Line 410**  
Move `s_bowl.usd` to:  /home/user-name/{path}/IsaacLab/source/isaaclab_assets/data
---

### 🔧 Edit `spoon_franka.py`

- **Line 30**  
Move `spoon_franka.urdf` to:  /home/user-name/{path}/IsaacLab/URDF


- **Line 31**  
Set the `usd_dir` variable :  usd_dir = "/home/user-name/{path}/IsaacLab/source/isaaclab_assets/data/franka"

---
## ▶️ Run the Simulation
python interactive_scene.py



