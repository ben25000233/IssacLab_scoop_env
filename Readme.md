
## Build environments

Isaaclab pakage

## Set parameters

# In interactive_scene.py : 
line 48 : 
put spoon_franka.py under "/home/user-name/{path}/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots"
line 410 :
put s_bowl.usd under "/home/user_name/{path}/IsaacLab/source/isaaclab_assets/data"

# In spoon_franka.py
line 30 : 
put spoon_franka.urdf under /home/user_name/{path}/IsaacLab/URDF
line 31 : 
set usd_dir = "/home/user_name/{path}/IsaacLab/source/isaaclab_assets/data/franka"
## Run
```
python interactive_scene.py
```

