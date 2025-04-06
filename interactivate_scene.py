import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import matplotlib.pyplot as plt


import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
import isaacsim.core.utils.prims as prim_utils

from isaaclab.sensors import CameraCfg, TiledCameraCfg

from pcd_functions import Functions

##
# Pre-defined configs
##

# /home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/ (spoon_franka.py / franka.py)
from isaaclab_assets import SCOOP_FRANKA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    DeformableObject, 
    DeformableObjectCfg,
)

import random
import tqdm
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

def add_soft(): 

    radius = 0.015
    soft_origins = define_origins(n = 3, layer = 3, spacing=radius * 2)
    str_usd_path ="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/str.usd"


    # Define deformable material properties (required for soft bodies)
    soft_material = sim_utils.DeformableBodyMaterialCfg(
        youngs_modulus=1e5,
        poissons_ratio=0.4,
        density=None,  # Optional
        elasticity_damping=0.00001
    )

    # ----use predefine Sphere shape 
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=radius,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material = soft_material
    )
    
    # ----use own usd 
    
    # Define MeshCfg for the soft body
    soft_body_cfg = sim_utils.MeshCfg(
        visual_material=sim_utils.PreviewSurfaceCfg(),  # Assign visual material 
        physics_material=soft_material,  # Assign soft body physics properties
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),  # Deformable physics properties
    )

    # import usd file 
    str_cfg = sim_utils.UsdFileCfg(
            usd_path=str_usd_path, 
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            scale = (0.1, 0.1, 0.1)
            )

    # ---- determine which cfg file be used (cfg_sphere / str_cfg)
    obj_cfg = cfg_sphere


    for idx, origin in tqdm.tqdm(enumerate(soft_origins), total=len(soft_origins)):
        obj_cfg.func(f"/World/soft/Object{idx:02d}", obj_cfg, translation=origin)
        

    soft_cfg = DeformableObjectCfg(
        prim_path=f"/World/soft/Object.*",
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
        debug_vis=True,
    )

    return soft_cfg

def add_rigid(): 

    radius = 0.015
    rigid_origins = define_origins(n = 3, layer = 3, spacing=radius * 2)
    str_usd_path ="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/str.usd"


    cfg_sphere = sim_utils.SphereCfg(
        radius = radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),

    )

    cfg_cube = sim_utils.CuboidCfg(
        size=(radius, radius, radius),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )

    cfg_cone = sim_utils.ConeCfg(
        radius=radius,
        height=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )

    cfg_cylinder = sim_utils.CylinderCfg(
        radius=radius,
        height=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )


    str_cfg = sim_utils.UsdFileCfg(
        usd_path=str_usd_path, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        # mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        # collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        scale = (0.1, 0.1, 0.1), 
    )


    obj_cfg = cfg_cylinder
    obj_cfg.semantic_tags = [("class", "food")]


    for idx, origin in tqdm.tqdm(enumerate(rigid_origins), total=len(rigid_origins)):
        obj_cfg.func(f"/World/rigid/Object{idx:02d}", obj_cfg, translation=origin)
        

    rigid_cfg = RigidObjectCfg(
        prim_path=f"/World/rigid/Object.*",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
        # debug_vis=True,
    )

    return rigid_cfg

def add_camera(cam_type = "front"):

    front_cam_pose = np.load("./real_cam_pose/front_cam2base.npy")
    front_cam_pos = front_cam_pose[0:3, 3]
    front_cam_rot = Rot.from_matrix(front_cam_pose[0:3, 0:3]).as_quat()

    back_cam_pose = np.load("./real_cam_pose/back_cam2base.npy")
    back_cam_pos = back_cam_pose[0:3, 3]
    back_cam_rot = Rot.from_matrix(back_cam_pose[0:3, 0:3]).as_quat()

    if cam_type == "front":
        cam_pos = (front_cam_pos[0], front_cam_pos[1], front_cam_pos[2])
        cam_rot = (front_cam_rot[3], front_cam_rot[0], front_cam_rot[1], front_cam_rot[2])

        camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.1,
        height=960,
        width=1280,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
        semantic_filter = "class:bowl",
    )
    else:
        cam_pos = (back_cam_pos[0], back_cam_pos[1], back_cam_pos[2])
        cam_rot = (back_cam_rot[3], back_cam_rot[0], back_cam_rot[1], back_cam_rot[2])

        camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/back_cam",
        update_period=0.1,
        height=960,
        width=1280,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
    )



    return camera

    
def define_origins(n: int, layer: int, spacing: float) -> list[list[float]]:
    """
    Defines the origins of a 3D grid with n * n particles per layer and m layers stacked along the z-axis.

    Args:
        n (int): The number of particles per row/column in each layer (n * n grid).
        layer (int): The number of layers stacked along the z-axis.
        spacing (float): The spacing between particles in the grid and between layers.

    Returns:
        list[list[float]]: A list of origins, where each origin is a 3D coordinate [x, y, z].
    """
    # Calculate the total number of origins
    num_origins = n * n * layer

    # Initialize a tensor to store all origins
    env_origins = torch.zeros(num_origins, 3)

    # Create 2D grid coordinates for the n x n grid in each layer
    xx, yy = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="xy")
    xx = xx.flatten() * spacing - spacing * (n - 1) / 2
    yy = yy.flatten() * spacing - spacing * (n - 1) / 2

    # Fill in the coordinates for each layer
    for layer in range(layer):
        start_idx = layer * n * n
        end_idx = start_idx + n * n

        # Set x, y, and z coordinates for this layer
        env_origins[start_idx:end_idx, 0] = xx + 0.59
        env_origins[start_idx:end_idx, 1] = yy - 0.11
        env_origins[start_idx:end_idx, 2] = layer * spacing + 0.1

    # Convert the origins to a list of lists and return
    return env_origins.tolist()



@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
                    size=(2, 2, 1),
                    # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # articulation

    robot = SCOOP_FRANKA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # soft body
    # soft_object = add_soft()

    rigid_object = add_rigid()

    front_camera = add_camera("front")
    back_camera = add_camera("back")
 
    bowl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/tool/bowl/s_bowl.usd", 
            scale=(1, 1, 1), 
            # making a rigid object static
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            semantic_tags = [("class", "bowl"), ("id", "2")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.59, -0.11, 0.05)),
    )






class DataCollection():
    def __init__(self, mean_traj, mean_eepose_euler, mean_eepose_qua):
        self.mean_traj = mean_traj
        self.mean_eepose_euler = mean_eepose_euler
        self.mean_eepose_qua = mean_eepose_qua

        self.eepose_offset = 0.045
        self.Functions = Functions()

        self.num_envs = 1

        # robot proprioception data
        self.record_ee_pose = [[] for _ in range(self.num_envs)]

        # front camera data
        self.front_rgb_list = [[] for _ in range(self.num_envs)]
        self.front_depth_list = [[] for _ in range(self.num_envs)]
        self.front_seg_list = [[] for _ in range(self.num_envs)]
        self.front_pcd_color_list = [[] for _ in range(self.num_envs)]

        # back camera data
        self.back_rgb_list = [[] for _ in range(self.num_envs)]
        self.back_depth_list = [[] for _ in range(self.num_envs)]
        self.back_seg_list = [[] for _ in range(self.num_envs)]
        self.back_pcd_color_list = [[] for _ in range(self.num_envs)]

    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        robot = scene["robot"]
        front_camera = scene["front_camera"]
        back_camera = scene["back_camera"]
        self.device = sim.device

        # camera_positions = torch.tensor([[1.5, 0, 0.8]], device=sim.device)
        # camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
        # camera.set_world_poses_from_view(camera_positions, camera_targets)


        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # real-world trajectory
        ee_goals = np.load("mean_eepose_qua.npy") 
        ee_goals = torch.tensor(ee_goals, device=sim.device)

        # modify the trajectory to simulation
        modify_ee_goals = self.eepose_real2sim_offset(ee_goals)
        modify_ee_goals = torch.tensor(modify_ee_goals, device=sim.device)

        init_joint =  torch.tensor([[1.77321586, -0.82879892, -1.79624463, -1.65110402, -0.77492251, 1.80433866, -0.78061272, 0.04, 0.04]], device = sim.device)
        
        # Create buffers to store actions
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = modify_ee_goals[0]

        # robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_finger_left"])
        
        # Resolving the scene entities
        robot_entity_cfg.resolve(scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0

        count = 0
        current_goal_idx = 0

        bowl_semantic_id = None
        food_semantic_id = None
        # Simulation loop
        while simulation_app.is_running():
            # init set

            goal_pose = modify_ee_goals[current_goal_idx]
            if count == 0 :
                # goal_pose = modify_ee_goals[current_goal_idx]
                joint_vel = robot.data.default_joint_vel.clone()
                joint_pos = init_joint
                ik_commands[:] = goal_pose
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()

            sim_current_pose = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            real_current_pose = self.eepose_sim2real_offset(sim_current_pose)
            target_pose = ee_goals[current_goal_idx].unsqueeze(0)
            dis = torch.norm(target_pose.to('cpu') - real_current_pose.to('cpu'), p=2).item()
        
            # scooping speed
            if count % 10 == 1:
   
                # change goal
                current_goal_idx += 1
                goal_pose = modify_ee_goals[current_goal_idx]

                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                ik_commands[:] = goal_pose
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)


                front_rgb_image  = front_camera.data.output["rgb"][0].cpu().numpy()
                front_depth_image  = front_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                front_seg_image  = front_camera.data.output["semantic_segmentation"][0].cpu().numpy()

                back_rgb_image  = back_camera.data.output["rgb"][0].cpu().numpy()
                back_depth_image  = back_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                back_seg_image  = back_camera.data.output["semantic_segmentation"][0].cpu().numpy()

                

                # hvae no idea why semantic_filter does not work, so use the id_to_labels to filter first
                # Find the ID of the bowl in the semantic segmentation

                id_to_labels = front_camera.data.info[0]["semantic_segmentation"]["idToLabels"]
                
                if bowl_semantic_id or food_semantic_id is None:
                    for semantic_id_str, label_info in id_to_labels.items():
                        if label_info.get("class") == "bowl":
                            bowl_semantic_id = int(semantic_id_str)
                        if label_info.get("class") == "food":
                            food_semantic_id = int(semantic_id_str)
    

                food_pcd = self.Functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "food", object_id = food_semantic_id)
                bowl_pcd = self.Functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "bowl", object_id = bowl_semantic_id)
                # self.Functions.check_pcd_color(food_pcd)
                # self.Functions.check_pcd_color(bowl_pcd)
    
                # pcd = self.Functions.depth_to_point_cloud(back_depth_image[..., 0], mask=back_seg_image[..., 0], object_id=2)
                # self.Functions.check_pcd_color(pcd)
                # front_camera.segantic_filter("class:bowl")
          
                # plt.imshow(bowl_rgb_image)
                # plt.show()
                # exit()

                self.front_rgb_list[0].append(front_rgb_image)
                self.front_depth_list[0].append(front_depth_image)
                self.front_seg_list[0].append(front_seg_image)

                self.back_rgb_list[0].append(back_rgb_image)
                self.back_depth_list[0].append(back_depth_image)
                self.back_seg_list[0].append(back_seg_image)
                

   
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


            count += 1  
            # apply actions
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()

            sim_time += sim_dt
            # update sim-time
            # update buffers
            scene.update(sim_dt)

            # obtain quantities from simulation
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

            # vis offset
            ee_pose_w = self.eepose_sim2real_offset(ee_pose_w)
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(target_pose[:, 0:3], target_pose[:, 3:7])

            
      
            
     


    def eepose_sim2real_offset(self, sim_qua_list):
        sim_qua_list = sim_qua_list.to("cpu")
        update_qua_list = []

        for sim_qua in sim_qua_list:
            real_quat_rotation_xyzw = np.array([sim_qua[4], sim_qua[5], sim_qua[6], sim_qua[3]])
            rotation = Rot.from_quat(real_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()
            z_rot = rotation_matrix[:, 2]
            updata_qua_pose = [sim_qua[0], sim_qua[1], sim_qua[2]] + z_rot * self.eepose_offset

            update_qua = np.array([updata_qua_pose[0], updata_qua_pose[1], updata_qua_pose[2], sim_qua[3], sim_qua[4], sim_qua[5], sim_qua[6]])
            update_qua_list.append(update_qua)

        return torch.tensor(update_qua_list).to(self.device)


    def eepose_real2sim_offset(self, real_qua_list):
        real_qua_list = real_qua_list.to("cpu")
        update_qua_list = []

        for real_qua in real_qua_list:
            real_quat_rotation_xyzw = np.array([real_qua[4], real_qua[5], real_qua[6], real_qua[3]])
            rotation = Rot.from_quat(real_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()
            z_rot = rotation_matrix[:, 2]
            updata_qua_pose = [real_qua[0], real_qua[1], real_qua[2]] - z_rot * self.eepose_offset
            update_qua = np.array([updata_qua_pose[0], updata_qua_pose[1], updata_qua_pose[2], real_qua[3], real_qua[4], real_qua[5], real_qua[6]])
            update_qua_list.append(update_qua)

        return torch.tensor(update_qua_list).to(self.device)
    

    


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=1/180, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.5, 0, 0.8], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    env = DataCollection(mean_traj=None, mean_eepose_euler=None, mean_eepose_qua=None)
    env.run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
