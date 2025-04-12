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
import h5py


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

from pcd_functions import Pcd_functions

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
import yaml

def add_soft(): 

    r_radius = round(random.uniform(0.0025, 0.01), 4)
    l_radius = round(random.uniform(0.0025, 0.01), 4)
    mass = round(random.uniform(0.0001, 0.005), 4)
    friction = round(random.uniform(0, 1),2)
    max_num = int(256/pow(2, (r_radius - 0.0025)*1000)) 
    amount = random.randint(1, max(1, max_num))+2

    amount = 2
    r_radius = 0.008
    l_radius = 0.008

    # youngs_modulus = round(random.uniform(0.0001, 0.005), 4)

    soft_origins = define_origins(n = 4, layer = amount, spacing=max(r_radius, l_radius) * 2.1)


    # Define deformable material properties (required for soft bodies)
    soft_material = sim_utils.DeformableBodyMaterialCfg(
        youngs_modulus=1e5,
        poissons_ratio=0.4,
        density=None,  # Optional
        elasticity_damping=0.00001,
        dynamic_friction = friction,
    )

    # ----use predefine Sphere shape 
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=r_radius,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = soft_material,
    )

    cfg_cube = sim_utils.MeshCuboidCfg(
        size=(r_radius * 2, r_radius * 2, l_radius * 2),
        # hight=l_radius * 2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = soft_material,
    )

    cfg_cone = sim_utils.MeshConeCfg(
        radius=r_radius,
        height=l_radius* 2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = soft_material,
    )

    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=r_radius,
        height=l_radius* 2,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = soft_material,

    )


    
    '''
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
    '''

    # ---- determine which cfg file be used (cfg_sphere / str_cfg)
    shapes = [cfg_sphere, cfg_cube, cfg_cone, cfg_cylinder]
    obj_cfg = random.choice(shapes)
    # obj_cfg = cfg_cylinder

    obj_cfg.semantic_tags = [("class", "food")]


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

    r_radius = round(random.uniform(0.0025, 0.01), 4)
    l_radius = round(random.uniform(0.0025, 0.01), 4)
    mass = round(random.uniform(0.0001, 0.005), 4)
    friction = round(random.uniform(0, 1),2)
    max_num = int(256/pow(2, (r_radius - 0.0025)*1000)) 
    amount = random.randint(1, max(1, max_num))+2

    amount = 1


    rigid_origins = define_origins(n = 5, layer = amount, spacing=max(r_radius, l_radius) * 2)
    # str_usd_path ="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/str.usd"


    cfg_sphere = sim_utils.SphereCfg(
        radius = r_radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
    )

    cfg_cube = sim_utils.CuboidCfg(
        size=(r_radius*2, r_radius*2, l_radius*2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
    )

    cfg_cone = sim_utils.ConeCfg(
        radius=r_radius,
        height=l_radius*2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
    )

    cfg_cylinder = sim_utils.CylinderCfg(
        radius=r_radius,
        height=l_radius*2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
    )


    # str_cfg = sim_utils.UsdFileCfg(
    #     usd_path=str_usd_path, 
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     # mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    #     # collision_props=sim_utils.CollisionPropertiesCfg(),
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
    #     scale = (0.1, 0.1, 0.1), 
    # )

    shapes = [cfg_sphere, cfg_cube, cfg_cone, cfg_cylinder]
    obj_cfg = random.choice(shapes)
    # obj_cfg = cfg_cylinder

    obj_cfg.semantic_tags = [("class", "food")]


    for idx, origin in tqdm.tqdm(enumerate(rigid_origins), total=len(rigid_origins)):
        obj_cfg.func(f"/World/rigid/Object{idx:02d}", obj_cfg, translation=origin)

        

    rigid_cfg = RigidObjectCfg(
        prim_path=f"/World/rigid/Object.*",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
        # debug_vis=True,
    )

    # return rigid_cfg, food_info
    return rigid_cfg

def add_camera(cam_type = "front"):

    front_cam_pose = np.load("./real_cam_pose/front_cam2base.npy")
    front_cam_pos = front_cam_pose[0:3, 3]
    front_cam_rot = Rot.from_matrix(front_cam_pose[0:3, 0:3]).as_quat()

    back_cam_pose = np.load("./real_cam_pose/back_cam2base.npy")
    back_cam_pos = back_cam_pose[0:3, 3]
    back_cam_rot = Rot.from_matrix(back_cam_pose[0:3, 0:3]).as_quat()

    focal_length = 16.6

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
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
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
            # focal_length=19.6, focus_distance=400.0, horizontal_aperture=20.955, 
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=20.955, 
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.CuboidCfg(
    #                 size=(2, 2, 1),
    #                 # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #             ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    # )

    # articulation

    robot = SCOOP_FRANKA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # soft body
    # soft_object = add_soft()

    rigid_object = add_rigid()
    # rigid_object, food_info = add_rigid()


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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.575, -0.11, 0.025)),
    )



class DataCollection():
    def __init__(self, mean_eepose_qua, init_pose = None, food_info = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.franka_init_pose = init_pose
        self.mean_eepose_qua = mean_eepose_qua
        self.offset_generate()

        self.gt_front = np.load("./real_cam_pose/front_cam2base.npy")
        self.gt_back = np.load("./real_cam_pose/back_cam2base.npy")

        # need modify
        self.ref_bowl = np.load("./real_food.npy")

        self.init_spoon_pcd = np.load("./ori_init_spoon_pcd.npy")
        self.offset_list = np.load("new_pcd_offset_list.npy")
        
        self.config_file = "./collect_time.yaml"

        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.count = int(self.config['count'])
        count = int(self.count) + 1
        self.config['count'] = count
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.config, file)

        self.eepose_offset = 0.045
        self.pcd_functions = Pcd_functions()

        self.num_envs = 1

        self.bowl_semantic_id = None
        self.food_semantic_id = None

        self.action_horizon = 12

        # self.r_radius, self.l_radius, self.mass, self.friction  = food_info
        

        # robot proprioception data
        self.record_ee_pose = [[] for _ in range(self.num_envs)]

        # front camera data
        self.front_rgb_list = [[] for _ in range(self.num_envs)]
        self.front_depth_list = [[] for _ in range(self.num_envs)]
        self.front_seg_list = [[] for _ in range(self.num_envs)]
        self.mix_all_pcd_list = [[] for _ in range(self.num_envs)]

        # back camera data
        self.back_rgb_list = [[] for _ in range(self.num_envs)]
        self.back_depth_list = [[] for _ in range(self.num_envs)]
        self.back_seg_list = [[] for _ in range(self.num_envs)]
        self.back_pcd_color_list = [[] for _ in range(self.num_envs)]

        self.spillage_amount = [[] for _ in range(self.num_envs)]
        self.scooped_amount = [[] for _ in range(self.num_envs)]
        self.spillage_vol = [[] for _ in range(self.num_envs)]
        self.scooped_vol = [[] for _ in range(self.num_envs)]
        self.spillage_type = [[] for _ in range(self.num_envs)]
        self.scooped_type = [[] for _ in range(self.num_envs)]
        self.binary_spillage = [[] for _ in range(self.num_envs)]
        self.binary_scoop = [[] for _ in range(self.num_envs)]
        self.pre_spillage = np.zeros(self.num_envs)

        self.action_space = {
            "left": torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "right" : torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "forward": torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "backward": torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "up": torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "down": torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.], [0.]]]).to(self.device),
            "rest": torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.], [0.]]]).to(self.device),
        }

    def offset_generate(self):
        
        # need to modify the offset
        front_step = 20
        self.U_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step
        self.D_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step
        self.L_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step
        self.R_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step
        self.B_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step
        self.F_index = np.array(random.sample(range(len(self.mean_eepose_qua)- front_step), random.randint(0, 50))) + front_step

    def apply_offset(self, index):

        offset_weight = 600

        offset = torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.],[0.]]]).to(self.device)
 
        if index in self.L_index :
            offset +=self.action_space.get("left")
        
        if index in self.R_index :
            offset += self.action_space.get("right")

        if index in self.D_index :
            offset += self.action_space.get("down")

        if index in self.F_index :
            offset += self.action_space.get("forward")

        if index in self.B_index :
            offset += self.action_space.get("backward")

        if index in self.U_index :
            offset += self.action_space.get("up")

        return offset.squeeze(0).squeeze(-1) / offset_weight

    def get_info(self, robot_entity_cfg = None):


        front_rgb_image  = self.front_camera.data.output["rgb"][0].cpu().numpy()
        front_depth_image  = self.front_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        front_seg_image  = self.front_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        back_rgb_image  = self.back_camera.data.output["rgb"][0].cpu().numpy()
        back_depth_image  = self.back_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        back_seg_image  = self.back_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        # plt.imshow(front_rgb_image)
        # plt.show()
        # simulation_app.close()


        food_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "food", object_id = self.food_semantic_id)
        back_food_world = self.pcd_functions.transform_to_world(food_pcd[:, :3], self.gt_back)
        object_seg = np.full((back_food_world.shape[0], 1), 2)
        back_food_world = np.hstack((back_food_world, object_seg))

        bowl_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "bowl", object_id = self.bowl_semantic_id)
        back_bowl_world = self.pcd_functions.transform_to_world(bowl_pcd[:, :3], self.gt_back)
        # object_seg must be 4
        object_seg = np.full((back_bowl_world.shape[0], 1), 3)
        back_bowl_world = np.hstack((back_bowl_world, object_seg))


        bowl_pcd = self.pcd_functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "bowl", object_id = self.bowl_semantic_id)
        front_bowl_world = self.pcd_functions.transform_to_world(bowl_pcd[:, :3], self.gt_front)
        # object_seg must be 4
        object_seg = np.full((front_bowl_world.shape[0], 1), 5)
        front_bowl_world = np.hstack((front_bowl_world, object_seg))


        # get eepose
        sim_current_pose = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        self.record_ee_pose[0].append(sim_current_pose[0].to("cpu"))

        # get tool 
        trans_tool = self.pcd_functions.from_ee_to_spoon(sim_current_pose[0], self.init_spoon_pcd)
        object_seg = np.full((trans_tool.shape[0], 1), 1)
        trans_tool = np.hstack((trans_tool, object_seg))

        # mix_all_pcd = np.concatenate(( trans_tool, back_food_world, self.ref_bowl), axis=0)
        mix_all_pcd = np.concatenate((back_bowl_world, self.ref_bowl, front_bowl_world), axis=0)
        mix_all_pcd = self.pcd_functions.align_point_cloud(mix_all_pcd, target_points = 30000)
        mix_all_nor_pcd = self.pcd_functions.nor_pcd(mix_all_pcd)
        self.pcd_functions.check_pcd_color(mix_all_nor_pcd)
        simulation_app.close()


    
        self.front_rgb_list[0].append(front_rgb_image)
        self.front_depth_list[0].append(front_depth_image)
        self.front_seg_list[0].append(front_seg_image)

        self.back_rgb_list[0].append(back_rgb_image)
        self.back_depth_list[0].append(back_depth_image)
        self.back_seg_list[0].append(back_seg_image)

        self.mix_all_pcd_list[0].append(mix_all_nor_pcd)

    
    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        self.robot = scene["robot"]
        self.front_camera = scene["front_camera"]
        self.back_camera = scene["back_camera"]
        self.device = sim.device

        reset_frame = 105


        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # real-world trajectory
        ee_goals = torch.tensor(self.mean_eepose_qua, device=sim.device)

        # modify the trajectory to simulation
        modify_ee_goals = self.eepose_real2sim_offset(ee_goals)
        modify_ee_goals = modify_ee_goals.clone().detach().to(sim.device)
        
        # Create buffers to store actions
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=self.device)
        ik_commands[:] = modify_ee_goals[0]

        # robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_finger_left"])
        
        # Resolving the scene entities
        robot_entity_cfg.resolve(scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self.robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0

        frame_num = 0
        current_goal_idx = 1
        goal_pose = modify_ee_goals[0]

        # hvae no idea why semantic_filter does not work, so use the id_to_labels to filter first
        # Find the ID of the bowl in the semantic segmentation

        

        id_to_labels = self.back_camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        for semantic_id_str, label_info in id_to_labels.items():
            if label_info.get("class") == "bowl":
                self.bowl_semantic_id = int(semantic_id_str)
            if label_info.get("class") == "food":
                self.food_semantic_id = int(semantic_id_str)


        # Simulation loop
        while simulation_app.is_running():
            # init set
            print(f"frame_num: {frame_num}")
            if frame_num <= reset_frame:
                init_joint =  torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]], device = sim.device)

                if frame_num == reset_frame :
                    # init_joint =  torch.tensor([[1.77321586, -0.82879892, -1.79624463, -1.65110402, -0.77492251, 1.80433866, -0.78061272, 0, 0]], device = sim.device)
                    init_joint =  torch.tensor(self.franka_init_pose, device = sim.device)
                    # self.cal_spillage_scooped(scene = scene, reset = 1)

                joint_vel = self.robot.data.default_joint_vel.clone()
                joint_pos = init_joint
                ik_commands[:] = goal_pose
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
                self.robot.reset()

            else :
                # scooping speed
                if frame_num % 5 == 0:
    
                    self.get_info(robot_entity_cfg)
                    offset = self.apply_offset(current_goal_idx)
                    modify_ee_goals += offset
                    
                    goal_pose = modify_ee_goals[current_goal_idx]

                    joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    ik_commands[:] = goal_pose
                    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                    # # reset controller
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(ik_commands)


                    if current_goal_idx % self.action_horizon == 0 :
                        print("current_goal_idx : ", current_goal_idx)
                        # self.cal_spillage_scooped(scene = scene, reset = 0)

                    # change goal
                    current_goal_idx += 0
                    if current_goal_idx == len(modify_ee_goals) :
                        self.record_info()
                        break
              
                    
   
            # obtain quantities from simulation
            jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


            frame_num += 1  
            # apply actions
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()

            sim_time += sim_dt
            # update sim-time
            # update buffers
            scene.update(sim_dt)

            '''
            # vis offset
            ee_pose_w = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_pose_w = self.eepose_sim2real_offset(ee_pose_w.to("cpu"))
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ee_goals[current_goal_idx].unsqueeze(0)[:, 0:3], ee_goals[current_goal_idx].unsqueeze(0)[:, 3:7])
            '''
            
    def record_info(self):

        record_len = self.action_horizon * len(self.spillage_amount[0])

        mix_all_pcd = self.list_to_nparray(self.mix_all_pcd_list)[:record_len]
        sim_eepose = self.list_to_nparray(self.record_ee_pose)[:record_len]
        real_eepose = self.eepose_sim2real_offset(sim_eepose)[:record_len].to("cpu")

        spillage_amount = self.list_to_nparray(self.spillage_amount)
        scoop_amount = self.list_to_nparray(self.scooped_amount)
        spillage_vol = self.list_to_nparray(self.spillage_vol)
        scoop_vol = self.list_to_nparray(self.scooped_vol)
        
        binary_spillage = self.list_to_nparray(self.binary_spillage)
        binary_scoop = self.list_to_nparray(self.binary_scoop)

        # r_radius, l_radius, mass, friction = self.food_info

        # if sum(binary_spillage) > 2 :
        #     weight_spillage = np.ones(8)
        # else : 
        #     weight_spillage = binary_spillage
        

        
        
   
        data_dict = {
            'eepose' : real_eepose,
            'mix_all_pcd' : mix_all_pcd,
            'spillage_amount': spillage_amount,
            # 'spillage_vol': spillage_vol,
            'binary_spillage' : binary_spillage,
            'scoop_amount': scoop_amount,
            # 'scoop_vol': scoop_vol,
            'binary_scoop' : binary_scoop,
            # 'r_radius' : self.r_radius,
            # 'l_radius' : self.l_radius,
            # 'mass' : self.mass,
            # 'friction' : self.friction,

            # 'amount' : self.ball_amount,
            # 'shape' : self.food_label,
        }

        # store the data
        with h5py.File(f'{f"/media/hcis-s22/data/spillage_dataset/time_{self.count}"}.h5', 'w') as h5file:
            for key, value in data_dict.items():
                h5file.create_dataset(key, data=value)
            



    def eepose_sim2real_offset(self, sim_qua_list):

        update_qua_list = []

        for sim_qua in sim_qua_list:
            real_quat_rotation_xyzw = np.array([sim_qua[4], sim_qua[5], sim_qua[6], sim_qua[3]])
            rotation = Rot.from_quat(real_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()
            z_rot = rotation_matrix[:, 2]
            updata_qua_pose = [sim_qua[0], sim_qua[1], sim_qua[2]] + z_rot * self.eepose_offset

            update_qua = np.array([updata_qua_pose[0], updata_qua_pose[1], updata_qua_pose[2], sim_qua[3], sim_qua[4], sim_qua[5], sim_qua[6]])
            update_qua_list.append(update_qua)

        return torch.tensor(np.array(update_qua_list)).to(self.device)


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

        return torch.tensor(np.array(update_qua_list)).to(self.device)
    
    def cal_spillage_scooped(self, env_index = 0, reset = 0, scene = None):
        # reset = 1 means record init spillage in experiment setting 
        current_spillage = 0
        scoop_amount = 0

        rigid_object = scene["rigid_object"].data.body_link_state_w
        y_pose = rigid_object[:,0, 1].to("cpu")
        z_pose = rigid_object[:,0, 2].to("cpu")

        spillage_mask = np.logical_or(z_pose < 0, y_pose > -0.02)
        current_spillage = np.count_nonzero(spillage_mask)

        scoop_mask = np.logical_or(z_pose > 0.15, np.logical_and(z_pose > 0, y_pose > 0))
        scoop_amount = np.count_nonzero(scoop_mask)


        
        if reset == 0:
         
            spillage_amount = current_spillage - self.pre_spillage[env_index]
            # spillage_vol = spillage_amount * (self.ball_radius**3) * 10**9
            # scoop_vol = scoop_amount * (self.ball_radius**3)* 10**9
            
            if int(spillage_amount) == 0:
                self.binary_spillage[env_index].append(0)
            else :
                self.binary_spillage[env_index].append(1)
    
            if int(scoop_amount) == 0:
                self.binary_scoop[env_index].append(0)
            else :
                self.binary_scoop[env_index].append(1)
          

            self.spillage_amount[env_index].append(int(spillage_amount))
            self.scooped_amount[env_index].append(int(scoop_amount))
            # self.spillage_vol[env_index].append(int(spillage_vol))
            # self.scooped_vol[env_index].append(int(scoop_vol))


            print(f"spillage amount :{int(spillage_amount)}")
            print(f"scoop_num : {int(scoop_amount)}")
       
        self.pre_spillage[env_index] = int(current_spillage)
        
    def list_to_nparray(self, lists):
        temp_array = []

        for i in range(len(lists)):
            temp_array.append(np.array(lists[i]))

        temp = np.stack(temp_array, axis=0)

        shape = temp.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        temp_1 = temp.reshape(new_shape )
 
        return temp_1
    


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=1/180, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.5, 0, 0.8], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # food_info = scene_cfg.food_info
 
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    start_step = 0
    franka_init_pose = np.load("mean_traj.npy")[start_step]
    franka_init_pose = np.append(franka_init_pose, 0)
    franka_init_pose = np.append(franka_init_pose, 0)
    franka_init_pose = torch.tensor(franka_init_pose, dtype=torch.float32).unsqueeze(0)

    ee_goals = np.load("mean_eepose_qua.npy")[start_step:160]

    env = DataCollection(mean_eepose_qua=ee_goals, init_pose = franka_init_pose, food_info = None)
    env.run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # modify IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py line 658 to shud down the app immediately
    simulation_app.close()
    
