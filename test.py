# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import random
import tqdm
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.assets import RigidObject, RigidObjectCfg



def define_origins(n: int, m: int, spacing: float) -> list[list[float]]:
    """
    Defines the origins of a 3D grid with n * n particles per layer and m layers stacked along the z-axis.

    Args:
        n (int): The number of particles per row/column in each layer (n * n grid).
        m (int): The number of layers stacked along the z-axis.
        spacing (float): The spacing between particles in the grid and between layers.

    Returns:
        list[list[float]]: A list of origins, where each origin is a 3D coordinate [x, y, z].
    """
    # Calculate the total number of origins
    num_origins = n * n * m

    # Initialize a tensor to store all origins
    env_origins = torch.zeros(num_origins, 3)

    # Create 2D grid coordinates for the n x n grid in each layer
    xx, yy = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="xy")
    xx = xx.flatten() * spacing - spacing * (n - 1) / 2
    yy = yy.flatten() * spacing - spacing * (n - 1) / 2

    # Fill in the coordinates for each layer
    for layer in range(m):
        start_idx = layer * n * n
        end_idx = start_idx + n * n

        # Set x, y, and z coordinates for this layer
        env_origins[start_idx:end_idx, 0] = xx
        env_origins[start_idx:end_idx, 1] = yy
        env_origins[start_idx:end_idx, 2] = layer * spacing

    # Convert the origins to a list of lists and return
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    current_path = os.getcwd()
 

    # -- Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", scale=(1.5, 2.0, 1.0))
    cfg.func("/World/Origin1/Table", cfg, translation=(0.0, 0.0, 1))

    # -- bowl
    bowl_cfg = sim_utils.UsdFileCfg(usd_path=f"{current_path}/bowl/bowl.usd", 
                                    scale=(1, 1, 1), 
                                    # making a rigid object static
                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                    # collision_props=sim_utils.CollisionPropertiesCfg(),
                                    )
    bowl_cfg.func("/World/Origin1/Bowl", bowl_cfg, translation=(0.5, 0.25, 1))

    # -- cup
    cup_cfg = sim_utils.UsdFileCfg(usd_path=f"{current_path}/cup/cup.usd", 
                                    scale=(1.6, 1.6, 1.6), 
                                    # making a rigid object static
                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                    # collision_props=sim_utils.CollisionPropertiesCfg(),
                                    )
    cup_cfg.func("/World/Origin1/Cup", cup_cfg, translation=(0.5, -0.25, 1))



    # add soft
    soft_cfg_1, soft_origins_1 = add_soft()
    deformable_object_1 = DeformableObject(cfg=soft_cfg_1)


    # add solid
    rigid_cfg, rigid_origins = add_rigid()
    rigid_object = RigidObject(cfg=rigid_cfg)
    

    # add rigid

    # return the scene information
    scene_entities = {
        "deformable_object_1": deformable_object_1,
        "rigid_object": rigid_object,
    }


    return scene_entities, rigid_origins

def add_soft():
    
    radius = 0.025

    # add soft
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=radius,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )

  
    objects_cfg = {
        "sphere": cfg_sphere,
    }

    # define amount of object
    soft_origins = define_origins(n = 2, m = 2, spacing=radius * 2.1)

    for i in range(len(soft_origins)):

        # soft_origins[i][2] += 1.1   =>  without penetration
        # soft_origins[i][2] += 1.2   =>  with    penetration

        soft_origins[i][2] += 1.1
        soft_origins[i][1] += 0.25 
        soft_origins[i][0] += 0.5 
    

    for idx, origin in tqdm.tqdm(enumerate(soft_origins), total=len(soft_origins)):

        obj_name = random.choice(list(objects_cfg.keys()))
        obj_cfg = objects_cfg[obj_name]
        obj_cfg.physics_material.youngs_modulus = 1e4
        obj_cfg.physics_material.poissons_ratio = 0.4
        obj_cfg.physics_material.density = None
        # obj_cfg.physics_material.dynamic_friction = 0.25
        obj_cfg.physics_material.elasticity_damping = 0.00001
        obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())

        # spawn the object
        obj_cfg.func(f"/World/Origin/Object{idx:02d}", obj_cfg, translation=origin)
        
    soft_cfg = DeformableObjectCfg(
        prim_path="/World/Origin/Object.*",
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
    )

    return soft_cfg, soft_origins

def add_rigid():
    radius = 0.005

    cfg_Sphere = sim_utils.SphereCfg(
        radius=radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )


    origins = define_origins(n = 5, m=30, spacing=radius * 2)
    for i in range(len(origins)):
        origins[i][2] += 1.05
        origins[i][1] += -0.25
        origins[i][0] += 0.5


    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin/rigidObject{i:02d}", "Xform", translation=origin)

    rigid_cfg = RigidObjectCfg(
        prim_path="/World/Origin/rigidObject.*/s",
        spawn=cfg_Sphere,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    return rigid_cfg, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 400 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, objects in enumerate(entities.values()):
                # set soft
                if index == 0 :
                    nodal_state = objects.data.default_nodal_state_w.clone()
                    objects.write_nodal_state_to_sim(nodal_state)
                    # reset the internal state
                    objects.reset()
                # set rigid
                else :
                    nodal_state = objects.data.default_root_state.clone()
                    nodal_state[:, :3] += origins
                    objects.write_root_state_to_sim(nodal_state)
                    # reset the internal state
                    objects.reset()
                
            print("[INFO]: Resetting robots state...")
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.8, 0.0, 2.3], [0.0, 0.0, 1])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
