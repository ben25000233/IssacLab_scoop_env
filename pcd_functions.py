import numpy as np
import open3d as o3d
import torch

class Functions():
    def __init__(self):
        pass
    def depth_to_point_cloud(self, depth, segmantation, object_id = None, object_type = None, device="cpu"):
        
        mask = (segmantation == object_id)
    
        intrinsic = self.compute_camera_intrinsics_matrix(640, 480, 65)
        h, w = depth.shape
    
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        if mask is not None:
            points = points[mask.reshape(-1), :]

        # points = self.align_point_cloud(points)

        if object_type == "food" :
            object_seg = np.full((points.shape[0], 1), 2)
        elif object_type == "bowl":
            object_seg = np.full((points.shape[0], 1), 4)

        # Concatenate the column to the original array
        seg_pcd = np.hstack((points, object_seg))

        return seg_pcd
    def compute_camera_intrinsics_matrix(self,image_width, image_heigth, horizontal_fov):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
        K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]])
        return K

    def transform_to_world(self,points, extrinsic):

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world_homogeneous = (extrinsic @ points_homogeneous.T).T
        points_world = points_world_homogeneous[:, :3]

        return points_world

    def nor_pcd(self, points):

        seg_info = points[:, 3].reshape(len(points), 1)
        points = points[:, :3]

        # normalize the pcd
        
        centroid = np.mean(points, axis=0)
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))


        centroid = [ 0.59115381, -0.1113387 ,  0.0755547 ]
        m = 0.6797932094342392
        
        points = points - centroid
        points = points / m

        seg_pcd = np.concatenate((points, seg_info), axis=1)
        return seg_pcd
    
    def align_point_cloud(self, points, target_points=10000):
        num_points = len(points)
    
        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            indices = np.sort(indices)

        else:
            # Resample with replacement to reach target_points
            indices = np.random.choice(num_points, target_points, replace=True)
            indices = np.sort(indices)

        new_pcd = np.asarray(points)[indices]
        
        return new_pcd

    
    def check_pcd_color(self, pcd):

        color_map = {
            0: [1, 0, 0],    # Red
            4: [0, 1, 0],    # Green
            1: [0, 0, 1],    # Blue
            3: [1, 1, 0],    # Yellow
            5: [1, 0, 1],     # Magenta
            2: [1, 0.5, 0]
        }
        points = []
        colors = []
    
        
        for i in range(pcd.shape[0]):
            points.append(pcd[i][:3])
            if pcd.shape[1] == 4:
                colors.append(color_map[pcd[i][3]])


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])

    def get_translation_matrix(self, x, y, z):
        return torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
        ])

    def from_ee_to_spoon(self, ee_point, pcd):
        # Convert inputs to torch tensors
        ee_point = torch.tensor(ee_point, dtype=torch.float32)
        pcd = torch.tensor(pcd, dtype=torch.float32)
        offsets = torch.tensor(self.offset_list, dtype=torch.float32)  # Shape: [N, 3]

        # Precompute constants
        adjust_matrix = torch.tensor(
            Rot.from_euler("XYZ", (0, 180, 90), degrees=True).as_matrix(),
            dtype=torch.float32,
        )

        # Create the end-effector pose matrix (4x4)
        ee_pose = torch.eye(4, dtype=torch.float32)
        ee_pose[0:3, 3] = ee_point[0:3]  # Translation
        rotation = Rot.from_quat([ee_point[4], ee_point[5], ee_point[6], ee_point[3]])
        rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
        ee_pose[0:3, 0:3] = torch.mm(rotation_matrix, adjust_matrix)

        # Create transformation matrices for offsets (Batch)
        T_spoon_to_center = torch.eye(4, dtype=torch.float32).repeat(len(offsets), 1, 1)  # [N, 4, 4]
        T_spoon_to_center[:, 0:3, 3] = offsets

        # Batch multiply: [N, 4, 4] @ [4, 4] -> [N, 4, 4]
        spoon_poses = torch.matmul(ee_pose.unsqueeze(0), T_spoon_to_center)

        # Extract positions: [N, 3]
        new_pcd = spoon_poses[:, 0:3, 3]

        return new_pcd.numpy()