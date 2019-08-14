import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np

batch_size = 1
# Select number of principal components for pose space
ncomps = 5
import trimesh
import pyrender
from pathlib import Path
import os
output_path = Path(os.getcwd() + '\out')
mesh_distance = 300
number_of_sample = 5000
sample_count =0
yfov = np.pi / 3.5
show_debug = True
# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='..\\mano\\models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
for it in range(number_of_sample):
    try:
        # Generate random shape parameters
        random_shape = torch.rand(batch_size, 10)
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        random_pose = 2 * np.random.rand() * torch.rand(batch_size, ncomps + 3)
        hand_verts, hand_joints = mano_layer(random_pose, random_shape)

        # print(hand_verts.shape)
        # demo.display_hand({
        #     'verts': hand_verts,
        #     'joints': hand_joints
        # },
        #     mano_faces=mano_layer.th_faces)

        hand_verts[:, :, 2] -= mesh_distance
        hand_joints[:, :, 2] -= mesh_distance


        ver = hand_verts.squeeze().detach().cpu().numpy()
        faces = mano_layer.th_faces.squeeze().detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=ver,
                               faces=faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        # scene = pyrender.Scene(bg_color = [124 ,252 ,0])
        scene = pyrender.Scene()
        scene.add(mesh)

        m = trimesh.creation.uv_sphere(radius=2)
        m.visual.vertex_colors = np.array([1.0, 0.0, 0.0])
        poses = np.tile(np.eye(4), (len(hand_joints[0].detach().cpu().numpy()), 1, 1))
        poses[:, :3, 3] = hand_joints[0].detach().cpu().numpy()
        mn = pyrender.Mesh.from_trimesh(m, poses=poses)
        # scene.add(mn)

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up

        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1)

        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)
        # Set up the light -- a single spot light in the same spot as the camera
        light = pyrender.SpotLight(color=np.ones(3), intensity=300000.0,
                                   innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=camera_pose)
        sw = 320
        sh = 320
        # # Render the scene
        r = pyrender.OffscreenRenderer(sw, sh)
        color, depth = r.render(scene)
        #
        # import matplotlib.pyplot as plt

        # plt.imshow(color)
        # plt.show()
        # pyrender.Viewer(scene, use_raymond_lighting=True, point_size=30)
        # Show the images
        data = {}
        data['kp_coord_xyz'] = hand_joints
        data['verts_coord_xyz'] = hand_verts
        data['mano_shape'] = random_shape
        data['mano_pose'] = random_pose
        data['depth_map'] = depth
        data['rgb_image'] = color

        kp_points_uv = []
        for j in range(hand_joints.shape[1]):
            fu = sh / 2 / (np.tan(yfov/2))
            fv = sw / 2 / (np.tan(yfov/2))

            pt = (hand_joints[0, j, :].detach().cpu().numpy() / hand_joints[0, j, :].detach().cpu().numpy()[2])
            pt[0] *= fv
            pt[1] *= fu

            kp_points_uv.append([sw / 2 - pt[0], sh / 2 + pt[1]])
        kp_points_uv = np.array(kp_points_uv)
        if show_debug:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(color)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(depth, cmap=plt.cm.gray_r)
            plt.show()
            plt.figure()
            plt.imshow(data['rgb_image'])
            plt.plot(kp_points_uv[:,0], kp_points_uv[:,1], 'yx')

            plt.show()

        valid_data = ((kp_points_uv[:, 0] > 0) & (kp_points_uv[:, 0] < sw) & (kp_points_uv[:, 1] > 0) & (
                    kp_points_uv[:, 1] < sh)).all()

        if valid_data:
            print('valid sample number {}'.format(sample_count))
            np.savez(output_path / 'data_{}.npz'.format(it), kp_coord_xyz=hand_joints,
                     verts_coord_xyz=hand_verts,
                     mano_shape=random_shape,
                     mano_pose=random_pose,
                     kp_coord_uv=kp_points_uv,
                     rgb_image=data['rgb_image'])
            sample_count+=1

            # print(hand_verts.shape)
            # demo.display_hand({
            #     'verts': hand_verts,
            #     'joints': hand_joints
            # },
            #     mano_faces=mano_layer.th_faces)


    except:
        pass
