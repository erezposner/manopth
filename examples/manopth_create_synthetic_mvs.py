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


def estimate_rotation_matrix_for_normalized_vectores(A, B):
    v = np.cross(A, B)
    ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + ssc + ssc @ ssc * (1 - np.dot(A, B)) / (np.linalg.norm(v)) ** 2
    # print(R)
    return R, ((R @ A - B) < 0.1).all()


output_path = Path(os.getcwd() + '\out_mvs')
# mesh_distance = 300
number_of_sample = 10000
sample_count = 0
number_of_perspectives = 2
yfov = np.pi / 3.5
skin_rgb_color = [165.0, 126.0, 110.0]
UseLiveView = False
camera_initial_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 350.0],
    [0.0, 0.0, 0.0, 1.0],
])
initialize_lookat = camera_initial_pose[:3, 3] / np.linalg.norm(camera_initial_pose[:3, 3])

show_debug = True
# Initialize MANO layer
# mano_layer = ManoLayer(
#     mano_root='..\\mano\\models', use_pca=True, ncomps=ncomps, flat_hand_mean=False, center_idx=9)
mano_layer = ManoLayer(

    mano_root='..\\mano\\models', use_pca=True, ncomps=ncomps, flat_hand_mean=False, center_idx=0)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
for it in range(1,number_of_sample):
    try:
        if show_debug:
            plt.pause(1)
            plt.close('all')
            ax.clear()
        # Generate random shape parameters
        random_shape = torch.rand(batch_size, 10)
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        random_pose = 4 * (np.random.rand() - 0.5) * torch.rand(batch_size, ncomps + 3)
        random_pose[0, :3] =  torch.from_numpy(8* (np.random.rand(3) - 0.5))   # mask root rotation
        hand_verts, hand_joints = mano_layer(random_pose, random_shape,
                                             torch.from_numpy(np.array([0, 0, 0])).unsqueeze(0).float())

        # print(hand_verts.shape)
        # demo.display_hand({
        #     'verts': hand_verts,
        #     'joints': hand_joints
        # },
        #       mano_faces=mano_layer.th_faces, show=True)
        #
        # plt.draw()
        # plt.pause(.1)
        # plt.show()

        mesh = trimesh.Trimesh(vertices=hand_verts.squeeze().detach().cpu().numpy(),
                               faces=mano_layer.th_faces.squeeze().detach().cpu().numpy())
        mesh.visual.vertex_colors = np.array(skin_rgb_color)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene()
        mn = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mn)

        # m = trimesh.creation.uv_sphere(radius=2)
        # m.visual.vertex_colors = np.array([1.0, 0.0, 0.0])
        # poses = np.tile(np.eye(4), (len(hand_joints[0].detach().cpu().numpy()), 1, 1))
        # poses[:, :3, 3] = hand_joints[0].detach().cpu().numpy()
        # mn1 = pyrender.Mesh.from_trimesh(m, poses=poses)
        # mn1 = pyrender.Node(mesh=mn1, matrix=np.eye(4))
        # scene.add_node(mn1)

        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1)
        # camera = pyrender.OrthographicCamera( xmag =1,ymag=1 ,znear =0.0001,zfar =100000)


        sw = 320
        sh = 320
        nc1 = pyrender.Node(camera=camera, matrix=camera_initial_pose)
        scene.add_node(nc1)
        light = pyrender.SpotLight(color=np.ones(3), intensity=300000.0,
                                   innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=camera_initial_pose)


        data = {}
        data['camera_intrinsic_matrix'] = []
        data['kp_coord_xyz_in_cam_coords'] = []
        data['kp_coord_xyz_in_mano_coords'] = []
        data['verts_coord_xyz_in_cam_coords'] = []
        data['mano_shape'] = []
        data['mano_pose'] = []
        data['depth_map'] = []
        data['rgb_image'] = []
        data['perpective_rt'] = []
        data['kp_points_uv'] = []
        data['mesh_points_uv'] = []
        data['valid_data'] = []


        for perpec in range(number_of_perspectives):
            # scene = pyrender.Scene(bg_color = [124 ,252 ,0])

            new_po = np.eye(4)
            new_po[:3, 3] = camera_initial_pose[:3,3] + [500,500,5] * (np.random.rand(3)-0.5)

            new_lookat = new_po[:3, 3] / np.linalg.norm(new_po[:3, 3])
            r, _ = estimate_rotation_matrix_for_normalized_vectores(initialize_lookat, new_lookat)
            new_po[:3, :3] = r
            if perpec<5:
                new_po=camera_initial_pose
            scene.set_pose(nc1, pose=new_po)
            # print(scene.get_pose(nc1))

            if UseLiveView:
                pyrender.Viewer(scene, use_raymond_lighting=True, point_size=30)

            else:
                # Render the scene
                r = pyrender.OffscreenRenderer(sw, sh)
                color, depth = r.render(scene)

            ## rotate hand mesh and joints to camera world coordinates
            c_hand_joints = torch.from_numpy(np.linalg.inv(new_po)) @ torch.cat((hand_joints[0], torch.ones(hand_joints.shape[1], 1)), dim=1).transpose(
                1, 0).contiguous().double()
            c_hand_verts = torch.from_numpy(np.linalg.inv(new_po)) @ torch.cat((hand_verts[0], torch.ones(hand_verts.shape[1], 1)), dim=1).transpose(
                1, 0).contiguous().double()

            ## Project mesh and joints to camera space
            mesh_points_uv = []
            for j in range(c_hand_verts.shape[1]):
                fu = sh / 2 / (np.tan(yfov / 2))
                fv = sw / 2 / (np.tan(yfov / 2))

                pt = (c_hand_verts[ :, j].detach().cpu().numpy() / (
                c_hand_verts[ :, j].detach().cpu().numpy()[2]))
                pt[0] *= fv
                pt[1] *= fu
                mesh_points_uv.append([sw / 2 - pt[0], sh / 2 + pt[1]])


            kp_points_uv = []
            rays = []
            for j in range(c_hand_joints.shape[1]):
                fu = sh / 2 / (np.tan(yfov / 2))
                fv = sw / 2 / (np.tan(yfov / 2))

                pt = (c_hand_joints[ :, j].detach().cpu().numpy() / (
                c_hand_joints[ :, j].detach().cpu().numpy()[2]))
                rays.append(pt.copy())
                pt[0] *= fv
                pt[1] *= fu
                kp_points_uv.append([sw / 2 - pt[0], sh / 2 + pt[1]])

            rays = np.array(rays)[:, 0:3]

            # from scipy.io import savemat
            #
            # savemat('data.mat', {'rt_matrix':new_po,'pcl_rotated': hand_joints[0].detach().cpu().numpy(),'focal':fu,'pp':sw/2,
            #                      'rays': rays,'pcl_cam_aligned':c_hand_joints.transpose(1,0).detach().cpu().numpy()[:,:3]})
            kp_points_uv = np.array(kp_points_uv)
            mesh_points_uv = np.array(mesh_points_uv)
            if show_debug:
                import matplotlib.pyplot as plt

                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.axis('off')
                # plt.imshow(color)
                # plt.subplot(1, 2, 2)
                # plt.axis('off')
                # plt.imshow(depth, cmap=plt.cm.gray_r)
                # plt.draw()
                # plt.pause(.1)

                plt.figure()
                plt.imshow(color)
                # rev = [0,4,3,2,1,8,7,6,5]
                # for k in range(kp_points_uv.shape[0]):

                plt.plot(kp_points_uv[:, 0], kp_points_uv[:, 1], 'rx')
                plt.plot(mesh_points_uv[:, 0], mesh_points_uv[:, 1], 'bx')
            #
                plt.draw()
                plt.pause(.1)
                # demo.display_hand({
                #     'verts': hand_verts,
                #     'joints': hand_joints
                # },
                #       mano_faces=mano_layer.th_faces, show=True)
                #
                # plt.draw()
                # plt.pause(.1)
                # plt.show()

            valid_data = ((kp_points_uv[:, 0] > 0) & (kp_points_uv[:, 0] < sw) & (kp_points_uv[:, 1] > 0) & (
                    kp_points_uv[:, 1] < sh)).all()
            K = torch.zeros((3, 3))


            # mesh_points_uv = []
            # for j in range(c_hand_verts.shape[1]):
            fu = sh / 2 / (np.tan(yfov / 2))
            fv = sw / 2 / (np.tan(yfov / 2))
            K[0, 0] = fu
            K[1, 1] = fv
            K[2, 2] = 1.0
            K[0, 2] = sw / 2
            K[1, 2] = sh / 2
            data['camera_intrinsic_matrix'] = K.detach().cpu().numpy()
            data['kp_coord_xyz_in_cam_coords'].append(c_hand_joints.transpose(1,0).detach().cpu().numpy()[np.newaxis,:,:3])
            data['kp_coord_xyz_in_mano_coords'].append(hand_joints.detach().cpu().numpy())
            data['verts_coord_xyz_in_cam_coords'].append(c_hand_verts.transpose(1,0).detach().cpu().numpy()[np.newaxis,:,:3])
            data['mano_shape'].append(random_shape.detach().cpu().numpy())
            data['mano_pose'].append(random_pose.detach().cpu().numpy())
            data['depth_map'].append(depth)
            data['rgb_image'].append(color)
            data['perpective_rt'].append(new_po)
            data['kp_points_uv'].append(kp_points_uv[np.newaxis,...])
            data['mesh_points_uv'].append(mesh_points_uv[np.newaxis,...])
            data['valid_data'].append(valid_data)

        if all(data['valid_data']):
            # print('valid sample number {}'.format(sample_count))
            np.savez(output_path / 'data_{}.npz'.format(it),
                     verts_coord_xyz_in_cam_coords=data['verts_coord_xyz_in_cam_coords'],
                     mano_shape=data['mano_shape'],
                     mano_pose=data['mano_pose'],
                     kp_coord_uv=data['kp_points_uv'],
                     perpective_rt=data['perpective_rt'],
                     kp_coord_xyz_in_mano_coords=data['kp_coord_xyz_in_mano_coords'],
                     kp_coord_xyz_in_cam_coords=data['kp_coord_xyz_in_cam_coords'],
                     depth_map=data['depth_map'],
                     valid_data=data['valid_data'],
                     mesh_points_uv=data['mesh_points_uv'],
                     rgb_image=data['rgb_image'])
            print('{}-th iteration - {}-th valid sample'.format(it,sample_count))
            sample_count += 1

        # print(hand_verts.shape)
        # demo.display_hand({
        #     'verts': hand_verts,
        #     'joints': c_hand_joints
        # },
        #     mano_faces=mano_layer.th_faces)


    except:
        pass
