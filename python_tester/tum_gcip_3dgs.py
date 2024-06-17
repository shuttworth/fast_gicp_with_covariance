#!/usr/bin/python3
import os
import sys
import time
import cv2
import pygicp
import numpy as np
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import open3d as o3d
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints

def quaternion_rotation_matrix(Q, t):
	r = R.from_quat(Q)
	rotation_mat = r.as_matrix()
	T = np.empty((4, 4))
	T[:3, :3] = rotation_mat
	T[:3, 3] = [t[0], t[1], t[2]]

	T[3, :] = [0, 0, 0, 1]     
	return T

def tum_load_poses(path):
	poses = []
	times = []
    # 记住，此处的关联文件修改非常重要 
	# association_file = open("associations/fr1_desk.txt")
	association_file = open("associations/fr3_office.txt")
	for association_line in association_file:
		association_line = association_line.split()
		time_dis = 99
		final_idx = 0
		with open(path, "r") as f:
			lines = f.readlines()

		for i in range(len(lines)):
			if i < 3:
				continue
			line = lines[i].split()
			t = abs(float(line[0]) - float(association_line[2]))
			if time_dis > t:
				time_dis = t
				final_idx = i
    
		line = lines[final_idx].split()
		xyz = np.array([  	float(line[1]),
									float(line[2]),
									float(line[3])])
		q = np.array([	float(line[4]),
						float(line[5]),
						float(line[6]),
						float(line[7])])
		c2w = quaternion_rotation_matrix(q, xyz)
		poses.append(c2w)
		times.append(line[0])
	return np.array(poses), times

# 初始化时候使用，下采样
def set_downsample_filter(downsample_scale):
    W = 640
    H = 480
    fx = 535.4
    fy = 539.2
    cx = 320.1
    cy = 247.6
    # Get sampling idxs
    sample_interval = downsample_scale
    h_val = sample_interval * torch.arange(0,int( H/sample_interval)+1)
    h_val = h_val-1
    h_val[0] = 0
    h_val = h_val * W
    a, b = torch.meshgrid(h_val, torch.arange(0,W,sample_interval))
    # For tensor indexing, we need tuple
    pick_idxs = ((a+b).flatten(),)
    # Get u, v values
    v, u = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
    u = u.flatten()[pick_idxs]
    v = v.flatten()[pick_idxs]
    
    # Calculate xy values, not multiplied with z_values
    x_pre = (u-cx)/fx # * z_values
    y_pre = (v-cy)/fy # * z_values
    
    return pick_idxs, x_pre, y_pre

def downsample_and_make_pointcloud2(depth_img, rgb_img, downsample_idxs, x_pre, y_pre, depth_trunc, depth_scale):
    colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[downsample_idxs]/255
    z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[downsample_idxs]/depth_scale
    zero_filter = torch.where(z_values!=0)
    filter = torch.where(z_values[zero_filter]<=depth_trunc)
    # Trackable gaussians (will be used in tracking)
    z_values = z_values[zero_filter]
    x = x_pre[zero_filter] * z_values
    y = y_pre[zero_filter] * z_values
    points = torch.stack([x,y,z_values], dim=-1)
    colors = colors[zero_filter]
    
    return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()


def main():
    if len(sys.argv) < 2:
        print('usage: tum_step2.py /path/to/TUM/sequence')
        return

    # List input files
    seq_path = sys.argv[1]
    print("seq_path = ",seq_path)


    downsample_rate = 5
    downsample_resolution = 0.03
    knn_max_distance = 99999.0
    visualize = False
    depth_scale = 5000.0
    depth_trunc = 3.0

    downsample_idxs, x_pre, y_pre = set_downsample_filter(downsample_rate)

    depth_filenames = sorted([seq_path + '/depth/' + x for x in os.listdir(seq_path+'/depth/') if x.endswith('.png')])
    rgb_filenames = sorted([seq_path + '/rgb/' + x for x in os.listdir(seq_path+'/rgb/') if x.endswith('.png')])
    num_images = len(rgb_filenames)

    gt_poses, gt_timestamps = tum_load_poses(seq_path + '/groundtruth.txt')

    gt_traj_vis = np.array([x[:3, 3] for x in gt_poses])

    reg = pygicp.FastGICP()

    # reg.set_num_threads(8)
    reg.set_max_correspondence_distance(downsample_resolution)
    reg.set_max_knn_distance(knn_max_distance)

    stamps = []		# for FPS calculation
    poses = [gt_poses[0]]  # sensor trajectory

    for i, (depth_filename, rgb_filename) in enumerate(zip(depth_filenames, rgb_filenames)):
        # Read depth image
        depth_image = np.array(o3d.io.read_image(depth_filename))
        rgb_image = cv2.imread(rgb_filename)
        current_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # fps
        start = time.time()
  
        # Make pointcloud
        # 这个函数是较为重要的，从深度图和彩色图中制作点云
        # points在下方诸多地方都使用到了，因为就是点云点嘛
        # 而colors、z_values、trackable_filter作为之后的input_values里的传参
        points, colors, z_values, trackable_filter = downsample_and_make_pointcloud2(depth_image, current_image, downsample_idxs, x_pre, y_pre , depth_scale , depth_trunc)


        # 初始化高斯
        # shared_new_gaussians = SharedGaussians(points.shape[0])

        if i == 0:
            current_pose = poses[-1]
            print('WSA i = ', i, " current_pose = ",current_pose)

            h_points = np.column_stack((points, np.ones(len(points))))
            points_registered = np.dot(current_pose, h_points.T).T
            points_registered = points_registered[:, :3]
            reg.set_input_target(points_registered)


            # 反转位姿以获得相机外在参数（旋转和平移）
            # current_pose = np.linalg.inv(current_pose)
            # T = current_pose[:3,3]
            # R = current_pose[:3,:3].transpose()
            # points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
            # reg.set_input_target(points)

            # num_trackable_points = trackable_filter.shape[0]
            # input_filter = np.zeros(points.shape[0], dtype=np.int32)
            # input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
            
            # reg.set_target_filter(num_trackable_points, input_filter)
            # reg.calculate_target_covariance_with_filter()
            
            # rots = reg.get_target_rotationsq()
            # scales = reg.get_target_scales()
            # rots = np.reshape(rots, (-1,4))
            # scales = np.reshape(scales, (-1,3))

            # Assign first gaussian to shared memory
            # 共享内存中的高斯分布
            # shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors), 
            #                                            torch.tensor(rots), torch.tensor(scales), 
            #                                            torch.tensor(z_values), torch.tensor(trackable_filter))

        else:
            print('WSA i = ', i)
            reg.set_input_source(points)
            # num_trackable_points = trackable_filter.shape[0]
            # input_filter = np.zeros(points.shape[0], dtype=np.int32)
            # input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
            # reg.set_source_filter(num_trackable_points, input_filter)

            initial_pose = poses[-1]
            print('WSA initial_pose = ', initial_pose)
            current_pose = reg.align(initial_pose)
            print('WSA current_pose')
            # reg.swap_source_and_target()
            poses.append(current_pose)

            points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)

        # fps
        stamps.append(1/(time.time()-start))
        stamps_ = stamps[-9:]
        fps = sum(stamps_) / len(stamps_)
        print('fps:%.3f' % fps)

        # # visualize pointcloud
        # if visualize and i%5==0:
        #     points_.transform(poses[-1])
        #     vis.add_geometry(points_)
        #     vis.update_geometry(points_)
        #     vis.poll_events()
        #     vis.update_renderer()

        # Plot the estimated trajectory
        traj = np.array([x[:3, 3] for x in poses])

        if i % 10 == 0:
            pyplot.clf()
            pyplot.title(f'Downsample ratio {downsample_resolution}\nfps : {fps:.2f}')
            pyplot.plot(traj[:, 0], traj[:, 1], label='g-icp trajectory', linewidth=3)
            pyplot.legend()
            pyplot.plot(gt_traj_vis[:, 0], gt_traj_vis[:, 1], label='ground truth trajectory')
            pyplot.legend()
            pyplot.axis('equal')
            pyplot.pause(0.01)

    time.sleep(10)
    print("pause 10s")

if __name__ == '__main__':
    print("start")
    main()
