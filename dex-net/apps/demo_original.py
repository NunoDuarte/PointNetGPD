import numpy as np
import voxelgrid
import pcl
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPcl
import os
import glob
import time
import torch
from scipy.stats import mode
import open3d as o3d
import multiprocessing as mp
import argparse
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath("__file__")))))
sys.path.append(os.environ['PointNetGPD_FOLDER'] + "/PointNetGPD")
from model.pointnet import PointNetCls
from mayavi import mlab
from pyntcloud import PyntCloud
from numba import jit
from model.gpd import *
from main_test import model

# global config:
yaml_config = YamlConfig(os.environ['PointNetGPD_FOLDER'] + "/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name, os.environ['PointNetGPD_FOLDER'] + "/dex-net/data/grippers")
ags = GpgGraspSamplerPcl(gripper, yaml_config)

parser = argparse.ArgumentParser(description="pointnetGPD")
parser.add_argument("--cuda", action="store_true", default=False)
args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available else False

value_fc = 0.4  # no use, set a random number
num_grasps = 10
num_workers = 8
max_num_samples = 100

minimal_points_send_to_point_net = 20
marker_life_time = 8

show_bad_grasp = False
save_grasp_related_file = False

show_final_grasp = True
single_obj_testing = False  # if True, it will wait for input before get pointcloud
check_pcd_grasp_points = False


def get_voxel_fun(points_, n):
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def get_voxel_fun_win(points, n):
    points_ = PyntCloud.from_instance('open3d', points)
    voxel_id = points_.add_structure('voxelgrid', n_x= n, n_y= n, n_z= n)
    get_voxel = points_.structures[voxel_id]
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def remove_table_points(points_voxel_, vis=True):
    xy_unique = np.unique(points_voxel_[:, 0:2], axis=0)
    new_points_voxel_ = points_voxel_
    pre_del = np.zeros([1])
    for i in range(len(xy_unique)):
        tmp = []
        for j in range(len(points_voxel_)):
            if np.array_equal(points_voxel_[j, 0:2], xy_unique[i]):
                tmp.append(j)
        if len(tmp) < 3:
            tmp = np.array(tmp)
            pre_del = np.hstack([pre_del, tmp])
    if len(pre_del) != 1:
        pre_del = pre_del[1:]
        new_points_voxel_ = np.delete(points_voxel_, pre_del, 0)
    print("Success delete [[ {} ]] points from the table!".format(len(points_voxel_) - len(new_points_voxel_)))

    if vis:
        p = points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = new_points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))  # plot 0 point
        mlab.show()

    return new_points_voxel_


def Resampler(points, k):
    if k <= points.shape[0]:
        rand_idxs = np.random.choice(points.shape[0], k, replace=False)
        return points[rand_idxs, :]
    elif points.shape[0] == k:
        return points
    else:
        rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                    np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
        return points[rand_idxs, :]


def check_hand_points_fun(real_grasp_):
    ind_points_num = []
    for i in range(len(real_grasp_)):
        grasp_bottom_center = real_grasp_[i][4]
        approach_normal = real_grasp_[i][1]
        binormal = real_grasp_[i][2]
        minor_pc = real_grasp_[i][3]
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    binormal, minor_pc, points,
                                                                    local_hand_points, "p_open")
        ind_points_num.append(len(ind_points_tmp))


def check_collision_square(grasp_bottom_center, approach_normal, binormal,
                           minor_pc, points_, p, way="p_open"):
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    grasp_matrix = matrix_.T
    points_ = points_ - grasp_bottom_center.reshape(1, 3)
    tmp = np.dot(grasp_matrix, points_.T)
    points_g = tmp.T
    use_dataset_py = False
    if not use_dataset_py:
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
    # for the way of pointGPD/dataset.py:
    else:
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth
        z_limit = width / 4
        y_limit = width / 2
        x1 = points_g[:, 0] > 0
        x2 = points_g[:, 0] < x_limit
        y1 = points_g[:, 1] > -y_limit
        y2 = points_g[:, 1] < y_limit
        z1 = points_g[:, 2] > -z_limit
        z2 = points_g[:, 2] < z_limit
        a = np.vstack([x1, x2, y1, y2, z1, z2])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

    vis = False

    if vis:
        p = points_g
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        p = points_g[points_in_area]
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.005, color=(0, 1, 0))
        mlab.show()

    return has_p, points_in_area, points_g


def collect_pc(grasp_, pc):
    """
    grasp_bottom_center, normal, major_pc, minor_pc
    """
    grasp_num = len(grasp_)
    grasp_ = np.array(grasp_)
    grasp_ = grasp_.reshape(-1, 5, 3)  # prevent to have grasp that only have number 1
    grasp_bottom_center = grasp_[:, 0]
    approach_normal = grasp_[:, 1]
    binormal = grasp_[:, 2]
    minor_pc = grasp_[:, 3]

    in_ind_ = []
    in_ind_points_ = []
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    for i_ in range(grasp_num):
        has_p, in_ind_tmp, points_g = check_collision_square(grasp_bottom_center[i_], approach_normal[i_],
                                                             binormal[i_], minor_pc[i_], pc, p)
        in_ind_.append(in_ind_tmp)
        in_ind_points_.append(points_g[in_ind_[i_]])
    return in_ind_, in_ind_points_


def cal_grasp(points_):
    start = time.time()
    n = 150 # n_voxel  # parameter related to voxel method
    points_voxel_ = get_voxel_fun(points_, n)
    print('time for getting voxel ', str(time.time() - start))

    if len(points_) < 2000:  # should be a parameter
        while len(points_voxel_) < len(points_) - 15:
            points_voxel_ = get_voxel_fun_win(points_, n)
            n = n + 100

    points_ = points_voxel_
    print('voxel', points_.shape)

    remove_points = False

    if remove_points:
        points_ = remove_table_points(points_, vis=True)

    point_cloud = pcl.PointCloud(points_)
    norm = point_cloud.make_NormalEstimation()
    norm.set_KSearch(30)  # critical parameter when calculating the norms
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]

    # FIXED: cam_pos_
    cam_pos_ = [0, 0, m]

    vector_p2cam = cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    select_point_above_table = 0.01
    # #  modify of gpg: make it as a parameter. avoid select points near the table.
    points_for_sample = points_[points_[:, 2] > select_point_above_table]

    if len(points_for_sample) == 0:
        return [], points_, surface_normal

    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc

    start1 = time.time()
    if True:
        grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps,
                                             max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)

    print('time for generating grasp poses ', str(time.time() - start1))

    check_grasp_points_num = True  # evaluate the number of points in a grasp
    check_hand_points_fun(grasps_together_) if check_grasp_points_num else 0

    in_ind, in_ind_points = collect_pc(grasps_together_, points_)
    score = []  # should be 0 or 1
    score_value = []  # should be float [0, 1]
    ind_good_grasp = []
    ind_bad_grasp = []
    repeat = 1
    real_good_grasp = []
    real_bad_grasp = []
    real_score_value = []

    for ii in range(len(in_ind_points)):
        if in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
            score.append(0)
            score_value.append(0.0)
            if show_bad_grasp:
                ind_bad_grasp.append(ii)
        else:
            predict = []
            grasp_score = []
            for _ in range(repeat):
                if len(in_ind_points[ii]) >= input_points_num:
                    points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                       input_points_num, replace=False)]
                else:
                    points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                       input_points_num, replace=True)]

                if_good_grasp, grasp_score_tmp = detect_network(model.eval(), points_modify)
                predict.append(if_good_grasp.item())
                grasp_score.append(grasp_score_tmp)

            predict_vote = mode(predict)[0][0]  # most occuring in a list

            grasp_score = np.array(grasp_score)
            which_one_is_best = 2

            score_vote = np.mean(grasp_score[np.where(predict == predict_vote)][:, 0, which_one_is_best])
            score.append(predict_vote)
            score_value.append(score_vote)

            if score[ii] == which_one_is_best:
                ind_good_grasp.append(ii)
            else:
                ind_bad_grasp.append(ii)

    print('score_value', score_value)
    print("Got {} good grasps, and {} bad grasps".format(len(ind_good_grasp),
                                                         len(in_ind_points) - len(ind_good_grasp)))

    if len(ind_good_grasp) != 0:
        real_good_grasp = [grasps_together_[i] for i in ind_good_grasp]
        real_score_value = [score_value[i] for i in ind_good_grasp]
        real_bad_grasp = [grasps_together_[i] for i in ind_bad_grasp]

    sorted_value_ind = list(index for index, item in sorted(enumerate(real_score_value),
                                                            key=lambda item: item[1],
                                                            reverse=True))

    sorted_real_good_grasp = [real_good_grasp[i] for i in sorted_value_ind]
    real_good_grasp = sorted_real_good_grasp
    real_score_value = sorted(real_score_value, reverse=True)

    print('Scores of Good grasps: ', real_score_value)

    all_points = point_cloud.to_array()
    for grasp_ in real_good_grasp:
        grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
        approach_normal = grasp_[1]
        binormal = grasp_[2]
        hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, binormal)
        show_grasp_3d(hand_points, color=(0.0, 1.0, 0.0))

    for grasp_ in real_bad_grasp:
        grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
        approach_normal = grasp_[1]
        binormal = grasp_[2]
        hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, binormal)
        show_grasp_3d(hand_points, color=(1.0, 0.0, 0.0))

    ags.show_points(all_points, color='lb', scale_factor=0.01)
    mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
    mlab.show()

    # This doesnt exist
    # if len(real_good_grasp) != 0:
    #     ags.show_detected_grasp(point_cloud, real_good_grasp, show_final_grasp= True)

    print('*' * 80)

    return grasps_together_, points_, surface_normal


def show_grasp_3d(hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color, opacity=0.5)


def detect_network(model_, local_pc):
    local_pc = local_pc.T
    local_pc = local_pc[np.newaxis, ...]
    local_pc = torch.FloatTensor(local_pc)
    #local_pc = local_pc.cuda()
    if args.cuda:
        local_pc = local_pc.cuda()

    output, _ = model_(local_pc)  # N*C
    print('output', output)
    output = output.softmax(1)
    print('output_softmax', output)
    pred = output.data.max(1, keepdim=True)[1]
    output = output.cpu()
    return pred[0], output.data.numpy()


if __name__ == '__main__':
    input_points_num = 750

    for i, object_pointcloud in enumerate(glob.glob(r'*.ply')):
        slm = o3d.io.read_point_cloud(object_pointcloud)
        points = np.array(slm.points)
        m = np.max(points[:, 2])
        points[:, 2] = m - points[:, 2]

        # point_table = points[points[:, 2] < 0.1]
        # point_table = Resampler(point_table, int(len(point_table)/4))
        # print('point_table', len(point_table))
        #
        # point_object = points[points[:, 2] > 0.1]
        # print('point_object', len(point_object))
        #
        # points_reduce = np.concatenate([point_table, point_object], axis= 0)
        points_reduce = points if len(points) < 50000 else Resampler(points, 30000)

        real_grasp, points, normals_cal = cal_grasp(points_reduce)













































