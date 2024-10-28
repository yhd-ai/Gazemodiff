"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_h36m.py
"""

import numpy as np
import os
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetMogaze(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3d_mogaze_onlyskeleton_global_highquility2.npz')
     
        self.subjects_split = {'train': ['p1_1','p1_2','p2_1', 'p5_1','p4_1','p6_1', 'p6_2',],
                               'test':['p7_1','p7_3'], }
        self.subjects = [x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[0, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                                     joints_left=[5, 6, 7, 8, 13, 14, 15, 16],
                                     joints_right=[9, 10, 11, 12, 17, 18, 19, 20])
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(21) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        self.S1_skeleton = data_o['p1_1']['pick_001_pose_xyz'][:, self.kept_joints].copy()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                #print(action)
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                #seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    dataset = DatasetMogaze('train', actions=actions)
    generator = dataset.sampling_generator()
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)
