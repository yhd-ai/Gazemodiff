"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_h36m.py
"""

import numpy as np
import os
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetGimo_withcontext(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3d_mogaze_onlyskeleton_global_highquility_gimo_realgaze.npz')
     
        self.subjects_split = {'train': ['train'],
                               'test': ['test']}
        self.subjects = [x for x in self.subjects_split[self.mode]]
        """"0pelvis", "1left_hip", "2right_hip", "3spine1", "4left_knee", "5right_knee", 
            "6spine2", "7left_ankle", "8right_ankle", "9spine3", "10left_foot", "11right_foot",
            "12neck", "13left_collar", "14right_collar", "15head", "16left_shoulder", "17right_shoulder",
            "18left_elbow", "19right_elbow", "20left_wrist", "21right_wrist", "22jaw", "23gaze", "24head_direction"
                             """
        self.skeleton = Skeleton(parents=[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,19,15,15],
                                     joints_left=[1, 4, 7, 10, 13, 16, 18, 20],
                                     joints_right=[2, 5, 8, 11, 14, 17, 19, 21])
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(24) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        self.S1_skeleton = data_o['train']['bedroom0122_2022-01-21-194925_242_pose_xyz'][:, self.kept_joints].copy()
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
