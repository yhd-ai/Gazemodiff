import numpy as np

path = './data/mogazelocal_multi_modal/mogazelocal_multimodal_t_his15_t_pred60_skiprate15.npz'
path_ind = './data/mogazelocal_multi_modal/t_his15_1_thre0.400_t_pred60_thre0.100_index_subp1_1.npz'

data = np.load(path,allow_pickle=True)
print(data.files)
pose = data['data_candidate']
print(pose.shape)

data_ind = np.load(path_ind,allow_pickle=True)
print(data_ind.files)
pose_m = data_ind['data_multimodal']
print(pose_m)



