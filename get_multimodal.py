import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import torch.distributions as D


sys.path.append(os.getcwd())
from config import Config, update_config
from utils import *
from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_humaneva import DatasetHumanEva
from data_loader.dataset_mogaze import DatasetMogaze
from data_loader.dataset_mogaze_withcontext import DatasetMogaze_withgaze
from data_loader.dataset_gimo import DatasetGimo
from data_loader.dataset_gimo_withcontext import DatasetGimo_withcontext

# from models.motion_pred import *
# from models.motion_pred_naf import *
from utils import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', default='p1_1')
    parser.add_argument('--act', default='Walking')
    args = parser.parse_args()

    """data"""
    dataset = 'mogaze_withcontext'
    dataset_cls = DatasetH36M if dataset == 'h36m' else DatasetHumanEva if dataset == 'human_eva' else DatasetMogaze if dataset == 'mogaze' else DatasetMogaze_withgaze if dataset =='mogaze_withcontext' else DatasetGimo if dataset == 'gimo' else DatasetGimo_withcontext
    dataset = dataset_cls('train', 0, 1, actions='all', use_vel=False)
    dataset_test = dataset_cls('test', 0, 1, actions='all', use_vel=False)
    print(dataset_test.subjects)
    data = dataset_test.data
    t_his = 15
    t_pre = 60
    parents = dataset_test.skeleton.parents()

    margin_f = 1
    thre_his = 0.4
    thre_pred = 0.1

    st = time.time()
    #  get all possible sequences
    skip_rate = 15

    data_candidate = []
    for sub in dataset_test.subjects:
        for key in data[sub].keys():
            data_tmp = np.copy(data[sub][key])
            
            data_tmp[:, 0] = 0
            nf = data_tmp.shape[0]
            idxs = np.arange(0, nf - t_his - t_pre, skip_rate)[:, None] + np.arange(t_his + t_pre)[None, :]
            data_tmp = data_tmp[idxs]

            # validation
            #data_tmp1 = util.absolute2relative(data_tmp, parents=parents)
            # data_tmp2 = util.absolute2relative(data_tmp1, parents=parents, invert=True, x0=data_tmp[:1, :1])
            # print(f'recovery error {np.max(np.abs(data_tmp2 - data_tmp)):.3f}')

            # data_tmp = util.absolute2relative(data_tmp, parents=parents)

            data_candidate.append(data_tmp)
    data_candidate = np.concatenate(data_candidate, axis=0)
    print(data_candidate.shape)
    np.savez_compressed(f'./data/mogaze_head_multi_modal_gaze/mogazeglobal_withcontext_t_his{t_his:d}_t_pred{t_pre:d}_skiprate{skip_rate}.npz',
                        data_candidate=data_candidate)

    data_candidate = np.load(f'./data/mogaze_head_multi_modal_gaze/mogazeglobal_withcontext_t_his{t_his:d}_t_pred{t_pre:d}_skiprate{skip_rate}.npz')[
        'data_candidate']
    # data_candidate = \
    #     np.load('./data/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz', allow_pickle=True)[
    #         'data_candidate.npy']

    data_multimodal = {}
    for sub in dataset_test.subjects:
        data_sub = {}
        """if sub not in args.sub:
            continue
            """
        for key in data[sub].keys():
            # if str.lower(args.act) not in str.lower(key):
            #     continue
            st = time.time()
            data_key = {}
            data_tmp = np.copy(data[sub][key])
            data_tmp[:, 0] = 0
            nf = data_tmp.shape[0]
            """
            candi_tmp = util.absolute2relative(data_candidate, parents=parents, invert=True,
                                               x0=data_tmp[None, ...][:, :1])
                                               """
            idxs = np.arange(0, nf - t_his - t_pre + 1)[:, None] + np.arange(t_his + t_pre)[None, :]

            # observation distance
            dist_his = np.mean(np.linalg.norm(data_tmp[idxs][:, t_his - margin_f:t_his, 1:][:, None, ...] -
                                              data_candidate[:, t_his - margin_f:t_his, 1:][None, ...], axis=4),
                               axis=(2, 3))

            for idx in np.arange(0, nf - t_his - t_pre + 1):
                dist_h = dist_his[idx]
                # dist_p = dist_pred[idx]

                idx_his = np.where(dist_h <= thre_his)[0]
                candi_tmp_tmp = data_candidate[idx_his]
                traj = data_tmp[idx:idx + t_his + t_pre]
                x0 = np.copy(traj[None, ...])
                x0[:, :, 0] = 0

                # future distance
                dist_pred = np.mean(np.linalg.norm(x0[:, t_his:, 1:] -
                                                   candi_tmp_tmp[:, t_his:, 1:], axis=3), axis=(1, 2))
                idx_pred = np.where(dist_pred >= thre_pred)[0]
                idx_cand = idx_his[idx_pred]

                # traj_multi = candi_tmp_tmp[idx_pred]
                data_key[idx] = idx_cand
                # data_key[f'{idx}_dist_his'] = dist_h[idx_his[idx_pred]]
                # data_key[f'{idx}_dist_pred'] = dist_pred[idx_pred]

            data_sub[key] = data_key
            #print(data_key)
            print(f'>>> time used for {sub}_{key}: {time.time() - st:.3f}')
            # break
        data_multimodal[sub] = data_sub
        # break
    np.savez_compressed(
        f'./data/mogaze_head_multi_modal_gaze/t_his{t_his:d}_{margin_f:d}_thre{thre_his:.3f}_t_pred{t_pre:d}_thre{thre_pred:.3f}_index_sub{args.sub}.npz',
        data_multimodal=data_multimodal)