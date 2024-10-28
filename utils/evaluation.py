import csv

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, multimodal_dict, model,gaze_gcn, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    
    def get_subprediction(data, model_select,gaze_gcn):
        #print(data.shape)
        b,s,j,f = data.shape
        traj_np = data[..., :, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]
        

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        #print(traj_dct_cond.shape)
        b= traj_dct_cond.shape[0]
        traj_dct_cond = gaze_gcn(traj_dct_cond.reshape(b,f,j,cfg.n_pre)).reshape(b,cfg.n_pre,-1)
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est
    
    def get_prediction(data, model_select, gaze_gcn):
        num_samples = data.shape[0]
        batch_size = 32
        num_batches = int(np.ceil(num_samples / batch_size))
        traj_est_batches = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_data = data[start_idx:end_idx]

            traj_est_batch = get_subprediction(batch_data, model_select, gaze_gcn)
            #print(traj_est_batch.shape)
            traj_est_batches.append(traj_est_batch.squeeze(0))
            

        traj_est = np.concatenate(traj_est_batches, axis=0)
        return traj_est[None,:,:,:]

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    K = 50
    pred = []
    error_list = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model,gaze_gcn)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
    
            pred = pred[:, :, cfg.t_his:, :]
          
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            
            
           
            for j in range(0, num_samples):
                
                if cfg.dataset == 'mogaze' or cfg.dataset == 'mogaze_withcontext':
                    pred_j = pred[:, j, :, :63]
                    gt_group_j = gt_group[j][np.newaxis, ...][:,:,:63]
                    traj_gt_arr_j = traj_gt_arr[j][:,:,:63]
                else:
                    pred_j = pred[:, j, :, :69]
                    gt_group_j = gt_group[j][np.newaxis, ...][:,:,:69]
                    traj_gt_arr_j = traj_gt_arr[j][:,:,:69]
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred_j,
                                                                        gt_group_j,
                                                                        traj_gt_arr_j)
                error_list.append(ade.cpu())
                
                stats_meter['APD']['HumanMAC'].update(apd)
                stats_meter['ADE']['HumanMAC'].update(ade)
                stats_meter['FDE']['HumanMAC'].update(fde)
                stats_meter['MMADE']['HumanMAC'].update(mmade)
                stats_meter['MMFDE']['HumanMAC'].update(mmfde)
            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    error_array = np.array(error_list)
    
    #np.save('error_mogaze_humanmac+gaze.npy',error_array)
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['HumanMAC'] = new_meter['HumanMAC'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
