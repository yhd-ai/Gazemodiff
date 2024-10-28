import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation, image_vis,endpose_vis
#import torch

def demo_visualize(mode, cfg, model,gaze_gcn, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    """if cfg.dataset != 'h36m' and mode != 'pred':
        raise NotImplementedError(f"sorry, {mode} is currently only available in h36m setting.")"""
    if mode == 'switch':
        for i in range(0, cfg.vis_switch_num):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg, mode='switch')
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col,
                             output=os.path.join(cfg.gif_dir, f'switch_{i}.gif'), mode=mode)

    elif mode == 'pred':
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        #print(action_list)
        humanmac_list = []
        for i in range(0, len(action_list)):
            if dataset['test'].get_seqlen(action_list[i],dataset_type=cfg.dataset) >=75:
                
                for j in  range(0,dataset['test'].get_seqlen(action_list[i],dataset_type=cfg.dataset)-75,15):
                    pose_gen = pose_generator(dataset['test'], model,gaze_gcn, diffusion, cfg,
                                        mode='pred', action=action_list[i], nrow=1,fr_start = j)
                    #print("generated_pose",pose_gen.shape)
                    suffix = action_list[i]
                    pose = render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col+2,
                                output=os.path.join(cfg.gif_dir, f'pred_{suffix}_{j}.gif'), mode=mode,dataset = cfg.dataset)
                    humanmac_list.append(pose)
        np.save('/projects/yan/humanmac/proposed_7_1_plotgaze.npy',humanmac_list)
    elif mode == 'endpose':
        """action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        #print(action_list)
        #humanmac_list = []
        humanmac = np.load('/projects/yan/humanmac/humanmac_1.npy')
        humanmac_proposed = np.load('/projects/yan/humanmac/humanmac_nogaze.npy')
        humanmac_proposed = humanmac_proposed[:,1:7,:,:,:]
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model,gaze_gcn, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=1)
            suffix = action_list[i]
            endpose_vis(dataset['test'].skeleton, pose_gen, humanmac_pose = humanmac[i],proposed_pose=humanmac_proposed[i],algos=['HumanMAC'], ncol=cfg.vis_col + 3,
                             output=os.path.join(cfg.gif_dir, f'compare_{suffix}.png'),num_compare=1)
            #humanmac_list.append(pose)
        #np.save('humanmac_nogaze.npy',humanmac_list)"""
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        #print(action_list)
        #humanmac_list = []
        #dlow = np.load('/projects/yan/DLow/dlow_nogaze_gimo.npy')
        #print(dlow.shape)
        humanmac = np.load('/projects/yan/humanmac/humanmac_gimo.npy')
        
        proposed_wogaze = np.load('/projects/yan/humanmac/proposed_wogaze_gimo_new.npy')
        
        proposed = np.load('/projects/yan/humanmac/proposed_gimo_plotgaze.npy')
        print(proposed_wogaze.shape)
        print(humanmac.shape)
        print(proposed.shape)
        humanmac = humanmac[:,1:,:,:]
        humanmac = np.concatenate([humanmac,humanmac[:,:,:,0:1,:]],axis=-2)
        proposed_wogaze = proposed_wogaze[:,1:,:,:]
        proposed_wogaze = np.concatenate([proposed_wogaze,proposed_wogaze[:,:,:,0:1,:]],axis=-2)
        proposed = proposed[:,1:,:,:]
        #print(proposed.shape)
        print(proposed_wogaze.shape)
        print(humanmac.shape)
        print(proposed.shape)
        #print(1)
        for i in range(0, len(humanmac)):
            #print(2)
          
            
            
            endpose_vis(dataset['test'].skeleton, humanmac[i], proposed_wogaze_pose = proposed_wogaze[i] ,proposed_pose=proposed[i], ncol=cfg.vis_col + 3,
                             output=os.path.join(cfg.gif_dir, f'endpose_{i}.png'),num_compare=2)
            #humanmac_list.append(pose)
        #np.save('humanmac_nogaze.npy',humanmac_list)
    elif mode == 'compare': 
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        #print(action_list)
        humanmac = np.load('/projects/yan/humanmac/humanmac_1.npy')
        humanmac_proposed = np.load('/projects/yan/humanmac/humanmac_nogaze.npy')
        #print(humanmac.shape)
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model,gaze_gcn, diffusion, cfg,
                                      mode='pred', action=action_list[i],  nrow=1)
            suffix = action_list[i]
            image_vis(dataset['test'].skeleton, pose_gen, humanmac_pose = humanmac[i],proposed_pose=humanmac_proposed[i],algos=['HumanMAC'], ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'compare_{suffix}.png'),num_compare=3)
        
    elif mode == 'control':
        # draw part-body controllable results
        fix_name = ['right_leg', 'left_leg', 'torso', 'left_arm', 'right_arm', 'fix_lower', 'fix_upper']
        for i in range(0, 7):
            mode_fix = 'fix' + '_' + str(i)
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode=mode_fix, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, fix_name[i] + '.gif'), mode=mode, fix_index=i)
    elif mode == 'zero_shot':
        amass_data = np.squeeze(np.load('./data/amass_retargeted.npy'))
        for i in range(0, 15):
            pose_gen = pose_generator(amass_data, model, diffusion, cfg, mode=mode, nrow=cfg.vis_row)
            render_animation(dataset['test'].skeleton, pose_gen, ['HumanMAC'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'zero_shot_{str(i)}.gif'), mode=mode)
    else:
        raise
