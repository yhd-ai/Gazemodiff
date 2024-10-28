import argparse
import sys

from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())
from config import Config, update_config
import torch

from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='train', help='train / eval / pred / switch/ control/ zero_shot')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.4)
    parser.add_argument('--multimodal_th_high', type=float, default=0)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--save_gif_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./results/mogaze_withcontext_59/models/ckpt_ema_50.pt')
    parser.add_argument('--ckpt_gcn', type=str, default='./results/mogaze_withcontext_59/models/ckpt_gcn_50.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--cfg_id',  default='gimo_realcfg0.1_gazefeature_posefeaturenonoise1500nosiinghead')
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    dataset, dataset_multi_test = dataset_split(cfg)
    
    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion,gaze_gcn = create_model_and_diffusion(cfg)
    #print(model)
    #model = torch.nn.DataParallel(model)
    #gaze_gcn = torch.nn.DataParallel(gaze_gcn)
    if args.resume:
        ckpt = torch.load(args.ckpt)
        print()
        model.load_state_dict(ckpt)
        ckpt_gcn = torch.load(args.ckpt_gcn)
        gaze_gcn.load_state_dict(ckpt_gcn)
        


    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            GCN=gaze_gcn,
            dataset=dataset,
            cfg=cfg,
            multimodal_dict=multimodal_dict,
            logger=logger,
            tb_logger=tb_logger)
        #print(2)
        trainer.loop()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        ckpt_gcn = torch.load(args.ckpt_gcn)
        gaze_gcn.load_state_dict(ckpt_gcn)
        
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model,gaze_gcn, logger, cfg)
    else:
        ckpt = torch.load(args.ckpt)
        ckpt_gcn = torch.load(args.ckpt_gcn)
        model.load_state_dict(ckpt)
        gaze_gcn.load_state_dict(ckpt_gcn)
        demo_visualize(args.mode, cfg, model, gaze_gcn,diffusion, dataset)
