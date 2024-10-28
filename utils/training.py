import copy
import time

from torch import optim, nn
import torch as th
from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
                 GCN,
                 dataset,
                 cfg,
                 multimodal_dict,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None

        self.model = model
        self.gaze_gcn = GCN
        self.diffusion = diffusion
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0

        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None


    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.optimizer_gcn = optim.Adam(self.gaze_gcn.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()
        self.gaze_gcn.train()
        #print(1)
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for traj_np in self.generator_train:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                #print("traj",traj_np.shape)
                b, s, j, f = traj_np.shape
                traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                #print("traj",traj_np.shape)
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
            
          
                #print('traj_pad',traj_pad.shape)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                #print("traj",traj_dct.shape)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                #print("traj",traj_dct_mod.shape)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None
            traj_dct_mod = self.gaze_gcn(traj_dct_mod.reshape(b,f,j,self.cfg.n_pre)).reshape(b,self.cfg.n_pre,-1)
            # train

            mod = {'y':traj_dct_mod,'uncond':False}
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(traj_dct, t)
            #print('xtshape',x_t.shape)
            #print('trajshape',traj_dct_mod.shape)
            predicted_noise = self.model(x_t, t, mod = mod)
            #print('pre',predicted_noise.shape)
            #print(noise.shape)
            """diff = predicted_noise - predicted_noise
            loss_batch = torch.linalg.norm(diff, axis=-1)
            min_loss, _ = loss_batch.min(dim=1)#
            loss = min_loss.sum()"""
            #print(loss.shape)
            
            
            """diff = predicted_noise -  noise
            loss_batch = th.linalg.norm(diff, axis=-1)
            min_loss, _ = loss_batch.min(dim=1)#
            loss = min_loss.sum()"""
            loss = self.criterion(predicted_noise, noise)
            #print(loss)

            #xout = self.crossattentionDecoder()
            """layer_params = self.gaze_gcn.named_parameters()
            for name,param in layer_params:
                print(name,param.sum())"""

            self.optimizer.zero_grad()
            self.optimizer_gcn.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_gcn.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))
        #if self.iter % self.cfg.save_gif_interval == 0:
            #pose_gen = pose_generator(self.dataset['train'], self.model,self.gaze_gcn, self.diffusion, self.cfg, mode='gif')
            #render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
            #                 output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        self.gaze_gcn.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np in self.generator_val:
            with torch.no_grad():
                b, s, j, f = traj_np.shape
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                traj_np = traj_np[..., :, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                #traj = self.crossattentionEncoder(traj)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad,
                                        self.cfg.zero_index)  #
                
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]

                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None
                mod = {'y':traj_dct_mod,'uncond':False}
                traj_dct_mod = self.gaze_gcn(traj_dct_mod.reshape(b,f,j,self.cfg.n_pre)).reshape(b,self.cfg.n_pre,-1)

                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                predicted_noise = self.model(x_t, t, mod=mod)
                """diff = predicted_noise - predicted_noise
                loss_batch = torch.linalg.norm(diff, axis=-1)
                min_loss, _ = loss_batch.min(dim=1)#
                loss = min_loss.sum()"""
                loss = self.criterion(predicted_noise, noise)
                """diff = predicted_noise -  noise
                loss_batch = th.linalg.norm(diff, axis=-1)
                min_loss, _ = loss_batch.min(dim=1)#
                loss = min_loss.sum()"""

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)
            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))

        """if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model,self.gaze_gcn, self.diffusion, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model,self.gaze_gcn, self.diffusion, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))
        """
        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            if self.cfg.ema is True:
                compute_stats(self.diffusion, self.multimodal_dict, self.ema_model,self.gaze_gcn, self.logger, self.cfg)
            else:
                compute_stats(self.diffusion, self.multimodal_dict, self.model, self.gaze_gcn,self.logger, self.cfg)
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
                torch.save(self.gaze_gcn.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_gcn_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
                torch.save(self.gaze_gcn.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_gcn_{self.iter + 1}.pt"))
