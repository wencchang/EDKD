import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1
        # self.epoch_class_probabilities = []

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict, log_csv):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)
        ################################
        log_csv.append(log_dict.items())
        ################################

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        #####################
        log_csv = []
        log_p_s_csv = []
        rows = 240
        columns = 100
        log_p_t_csv = [[0 for _ in range(columns)] for _ in range(rows)]
        log_beta_csv = []
        #####################
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            rows = 100
            columns = 500
            log_p_t_target = [[0 for _ in range(columns)] for _ in range(rows)]
            avg_p_t_class = []
            # self.train_epoch(epoch, log_csv, log_p_s_csv, log_p_t_target, log_beta_csv, self.epoch_class_probabilities)
            self.train_epoch(epoch, log_csv, log_p_s_csv, log_p_t_target, log_beta_csv)

            # print(log_p_t_target[0])
            for i in range(0, len(log_p_t_target)) :      
                for j in range(0, len(log_p_t_target[i])):   
                    a =+ log_p_t_target[i][j] 
                avg_p_t_class.append(a)
                # print(a)
            for i in range(0, len(avg_p_t_class)) :
                # print(len(avg_p_t_class)) 
                log_p_t_csv[epoch-1][i] = avg_p_t_class[i]
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))
        # #########################################
        # log_csv.append(
        #     {
        #         'epoch' : 'best_acc',
        #         "train_acc": float(self.best_acc),
        #     }
        # )
        # df = pd.DataFrame(log_csv)
        # df.to_csv(os.path.join(self.log_path, 'worklog.csv'))
        # df_ps = pd.DataFrame(log_p_s_csv)
        # df_pt = pd.DataFrame(log_p_t_csv)
        # df_beta = pd.DataFrame(log_beta_csv)
        # df_ps.to_csv(os.path.join(self.log_path, 'p_s_log.csv'))
        # df_pt.to_csv(os.path.join(self.log_path, 'p_t_log.csv'))
        # df_beta.to_csv(os.path.join(self.log_path, 'beta_log.csv'))
        # df_probs = pd.DataFrame(self.epoch_class_probabilities)
        # df_probs.to_csv(os.path.join(self.log_path, 'class_probabilities.csv'))
        #########################################
            
    # def train_epoch(self, epoch, log_csv, log_p_s_csv, log_p_t_target, log_beta_csv, epoch_class_probabilities):
    def train_epoch(self, epoch, log_csv, log_p_s_csv, log_p_t_target, log_beta_csv):

        # class_probabilities = {i: [] for i in range(100)}
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))
        
        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            # self.train_iter( data, epoch, train_meters, log_p_s_csv, log_p_t_csv, log_beta_csv, class_probabilities)
            # msg = self.train_iter( data, epoch, train_meters, log_p_s_csv, log_p_t_target, log_beta_csv, class_probabilities)
            msg = self.train_iter( data, epoch, train_meters, log_p_s_csv, log_p_t_target, log_beta_csv)

            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()
        # average_probabilities = []
        # for class_id in range(100):
        #     if class_probabilities[class_id]:
        #         avg_prob = np.mean(class_probabilities[class_id], axis=0)
        #         average_probabilities.append(avg_prob)
        #     else:
        #         average_probabilities.append(np.zeros((500,)))
        # epoch_class_probabilities.append(average_probabilities)

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)
        print('train_loss : ', train_meters["losses"].avg)
        print('test_loss : ', test_loss)
        # log
        log_dict = OrderedDict(
            {
                # 'epoch' : int(epoch),
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        self.log(lr, epoch, log_dict, log_csv)
        
        # saving checkpoint
        state = {

            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    # def train_iter(self, data, epoch, train_meters, log_p_s_csv, log_p_t_target, log_beta_csv, class_probabilities):
    def train_iter(self, data, epoch, train_meters, log_p_s_csv, log_p_t_target, log_beta_csv):

        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict, p_s, p_t, beta, target = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        
        # if epoch % 1 == 0:
        #     log_p_s_csv.append(
        #          {
        #         'epoch' : epoch,
        #         "log": p_s,
        #     }
        #     ) 
            
        #     for i in range(0, len(target)):
        #         log_p_t_target[target[i]-1].append(p_t[i].item())
                
        #     log_beta_csv.append(
        #         {
        #         'epoch' : epoch,
        #         "log": beta,
        #     }
        #     )
        return msg


class CRDTrainer(BaseTrainer):
    # def train_iter(self, data, epoch, train_meters, class_probabilities):
    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)
        # softmax = nn.Softmax(dim=1)
        # probabilities = softmax(preds).cpu().detach().numpy()
        # for idx, class_id in enumerate(target.cpu().numpy()):
        #     class_probabilities[class_id].append(probabilities[idx])

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        # self.optimizer.step((1 - epoch / 240.))
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
