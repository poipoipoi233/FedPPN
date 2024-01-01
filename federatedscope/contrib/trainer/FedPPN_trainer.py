from federatedscope.register import register_trainer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
import torch
import torch.nn as nn
import logging
import numpy as np
from collections import defaultdict
from federatedscope.contrib.model.label_prop import LabelPropagation
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
note: only support node-level task
"""

# Build your trainer here.
class FedPPN_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedPPN_Trainer, self).__init__(model, data, device, config,
                                             only_for_eval, monitor)

        self.task = config.MHFL.task
        self.num_classes = config.model.num_classes
        #LP: label propagation
        self.ctx.PPN = LabelPropagation(num_layers=config.FedPPN.LP_layer, alpha=config.FedPPN.LP_alpha)

        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.temperature = config.FedPPN.temperature

        self.register_our_hook()

        self.weight_private = 0.5

    def _hook_on_batch_forward(self, ctx):
        data = ctx.data_batch.to(ctx.device)
        split_mask = data[f'{ctx.cur_split}_mask']  # train_mask,val_mask,test_mask

        pred_all, emb_all = ctx.model(data)  # predict result and node embedding
        pred, emb = pred_all[split_mask], emb_all[split_mask]  # mask the result and node embedding
        labels = data.y[split_mask]  # mask the label

        loss_ce = ctx.criterion(pred, labels)  # cross-entropy loss

        if len(ctx.global_protos) == 0:
            loss = loss_ce
            prototype_pred = torch.zeros_like(pred) # for avoid error
            ensemble_pred =torch.zeros_like(pred) # same as above
        else:
            global_protos = torch.stack(list(ctx.global_protos.values()))  # (num_class, feature_dim)
            init_proto_label = initialize_prototype_label(data['train_mask'], global_protos, emb_all.detach(), data.y)

            prototype_emb = ctx.PPN(y=init_proto_label, edge_index=data.edge_index, train_mask=data['train_mask'])
            prototype_pred = ctx.model.FC(prototype_emb)[split_mask]

            ensemble_pred = (self.weight_private * pred + (1 - self.weight_private) * prototype_pred)
            # KD_loss = self.KL_Loss(self.LogSoftmax(pred / self.temperature),
            #                        self.Softmax(PL_pred.detach() / self.temperature))
            loss_PPN = ctx.criterion(prototype_pred, labels)
            loss_ensemble = ctx.criterion(ensemble_pred, labels)




            loss = loss_ce + loss_PPN + loss_ensemble

            # for ablation study
            # loss = loss_PPN + loss_ensemble # w/o CE
            # loss = loss_ce + loss_ensemble # w/o PPN
            # loss = loss_ce + loss_PPN # w/o ensemble
            # loss = loss_ce # only CE

            logger.info(
                f'client#{self.ctx.client_ID}'
                f'\t {ctx.cur_split}'
                f'\t round:{ctx.cur_state}'
                f'\t loss_CE:{loss_ce}'
                f'\t loos_PPN:{loss_PPN}'
                f'\t loss_ensemble:{loss_ensemble}'
                f'\t total_loss:{loss}'
            )

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(emb.detach().cpu())
        ctx.global_ys_prob.append(prototype_pred.detach().cpu())
        ctx.ensemble_ys_prob.append(ensemble_pred.detach().cpu())
        if len(ctx.global_protos)!=0 :
            ctx.PPN_node_emb_all = prototype_emb.detach().clone()  # 基于原型传播模块算出来的node embeddings
        ####

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        # ctx.optimizer_learned_weight_for_inference.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            # torch.nn.utils.clip_grad_norm_(ctx.weight_private,
            #                                ctx.grad_clip)

        ctx.optimizer.step()
        # ctx.optimizer_learned_weight_for_inference.step()
        # logger.info(f'当前weight:{ctx.weight_private}')
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def register_our_hook(self):
        # 训练结束聚合本地原型
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto, "on_fit_end")

        # 定义/初始化要用到的中间变量
        self.register_hook_in_train(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_variable_definition, "on_epoch_start")

        # 在client每次本地训练之前调用，用来初始化ctx.global_ys_prob；这个变量用于保存global_model的输出结果
        self.register_hook_in_train(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_clean, trigger='on_fit_start', insert_pos=-1)

    def _hook_on_epoch_start_for_variable_definition(self, ctx):
        ctx.agg_protos_label = CtxVar(dict(), LIFECYCLE.ROUTINE)
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.new_data = None

    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = defaultdict(list)
        agg_local_protos = dict()

        ctx.train_loader.reset()

        for batch_idx in range(ctx.num_train_batch):
            batch = next(ctx.train_loader)
            if self.task == "node":
                batch.to(ctx.device)
                split_mask = '{}_mask'.format(ctx.cur_split)
                labels = batch.y[batch[split_mask]]
                _, reps_all = ctx.model(batch)
                reps = reps_all[batch[split_mask]]
            else:
                images, labels = [_.to(ctx.device) for _ in batch]
                _, reps = ctx.model(images)

            owned_classes = labels.unique()
            for cls in owned_classes:
                filted_reps = reps[labels == cls].detach()
                reps_dict[cls.item()].append(filted_reps)

        for cls, protos in reps_dict.items():
            mean_proto = torch.cat(protos).mean(dim=0)
            agg_local_protos[cls] = mean_proto

        ctx.agg_local_protos = agg_local_protos

        # t-she可视化用
        if ctx.cfg.vis_embedding:
            ctx.node_emb_all = reps_all.clone().detach()
            ctx.node_labels = batch.y.clone().detach()

    def _hook_on_fit_start_clean(self, ctx):
        ctx.global_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存global model的输出结果用以验证
        ctx.ensemble_ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 保存global model的输出结果用以验证

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        training_begin_time = datetime.datetime.now()
        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        training_end_time = datetime.datetime.now()
        training_time = training_end_time - training_begin_time
        self.ctx.monitor.track_training_time(training_time)  # 记录每次本地训练的训练时间

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos


def initialize_prototype_label(train_mask, global_protos, reps_all, labels_all):
    # labels_init = torch.ones_like(reps_all) / len(reps_all)  # (N_all, feature_dim)
    labels_init = torch.zeros_like(reps_all)  # (N_all, feature_dim)
    labels_init[train_mask] = global_protos[labels_all[train_mask]]  # (N_train, feature_dim)
    # labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init


def call_my_trainer(trainer_type):
    if trainer_type == 'fedppn_trainer':
        trainer_builder = FedPPN_Trainer
        return trainer_builder


register_trainer('fedppn_trainer', call_my_trainer)
