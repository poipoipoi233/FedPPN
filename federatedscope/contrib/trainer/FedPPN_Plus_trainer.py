from federatedscope.register import register_trainer
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
import torch
import torch.nn as nn
import logging
import copy
import numpy as np
from collections import defaultdict
# from federatedscope.contrib.model.label_prop import LabelPropagation
from federatedscope.contrib.model.PPN import PPN, Learnable_PPN
import datetime
from torch_geometric.nn import LabelPropagation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
2023.12.12
重构了PPN函数
尝试两个改进：
1. 可学习的 ensemble weight；参考FedAPEN  ---看起来没啥用
2. 可学习的原型传播模块--不能给每个node打分，因为test graph 和 train graph不一样;

2023.12.15
1.尝试增加FedAPEN的蒸馏损失:not work ---2023.12.18



"""

KL_Loss = nn.KLDivLoss(reduction='batchmean')
LogSoftmax = nn.LogSoftmax(dim=1)
Softmax = nn.Softmax(dim=1)


# Build your trainer here.
class FedPPN_Plus_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FedPPN_Plus_Trainer, self).__init__(model, data, device, config,
                                                  only_for_eval, monitor)

        self.task = config.MHFL.task
        self.num_classes = config.model.num_classes

        self.epoch_for_learn_weight = config.FedPPN_Plus.epoch_for_learn_weight
        self.learn_for_adaptability = config.FedPPN_Plus.learning_for_adaptability
        # self.learned_weight_dict = dict()

        # LP: label propagation
        PPN_mode = config.FedPPN_Plus.PPN_mode
        self.PPN_mode = PPN_mode
        if PPN_mode == 'laernable_PPN':
            self.ctx.PPN = Learnable_PPN(num_layers=config.FedPPN.LP_layer)
            self.ctx.optimizer_learned_alpha = torch.optim.Adam(self.ctx.PPN.parameters(), lr=1e-3)
        elif PPN_mode == 'PPN':
            self.ctx.PPN = PPN(num_layers=config.FedPPN.LP_layer, alpha=config.FedPPN.LP_alpha)
        elif PPN_mode == 'LP':
            self.ctx.PPN = LabelPropagation(num_layers=config.FedPPN.LP_layer, alpha=config.FedPPN.LP_alpha)
        else:
            raise ValueError(
                f"PPN mode {PPN_mode} is not defined."
            )
        self.KL_Loss = nn.KLDivLoss(reduction='batchmean')
        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.temperature = config.FedPPN.temperature

        self.learned_weight_for_inference = torch.tensor([0.5], requires_grad=True, device=device)
        self.optimizer_learned_weight_for_inference = torch.optim.SGD([self.learned_weight_for_inference],
                                                                      lr=config.FedPPN_Plus.adaptive_weight_lr)

        self.weight_private = 0.5

        self.register_our_hook()
        self.loss_mse = nn.MSELoss()

    def _hook_on_batch_forward(self, ctx):
        data = ctx.data_batch.to(ctx.device)
        split_mask = data[f'{ctx.cur_split}_mask']  # train_mask,val_mask,test_mask

        pred_all, emb_all = ctx.model(data)  # predict result and node embedding
        pred, emb = pred_all[split_mask], emb_all[split_mask]  # mask the result and node embedding

        labels = data.y[split_mask]  # mask the label

        loss_ce = ctx.criterion(pred, labels)  # cross-entropy loss

        if self.learn_for_adaptability:
            private_weight = self.learned_weight_for_inference
        else:
            private_weight = 0.5

        if len(ctx.global_protos) == 0:
            loss = loss_ce
            prototype_pred = torch.zeros_like(pred)  # for avoid error
            ensemble_pred = torch.zeros_like(pred)  # same as above
        else:
            global_protos = torch.stack(list(ctx.global_protos.values()))  # (num_class, feature_dim)
            L2_prediction_all = torch.norm(emb_all.unsqueeze(1)-global_protos.unsqueeze(0),dim=2)

            if self.PPN_mode == 'LP':
                prototype_pred = ctx.PPN(data.y, data.edge_index, mask=data['train_mask'])[split_mask]
            else:
                prototype_emb, num_not_updated, not_updated_indices = ctx.PPN(data['train_mask'], global_protos, data.y,
                                                                              data.edge_index)  # (N,D)
                prototype_pred_all = ctx.model.FC(prototype_emb)  # (N, D)
                if self.cfg.FedPPN_Plus.use_knn_graph:
                    prototype_knn_emb, _, _ = ctx.PPN(data['train_mask'], global_protos, data.y,
                                                      ctx.knn_graph[f'{ctx.cur_split}_knn'])  # (N,D)
                    prototype_knn_pred_all = ctx.model.FC(prototype_knn_emb)

                # prototype_pred_all[not_updated_indices]=prototype_knn_pred_all[not_updated_indices]
                # logger.info(f'Client:{ctx.client_ID} {ctx.cur_split}: nodes not been updated {num_not_updated}')

            prototype_pred_all[not_updated_indices] = pred_all[not_updated_indices]  # 没有被PPN更新的节点，其预测完全基于本地GNN的预测
            # ensemble_pred_all = private_weight * pred_all + (1 - private_weight) * prototype_pred_all +L2_prediction_all  # (N,D)

            ensemble_pred_all = pred_all +prototype_pred_all + L2_prediction_all  # (N,D)

            prototype_pred = prototype_pred_all[split_mask]
            ensemble_pred = ensemble_pred_all[split_mask]

            loss_PPN = ctx.criterion(prototype_pred, labels)
            loss_ensemble = ctx.criterion(ensemble_pred, labels)

            # MD Loss
            kl_private = KL_Loss(LogSoftmax(pred_all), Softmax(prototype_pred_all.detach()))

            # edge-free loss
            # L2_loss = ctx.criterion(L2_prediction,labels)


            # kl_shared = KL_Loss(LogSoftmax(prototype_pred), Softmax(pred.detach()))
            # loss = loss_ce + loss_PPN + loss_ensemble+ kl_private+kl_shared

            # for ablation study
            # loss = loss_PPN + loss_ensemble # w/o CE
            # loss = loss_ce + loss_ensemble # w/o PPN
            # loss = loss_ce + loss_PPN # w/o ensemble
            # loss = loss_ce # only CE

            loss = loss_ce + loss_PPN + loss_ensemble# +L2_loss
            # logger.info(
            #     f'client#{ctx.client_ID}'
            #     f'\t {ctx.cur_split}'
            #     f'\t round:{ctx.cur_state}'
            #     f'\t loss_CE:{loss_ce}'
            #     f'\t loos_PPN:{loss_PPN}'
            #     f'\t loss_ensemble:{loss_ensemble}'
            #     f'\t total_loss:{loss}'
            # )

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        ####
        ctx.ys_feature.append(emb.detach().cpu())
        ctx.global_ys_prob.append(prototype_pred.detach().cpu())
        ctx.ensemble_ys_prob.append(ensemble_pred.detach().cpu())
        # if len(ctx.global_protos) != 0 :
        #     ctx.PPN_node_emb_all = prototype_emb.detach().clone()  # 基于原型传播模块算出来的node embeddings
        ####

    def learn_weight_for_inference(self):
        """
        Learning for Adaptability
        """
        ctx = self.ctx
        ctx.model.to(ctx.device)
        ctx.model.eval()

        data = ctx.data.train_data[0].to(ctx.device)
        split_mask = data['adaptability_mask']
        target = data.y[split_mask]
        global_protos = torch.stack(list(ctx.global_protos.values())).to(ctx.device)
        for _ in range(self.epoch_for_learn_weight):
            output_private, _ = ctx.model(data)
            output_private = output_private[split_mask].detach()  # TODO: split在detach()前后是否会影响结果?
            PPN_reps = ctx.PPN(data['train_mask'], global_protos, data.y, data.edge_index)[split_mask]
            output_PPN = ctx.model.FC(PPN_reps).detach()
            ensemble_output = self.learned_weight_for_inference * output_private + (
                    1 - self.learned_weight_for_inference) * output_PPN
            loss = ctx.criterion(ensemble_output, target)  # CE_LOSS
            loss.backward()
            self.optimizer_learned_weight_for_inference.step()
            torch.clip_(self.learned_weight_for_inference.data, 0.0, 1.0)

        # self.learned_weight_dict[ctx.cur_state] = self.learned_weight_for_inference.cpu().data.item()
        ctx.model = ctx.model.cpu()
        # ctx.global_model = ctx.global_model.cpu()

        print('client {0} learned weight for inference: {1}'.format(self.ctx.client_ID,
                                                                    self.learned_weight_for_inference.data.item()))

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        # ctx.optimizer_learned_weight_for_inference.zero_grad()
        if self.PPN_mode == 'laernable_PPN':
            ctx.optimizer_learned_alpha.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
            torch.nn.utils.clip_grad_norm_(ctx.PPN.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        if self.PPN_mode == 'laernable_PPN':
            ctx.optimizer_learned_alpha.step()
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
    if trainer_type == 'fedppn_plus_trainer':
        trainer_builder = FedPPN_Plus_Trainer
        return trainer_builder


register_trainer('fedppn_plus_trainer', call_my_trainer)
