from federatedscope.register import register_trainer
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.context import CtxVar
import torch
import torch.nn as nn
import numpy as np
import logging
import copy
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FPL_Node_Trainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FPL_Node_Trainer, self).__init__(model, data, device, config,
                                               only_for_eval, monitor)
        self.temperature = config.fpl.temperature
        self.device=device
        self.register_hook_in_train(self._hook_on_fit_end_agg_proto,
                                    "on_fit_end")
        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                   "on_epoch_start")

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        labels = batch.y[batch['{}_mask'.format(ctx.cur_split)]]

        #####################################################################################
        if len(ctx.global_protos) != 0:
            all_global_protos_keys = np.array(list(ctx.global_protos.keys()))
            all_global_protos = []
            mean_global_protos = []
            for protos_key in all_global_protos_keys:
                temp_global_protos = ctx.global_protos[protos_key]
                temp_global_protos = torch.cat(temp_global_protos, dim=0).to(ctx.device)
                all_global_protos.append(temp_global_protos.cpu())
                mean_global_protos.append(torch.mean(temp_global_protos, dim=0).cpu())
            all_global_protos = [item.detach() for item in all_global_protos]
            mean_global_protos = [item.detach() for item in mean_global_protos]

        pred_all, embedding_all = ctx.model(batch)
        emb = embedding_all[batch['{}_mask'.format(ctx.cur_split)]]
        pred = pred_all[batch['{}_mask'.format(ctx.cur_split)]]

        cls_loss = ctx.criterion(pred, labels)

        if len(ctx.global_protos) == 0:
            loss_InfoNCE = 0 * cls_loss
        else:
            i = 0
            loss_InfoNCE = None
            for label in labels:
                if label.item() in ctx.global_protos.keys():
                    emb_now = emb[i].unsqueeze(0)
                    loss_instance = self.hierarchical_info_loss(emb_now, label, all_global_protos, mean_global_protos,
                                                                all_global_protos_keys, ctx.device)
                    if loss_InfoNCE is None:
                        loss_InfoNCE = loss_instance
                    else:
                        loss_InfoNCE += loss_instance
                i += 1
            loss_InfoNCE = loss_InfoNCE / i
        loss_InfoNCE = loss_InfoNCE  # TODO 这行是否多余？

        loss = cls_loss + loss_InfoNCE * 1.0  # TODO:增加超参数

        logger.info(f'cls_loss:{cls_loss},loss_InfoNCE:{loss_InfoNCE}')

        if ctx.cfg.vis_embedding:
            ctx.protos_vis = embedding_all.clone().detach()
            ctx.y_for_vis = batch.y.clone().detach()
        ######################################################################################
        ctx.batch_size = torch.sum(ctx.data_batch['{}_mask'.format(ctx.cur_split)]).item()
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)

    def _hook_on_fit_end_agg_proto(self, ctx):
        self.get_aggprotos()
        protos = ctx.agg_protos_label
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0].mean(dim=0)

        setattr(ctx, "agg_protos", protos)

    def get_distance_tensor(self, protos, global_protos):
        num_nodes = len(protos)
        protos_expanded = protos.unsqueeze(1)
        global_protos_stack = torch.stack(list(global_protos.values()))  # 将global_protos的值合并为[2, 64]的张量
        distances = F.pairwise_distance(protos_expanded, global_protos_stack.unsqueeze(0).expand(num_nodes, -1, -1))
        return distances

    def _hook_on_epoch_start_for_proto(self, ctx):
        """
            开始每个local epoch的训练之前，初始化agg_protos_label为空字典
        """
        agg_protos_label = {}
        ctx.agg_protos_label = CtxVar(agg_protos_label, LIFECYCLE.ROUTINE)

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_protos

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys, device):
        f_pos = np.array(all_f,dtype=object)[all_global_protos_keys == label.item()][0].to(device)
        f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(device)
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = np.array(mean_f)[all_global_protos_keys == label.item()][0].to(device)
        mean_f_pos = mean_f_pos.view(1, -1)
        # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
        # mean_f_neg = mean_f_neg.view(9, -1)

        loss_mse = nn.MSELoss()
        cu_info_loss = loss_mse(f_now, mean_f_pos)

        hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss
    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.temperature

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss
    @torch.no_grad()
    def get_aggprotos(self):
        data_batch = next(self.ctx.get("{}_loader".format('train')))
        batch = data_batch.to(self.ctx.device)
        labels = batch.y[batch['{}_mask'.format('train')]]
        #####################################################################################
        pred_all, embedding_all = self.ctx.model(batch)
        embeddings = embedding_all[batch['{}_mask'.format('train')]]

        for label in labels.unique():
            self.ctx.agg_protos_label[label.item()] = [embeddings[labels == label.item()]]


def call_my_torch_trainer(trainer_type):
    if trainer_type == 'fpl_node_trainer':
        trainer_builder = FPL_Node_Trainer
        return trainer_builder


register_trainer('FPL_Trainer', call_my_torch_trainer)
