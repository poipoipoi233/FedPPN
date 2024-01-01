from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.register import register_trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FCCL_Node_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FCCL_Node_Trainer, self).__init__(model, data, device, config,
                                                only_for_eval, monitor)
        self.criterionKL = nn.KLDivLoss(reduction='batchmean')  # 计算KL散度损失 指定损失减少方式为对所有样本损失平均

        self.register_hook_in_train(new_hook=self._hook_on_fit_start_setup,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_eval(new_hook=self._hook_on_fit_start_setup,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_train(new_hook=self._hook_on_fit_end_free_cuda,
                                    trigger="on_fit_end",
                                    insert_pos=-1)

        self.register_hook_in_eval(new_hook=self._hook_on_fit_end_free_cuda,
                                   trigger="on_fit_end",
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        split_mask = batch[f'{ctx.cur_split}_mask']
        labels = batch.y[split_mask]

        outputs = ctx.model(batch)[split_mask]  # 私有数据的Zi
        logsoft_outputs = F.log_softmax(outputs, dim=1)  # 进行对数softmax操作
        with torch.no_grad():  # 计算操作不会被记录梯度
            inter_soft_outpus = F.softmax(ctx.inter_model(batch)[split_mask], dim=1)
            pre_soft_outpus = F.softmax(ctx.pre_model(batch)[split_mask], dim=1)

        inter_loss = self.criterionKL(logsoft_outputs, inter_soft_outpus)  # 公式4 inter损失
        pre_loss = self.criterionKL(logsoft_outputs, pre_soft_outpus)  # 公式5 intra损失
        loss_hard = ctx.criterion(outputs, labels)  # 交叉熵损失

        loss = loss_hard + (inter_loss + pre_loss) * self.ctx.cfg.fccl.loss_dual_weight  # 公式7 总损失

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(outputs, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_fit_start_setup(self, ctx):
        # 在每次开始训练前 将inter_model和pre_model放置对应的device中,同时变换其train/eval状态
        device = ctx.device
        ctx.inter_model.to(device)
        ctx.pre_model.to(device)
        if ctx.cur_mode in [MODE.TRAIN]:
            ctx.inter_model.train()
            ctx.pre_model.train()
        elif ctx.cur_mode in [MODE.VAL, MODE.TEST]:
            ctx.inter_model.eval()
            ctx.pre_model.eval()
    def _hook_on_fit_end_free_cuda(self, ctx):
        #将inter_model和pre_model从cuda中释放
        ctx.inter_model.to(torch.device("cpu"))
        ctx.pre_model.to(torch.device("cpu"))


def call_my_trainer(trainer_type):
    if trainer_type == 'fccl_node_trainer':
        trainer_builder = FCCL_Node_Trainer
        return trainer_builder


register_trainer('fccl_node_trainer', call_my_trainer)
