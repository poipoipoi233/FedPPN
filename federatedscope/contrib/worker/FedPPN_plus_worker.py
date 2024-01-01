from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
import torch
from torch_geometric.nn import knn_graph
from collections import Counter
import datetime

logger = logging.getLogger(__name__)


class FedPPN_Plus_Server(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FedPPN_Plus_Server, self).__init__(ID, state, config, data, model, client_num,
                                                 total_round_num, device, strategy, **kwargs)

    def check_and_move_on(self, check_eval_result=False, min_received_num=None):
        min_received_num = len(self.comm_manager.get_neighbors().keys())

        if check_eval_result and self._cfg.federate.mode.lower() == "standalone":
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                # update global protos
                #################################################################
                local_protos_dict = dict()
                msg_list = self.msg_buffer['train'][self.state]
                for key, values in msg_list.items():
                    local_protos_dict[key] = values[1]
                global_protos = self._proto_aggregation(local_protos_dict)
                #################################################################

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._broadcast_custom_message(msg_type='global_proto', content=global_protos)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def _proto_aggregation(self, local_protos_dict):
        agg_global_protos = dict()
        for client_id in local_protos_dict:
            local_protos = local_protos_dict[client_id]
            for label in local_protos.keys():
                if label in agg_global_protos:
                    agg_global_protos[label].append(local_protos[label])
                else:
                    agg_global_protos[label] = [local_protos[label]]

        for [label, proto_list] in agg_global_protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_global_protos[label] = proto / len(proto_list)
            else:
                agg_global_protos[label] = proto_list[0].data

        return agg_global_protos

    def eval(self):
        self._broadcast_custom_message(msg_type='evaluate', content=None, filter_unseen_clients=False)

    def _broadcast_custom_message(self, msg_type, content,
                                  sample_client_num=-1,
                                  filter_unseen_clients=True):
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=content))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class FedPPN_Plus_Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FedPPN_Plus_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                                 strategy, is_unseen_client, *args, **kwargs)
        self.trainer.ctx.global_protos = []
        self.trainer.ctx.client_ID = self.ID
        self.register_handlers('global_proto',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])

        # For visualization
        self.client_agg_proto = dict()
        self.client_node_emb_all = dict()
        self.client_node_labels = dict()
        self.glob_proto_on_client = dict()
        self.client_PL_node_emb_all = dict()
        self.client_learnded_weight = dict()

        self.learn_for_adaptability = self._cfg.FedPPN_Plus.learning_for_adaptability
        if self.learn_for_adaptability:
            self.set_adaptability_dataset()

        # create KNN-GRAPH
        if self._cfg.FedPPN_Plus.use_knn_graph:
            knn_dict=dict()
            knn_dict['train_knn'], knn_dict['val_knn'], knn_dict['test_knn'] = self.create_knn_graph(data,device)
            self.trainer.ctx.knn_graph=knn_dict

        # # 收集训练集的标签分布
        # class_num = config.model.num_classes
        # self.train_label_distribution = {i: 0 for i in range(class_num)}
        # train_mask = data['data'].train_mask
        # train_label = data['data'].y[train_mask]
        # train_label_distribution_new = [j.item() if isinstance(j, torch.Tensor) else j[1] for j in train_label]
        # train_label_distribution_new = dict(Counter(train_label_distribution_new))
        # self.train_label_distribution.update(train_label_distribution_new)

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        content = message.content

        if message.msg_type == 'global_proto':
            self.trainer.update(content)
            if self.learn_for_adaptability:
                self.trainer.learn_weight_for_inference()
        self.state = round
        self.trainer.ctx.cur_state = self.state
        sample_size, model_para, results, agg_protos = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True
        )

        logger.info(train_log_res)

        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                 save_file_name="")

        self.history_results = merge_dict_of_results(
            self.history_results, train_log_res['Results_raw']
        )

        if self._cfg.vis_embedding:
            self.glob_proto_on_client[round] = self.trainer.ctx.global_protos
            self.client_node_emb_all[round] = self.trainer.ctx.node_emb_all
            self.client_node_labels[round] = self.trainer.ctx.node_labels
            self.client_agg_proto[round] = agg_protos
            self.client_PL_node_emb_all[round] = self.trainer.ctx.PL_node_emb_all

        if self._cfg.FedPPN_Plus.show_learned_weight_curve:
            self.client_learnded_weight[round] = self.trainer.learned_weight_for_inference.data.item()

        if self._cfg.FedPPN_Plus.PPN_mode == 'laernable_PPN':
            print('client {0} learned alpha: {1}'.format(self.ID,
                                                         self.trainer.ctx.PPN.alpha.data.item()))

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos)
                    )
        )

    def set_adaptability_dataset(self):
        adaptability_ratio = self._cfg.fedapen.adaptability_ratio
        train_data = self.trainer.ctx.data.train_data[0]
        train_mask = train_data.train_mask

        num_train = train_mask.sum().item()
        num_adaptability = int(num_train * adaptability_ratio)

        new_num_train = num_train - num_adaptability
        train_node_indices = torch.nonzero(train_mask).squeeze()
        new_train_indices = train_node_indices[:new_num_train]
        # new_train_indices = train_node_indices
        adaptability_indices = train_node_indices[new_num_train:]

        train_mask.fill_(False)  #
        adaptability_mask = torch.zeros_like(train_mask)

        train_mask[new_train_indices] = True
        adaptability_mask[adaptability_indices] = True

        # train_data.train_mask = train_mask
        train_data.adaptability_mask = adaptability_mask

    def callback_funcs_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content, strict=True)
        if self._cfg.vis_embedding:
            folderPath = self._cfg.MHFL.emb_file_path
            torch.save(self.glob_proto_on_client, f'{folderPath}/global_protos_on_client_{self.ID}.pth')  # 全局原型
            torch.save(self.client_agg_proto, f'{folderPath}/agg_protos_on_client_{self.ID}.pth')  # 本地原型
            torch.save(self.client_node_emb_all,
                       f'{folderPath}/local_node_embdeddings_on_client_{self.ID}.pth')  # 每个节点的embedding
            torch.save(self.client_node_labels, f'{folderPath}/node_labels_on_client_{self.ID}.pth')  # 标签
            torch.save(self.data, f'{folderPath}/raw_data_on_client_{self.ID}.pth')  # 划分给这个client的pyg data
            torch.save(self.client_PL_node_emb_all, f'{folderPath}/PP_node_embeddings_on_client_{self.ID}.pth')

        # if self._cfg.FedPPN_Plus.show_learned_weight_curve:
        #     self.plot_learned_weight_curve()

        self._monitor.finish_fl()

    def create_knn_graph(self, data, device,k=6):
        train_data, val_data, test_data = data.train_data[0], data.val_data[0], data.test_data[0]
        train_knn = knn_graph(x=train_data.x, k=k).to(device)
        val_knn = knn_graph(x=val_data.x, k=k).to(device)
        test_knn = knn_graph(x=test_data.x, k=k).to(device)
        return train_knn, val_knn, test_knn
    def plot_learned_weight_curve(self):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()

        x = np.arange(0, self.state+1)
        y = np.array(list(self.client_learnded_weight.values()))
        ax.plot(x, y)
        ax.set_xlabel = 'epoch'
        ax.set_ylabel = 'learned_weight'
        ax.set_title(f'Client {self.ID}')
        plt.show()


def call_my_worker(method):
    if method == 'fedppn_plus':
        worker_builder = {'client': FedPPN_Plus_Client, 'server': FedPPN_Plus_Server}
        return worker_builder


register_worker('fedppn_plus', call_my_worker)
