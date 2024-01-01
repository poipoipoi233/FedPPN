set -e
cd ../../../ #到federatedscope目录

dataset=cora
# common hyperparameters
gpu=0
client_num=3
local_update_step=16
optimizer='SGD'
seed=(0 1 2)
lrs=0.25
total_round=200
patience=50
momentum=0.9
freq=1
global_eval=False
local_eval_whole_test_dataset=True

# FedProto-specific parameters
proto_weight=1.0

# Analytical experiments
only_CE_loss=True
use_similarity_for_inference=True
vis_embedding=False
plot_acc_curve=False

result_floder=model_heterogeneity/result/FedProto_Analytical_Experiments
main_cfg="model_heterogeneity/SFL_methods/FedProto/FedProto_on_"$dataset".yaml"
client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
exp_name="Analytical_experiments_2_for_"$dataset"_"$client_num"_clients"


# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${client_num} \
    federate.make_global_eval ${global_eval} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
    seed ${1} \
    train.local_update_steps ${local_update_step} \
    train.optimizer.lr ${lrs} \
    federate.total_round_num ${total_round} \
    train.optimizer.type ${optimizer} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    eval.freq ${freq} \
    fedproto.proto_weight ${proto_weight} \
    fedproto.only_CE_loss ${only_CE_loss} \
    fedproto.use_similarity_for_inference ${use_similarity_for_inference} \
    vis_embedding ${vis_embedding}
}

for s in "${seed[@]}"; do
  train_model "$s"
done
