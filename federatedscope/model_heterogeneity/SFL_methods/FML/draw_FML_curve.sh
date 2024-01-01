set -e
cd ../../../ #到federatedscope目录

# basic configuration
gpu=0
method=FML
result_folder_name=Convergence_Curve_1110
global_eval=False
local_eval_whole_test_dataset=True


script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}

# common hyperparameters
dataset=('cora' 'citeseer' 'pubmed')
total_client=(3 5)
optimizer='SGD'
seed=(0)
total_round=200
patience=200
momentum=0.9
freq=1

# Local-specific parameters
# pass
save_history_result_per_client=True
save_history_result_avg=True
# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${1} \
    federate.make_global_eval ${global_eval} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
    seed ${2} \
    federate.total_round_num ${total_round} \
    train.optimizer.type ${optimizer} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    eval.freq ${freq} \
    save_history_result_per_client ${save_history_result_per_client} \
    save_history_result_avg ${save_history_result_avg}
}

# Loop over parameters for HPO
cnt=0
pass_round=0
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for s in "${seed[@]}"; do
      let cnt+=1
      if [ "$cnt" -lt $pass_round ]; then
        continue
      fi
      main_cfg=$script_floder"/"$method"_on_"$data".yaml"
      client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
      exp_name=$method"_"$data"_"$client_num"_clients"
      train_model "$client_num" "$s"
    done
  done
done