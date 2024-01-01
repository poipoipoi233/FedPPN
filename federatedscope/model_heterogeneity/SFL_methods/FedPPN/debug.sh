cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml \
--client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml \
data.local_eval_whole_test_dataset True \
federate.client_num 3

#
#--cfg
#model_heterogeneity/SFL_methods/POIV5/POIV5_on_pubmed.yaml
#--client_cfg
#model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml
#data.local_eval_whole_test_dataset
#True
#federate.client_num
#5