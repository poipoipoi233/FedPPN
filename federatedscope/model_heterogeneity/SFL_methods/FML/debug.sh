cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FML/FML_on_cora.yaml
--client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml
data.local_eval_whole_test_dataset True
