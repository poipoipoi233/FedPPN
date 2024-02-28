# Model-Heterogeneous Federated Graph Learning with Prototype Propagation
This is the official code of the paper: Model-Heterogeneous Federated Graph Learning with Prototype Propagation




## Acknowledgment

All code implementations are based on the FederatedScope V0.3.0: https://github.com/alibaba/FederatedScope 

We are grateful for their outstanding work.




## Models & Dataset

### Model setting

We consider three heterogeneous Graph Neural Network (GNN) backbones, i.e., GCN, GAT, and GPR-GNN . Each of the GNN backbones has very different message propagation mechanisms. For each client, we assign its local model by sampling from the three backbones. We further modify each local model's number of layers and hidden state dimensions to enhance heterogeneity.

For details of the model architecture, please refer to [model settings folder](federatedscope/model_heterogeneity/model_settings) and [model definition folder](federatedscope/gfl/model)


### Dataset

Following the work of FGSSL[[Huang *et* *al.,* 2023]](https://www.ijcai.org/proceedings/2023/426), we conduct experiments on three benchmark graph datasets: Cora, CiteSeer, and PubMed.



## Quickly Start

### Step 1. Install FederatedScope

Users need to clone the source code and install FederatedScope (we suggest python version >= 3.9).

- clone the source code

```python
git clone https://github.com/zza234s/FedPPN
cd FedPPN
```

- install the required packages:

```python
conda create -n fs python=3.9
conda activate fs

# install pytorch
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# install some extra dependencies
conda install -y pyg==2.0.4 -c pyg
conda install -y nltk
pip install rdkit
pip install ipdb
pip install kornia
pip install timm
pip install ogb
```


- Next, after the required packages is installed, you can install FederatedScope from `source`:

```python
pip install -e .[dev]
```


### Step 2. Run Algorithm

- Enter the "federatedscope" folder

```python
cd federatedscope
```

- Run the script (The main experiments):

```python
# python main.py --cfg ${main YAML file} --client_cfg ${model settings YAML file} federate.client_num ${total number of clients}

# Cora
# For 3 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml federate.client_num 3

# For 5 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

# For 10 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml federate.client_num 10

# Citeseer
# For 3 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml federate.client_num 3

# For 5 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

# For 10 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml federate.client_num 10

# PubMed
# For 3 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml federate.client_num 3

# For 5 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

# For 10 clients
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/10_Heterogeneous_GNNs.yaml federate.client_num 10

```



## Run Baselines 

Users can run our reproduced baseline methods in the same way as running the FedPPN, by replace ${main YAML file} and  ${model settings YAML file}.

-  Take running different methods on the Cora dataset with 5 clients as an example:

```python
#Local
python main.py --cfg model_heterogeneity/SFL_methods/Local/Local_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

#FML
python main.py --cfg model_heterogeneity/SFL_methods/FML/FML_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

#FedKD
python main.py --cfg model_heterogeneity/SFL_methods/FedKD/FedKD_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

#FedProto
python main.py --cfg model_heterogeneity/SFL_methods/FedProto/FedProto_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

#FedPCL
python main.py --cfg model_heterogeneity/SFL_methods/FedPCL/FedPCL_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5

#FedGH
python main.py --cfg model_heterogeneity/SFL_methods/FedGH/FedGH_on_cora.yaml --client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml federate.client_num 5
```



# Experiments on additional model architecture groups

We run experiments under three additional local model architecture groups (i.e., MHGNN_1, MHGNN_2, MHGNN_3). Please see Sec C.1 in our [appendices](./FedPPN_APPENDIX.pdf) (or [model settings folder](federatedscope/model_heterogeneity/model_settings)) for more details.

```python
# MHGNN_1
# Cora
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_1.yaml federate.client_num 3
# Citeseer
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_1.yaml federate.client_num 3
# PubMed
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_1.yaml federate.client_num 3

# MHGNN_2
# Cora
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 3
# Citeseer
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 3
# PubMed
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 3

# MHGNN_3
# Cora
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_cora.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 7
# Citeseer
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_citeseer.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 7
# PubMed
python main.py --cfg model_heterogeneity/SFL_methods/FedPPN/FedPPN_on_pubmed.yaml --client_cfg model_heterogeneity/model_settings/MHGNN_2.yaml federate.client_num 7


```

