import torch
import numpy as np
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt

root_path = '../../result/Convergence_Curve_1110'
dataset = ['cora', 'citeseer', 'pubmed']

metric_str_map = {
    'Local': 'test_acc',
    'FedPPN': 'test_ensemble_model_acc'
}


def load_data(path, method):
    avg_file_path = f'{path}/avg_history_result.json'
    metric = metric_str_map.get(method, 'test_acc')
    with open(avg_file_path, 'r') as f:
        avg_history_result = json.load(f)
        avg_acc = avg_history_result[metric]
        avg_acc = np.array(avg_acc)

    # load client history result
    clients_acc = []
    for i in range(1, client_num + 1):
        with open(f'{path}/client_{i}_history_result.json', 'r') as f:
            client_history_result = json.load(f)
            clients_acc.append(client_history_result[metric])

    # find max and min values in clients_acc
    clients_acc = np.array(clients_acc)
    max_acc = np.max(clients_acc, axis=0)
    min_acc = np.min(clients_acc, axis=0)
    clients_std = np.std(clients_acc, axis=0)

    return avg_acc, clients_acc, max_acc, min_acc, clients_std


methods = ['Local', 'FedPPN', 'FedProto']
colors = ['red', 'blue', 'orange']
client_num = 5
plt.figure(figsize=(8, 6))
for idx, method in enumerate(methods):
    method_path = f'{root_path}/{method}_{dataset[0]}_{client_num}_clients'
    avg_acc, clients_acc, max_acc, min_acc, clients_std = load_data(method_path, method)
    plt.plot(avg_acc, color=colors[idx], label=method)
    plt.fill_between(range(len(avg_acc)), max_acc, min_acc, color=colors[idx], alpha=0.1)
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy')
    plt.legend()
plt.show()
