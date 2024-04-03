import argparse
import os
from datetime import time

import numpy as np
import torch_geometric
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils

from data import load_data
from models.model import SIPSEModel
from metrics import *

import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description = "Polypharmacy Side Effect Prediction")
parser.add_argument("--seed", type = int, default = 1, help = "random seed")
parser.add_argument("--num_epoch", type = int, default = 300, help = "number of epochs")
parser.add_argument("--lr", type = float, default = 0.0005, help = "learning rate")
parser.add_argument("--chkpt_dir", type = str, default = "./checkpoint/", help = "checkpoint directory")
parser.add_argument("--latent_encoder_type", type = str, default = "linear", help = "latent encoder type")
parser.add_argument("--dropout", type = float, default = 0., help = "dropout rate")
parser.add_argument("--device", type = str, default = "cuda:0", help = "training device")
parser.add_argument("--noise_dim", type = list, default = [48], help = "noise_dim")
parser.add_argument("--output_dim", type = list, default = [64, 32], help = "output_dim")
parser.add_argument("--z_dim", type = int, default = 32, help = "z_dim")
parser.add_argument("--x_dim", type = int, default = 645, help = "x_dim")
parser.add_argument("--num_drugs", type = int, default = 645, help = "num_drugs")
parser.add_argument("--num_genes", type = int, default = 19081, help = "num_genes")
parser.add_argument("--K", type = int, default = 1, help = "K expend dim")
parser.add_argument("--J", type = int, default = 2, help = "J expend dim")
parser.add_argument("--eps", type = float, default = 1e-10, help = "eps")
parser.add_argument("--batch_size", type = int, default = 500, help = "batch_size")
hidden_dims = [525]

args = parser.parse_args()
torch_geometric.seed_everything(args.seed)

print("Load Data")
data,_ = load_data(return_augment=True)
edge_types = data.edge_types
augment = data["drug"].augment
rev_edge_types = []

for (src, relation, dst) in edge_types:
    rev_relation = f"rev_{relation}"
    rev_edge_types.append((dst, rev_relation, src))

transform = pyg_T.Compose([
    pyg_T.AddSelfLoops(),
    pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected = True,
        edge_types = edge_types, rev_edge_types = rev_edge_types, neg_sampling_ratio = 0.0
        , disjoint_train_ratio = 0.2)])

train_data, valid_data, test_data = transform(data)

for node in data.node_types:
    train_data[node].x = train_data[node].x.to_sparse().float()
    valid_data[node].x = valid_data[node].x.to_sparse().float()
    test_data[node].x = test_data[node].x.to_sparse().float()

for edge_type in rev_edge_types:
    del train_data[edge_type]
    del valid_data[edge_type]
    del test_data[edge_type]

for edge_type in edge_types:
    if edge_type[0] == edge_type[2]:
        train_data[edge_type].edge_index = pyg_utils.to_undirected(train_data[edge_type].edge_index)
        valid_data[edge_type].edge_index = pyg_utils.to_undirected(valid_data[edge_type].edge_index)
        test_data[edge_type].edge_index = pyg_utils.to_undirected(test_data[edge_type].edge_index)

print("Initialize model...")

side_effect_edge_types = list(train_data.edge_index_dict.keys())
keys_to_remove = [('gene', 'get_target', 'drug'), ('gene', 'interact', 'gene'),
                  ('drug', 'has_target', 'gene')]
for key in keys_to_remove:
    if key in side_effect_edge_types:
        side_effect_edge_types.remove(key)

train_DD_edge_index = train_data.edge_index_dict
for key in keys_to_remove:
    train_DD_edge_index.pop(key, None)

latent_dim = 16
net = SIPSEModel(side_effect_edge_types,
                 K=args.K, J=args.J,
                 noise_dim=args.noise_dim,
                 output_dim=args.output_dim,
                 hidden_dims=hidden_dims,
                 z_dim=args.z_dim,
                 dropout=args.dropout,
                 device=args.device,
                 eps=args.eps).to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
num_epoch = args.num_epoch
print("Training device: ", args.device)

best_val_roc = 0

augment = augment.float().to(args.device)
x_dict = train_data.x_dict['drug']
x_dict = x_dict.to(args.device)

valid_DD_edge_index = valid_data.edge_index_dict
test_DD_edge_index = test_data.edge_index_dict
lose_weights = {}
edge_types = list(train_DD_edge_index.keys())

for key in keys_to_remove:
    train_DD_edge_index.pop(key, None)
    valid_DD_edge_index.pop(key, None)
    test_DD_edge_index.pop(key, None)

print("Training...")
for i in range(0, len(train_DD_edge_index), args.batch_size):
    batch_edge_types = list(train_DD_edge_index.keys())[i:i+args.batch_size]
    print(f" batch {i} - {i+args.batch_size -1}" + '-'*60)
    with open("effect_batch1.txt", 'a') as f:
        f.write(f" batch {i} - {i+args.batch_size -1}")
    best_val_roc = 0.
    train_edge = {}
    valid_edge = {}
    for batch_edge in batch_edge_types:
        train_edge[batch_edge] = train_DD_edge_index[batch_edge]
        valid_edge[batch_edge] = valid_DD_edge_index[batch_edge]


    for edge_type in train_edge.keys():
        train_edge[edge_type] = train_edge[edge_type].to(args.device)
        valid_edge[edge_type] = valid_edge[edge_type].to(args.device)

    if os.path.exists(args.chkpt_dir + f"hetero_vgae_{args.seed}.pt"):
        net.load_state_dict(torch.load(args.chkpt_dir + f"hetero_vgae_{args.seed}.pt"))

    for epoch in range(num_epoch):
        warm_up = np.min([epoch / 300, 1])
        start = time.time()
        net.train()

        z_sample_iws , log_H_iws, _ = net.encode(x_dict, train_edge, augment)
        loss = net.cl_loss_iw(z_sample_iws, train_edge, warm_up, log_H_iws)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.detach().item()

        net.eval()
        with torch.no_grad():
            _ , _, psi_iw_vecs = net.encode(x_dict, valid_edge, augment)
            pos_edge_label_index_dict = valid_edge
            side_effect_edge_types_slice = list(valid_edge.keys())
            edge_label_index_dict = {}
            edge_label_dict = {}
            for edge_type in valid_edge.keys():
                src, relation, dst = edge_type
                if relation in ['get_target', 'interact', 'has_target']:
                    continue
                num_nodes = (valid_data.x_dict[src].shape[0], valid_data.x_dict[dst].shape[0])
                neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type],
                                                                   num_nodes=num_nodes)
                edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type],
                                                              neg_edge_label_index], dim=-1)

                pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
                neg_label = torch.zeros(neg_edge_label_index.shape[1])
                edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

            edge_pred = net.decode_all_relation(psi_iw_vecs, edge_label_index_dict, sigmoid=True)
            for relation in edge_pred.keys():
                edge_pred[relation] = edge_pred[relation].cpu()

            roc_auc, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, side_effect_edge_types_slice)

        end = time.time()
        print(f"| Epoch: {epoch} | Loss: {loss_value} | Val ROC: {roc_auc} |  Best ROC: {best_val_roc} | Time: {end - start}")
        with open("effect_batch1.txt", 'a') as f:
            f.write(f"| Epoch: {epoch} | Loss: {loss_value} | Val ROC: {roc_auc} |  Best ROC: {best_val_roc}\n")

        if best_val_roc < roc_auc:
            best_val_roc = roc_auc
            torch.save(net.state_dict(), args.chkpt_dir + f"hetero_vgae_{args.seed}.pt")
            print("----- Save Model -----")

for edge_type in test_DD_edge_index.keys():
    test_DD_edge_index[edge_type] = test_DD_edge_index[edge_type].to(args.device)

net.load_state_dict(torch.load(args.chkpt_dir + f"hetero_vgae_{args.seed}.pt"))
net.eval()

with torch.no_grad():
    _, _, psi_iw_vecs = net.encode(x_dict, test_DD_edge_index, augment)
    pos_edge_label_index_dict = test_DD_edge_index
    side_effect_edge_types_slice = list(test_DD_edge_index.keys())
    edge_label_index_dict = {}
    edge_label_dict = {}
    for edge_type in test_DD_edge_index.keys():
        src, relation, dst = edge_type
        if relation in ['get_target', 'interact', 'has_target']:
            continue
        num_nodes = (test_data.x_dict[src].shape[0], test_data.x_dict[dst].shape[0])
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type],
                                                           num_nodes=num_nodes)
        edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type],
                                                      neg_edge_label_index], dim=-1)

        pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

    edge_pred = net.decode_all_relation(psi_iw_vecs, edge_label_index_dict, sigmoid=True)
    for relation in edge_pred.keys():
        edge_pred[relation] = edge_pred[relation].cpu()

    roc_auc, total_roc_auc = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, side_effect_edge_types_slice)
    prec, prec_dict = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict,
                                                                  side_effect_edge_types_slice)
    apk, apk_dict = cal_apk(edge_pred, edge_label_dict, side_effect_edge_types_slice, k=50)

    print("-" * 100)
    print()
    print(f'| Test AUROC: {roc_auc} | Test AUPRC: {prec} | Test AP@50: {apk}')
    with open("effect_batch1.txt", 'a') as f:
        f.write(f"| Test AUROC: {roc_auc} | Test AUPRC: {prec} | Test AP@50: {apk}\n")