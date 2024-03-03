import os

import scipy as sp
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import SparsePCA
from tdc.chem_utils import featurize

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def load_ppi(filepath = "bio-decagon-ppi/bio-decagon-ppi.csv"):
    df = pd.read_csv(filepath)
    print("Load Protein-Protein Interaction Graph")
    src, dst = df["Gene 1"].tolist(), df["Gene 2"].tolist()
    del df
    nodes = set(src + dst)
    net = nx.Graph()
    net.add_edges_from(zip(src, dst), verbose = False)
    gene_2_idx = {}

    for idx, node in enumerate(nodes):
        gene_2_idx[node] = idx
    print("Num nodes: ", len(nodes))
    print("Num edges: ", len(net.edges()))
    print()
    return net, gene_2_idx

def load_targets(filepath = "bio-decagon-targets/bio-decagon-targets.csv"):
    df = pd.read_csv(filepath)
    print("Load Drug-Target Interaction Graph")
    print("Num of interaction: ", df.shape[0])
    print()

    stitch_ids = df["STITCH"].tolist()
    genes = df["Gene"].tolist()
    stitch_2_gene = defaultdict(set)
    for stitch_id, gene in zip(stitch_ids, genes):
        stitch_2_gene[stitch_id].add(gene)

    return stitch_2_gene 

def load_categories(filepath = "bio-decagon-effectcategories/bio-decagon-effectcategories.csv"):
    df = pd.read_csv(filepath)

    side_effects = df["Side Effect"]
    side_effect_names = df["Side Effect Name"]
    disease_classes = df["Disease Class"]

    side_effect_2_name = {}
    side_effect_2_class = {}

    for side_effect, name, class_ in zip(side_effects, side_effect_names, disease_classes):
        side_effect_2_name[side_effect] = name
        side_effect_2_class[side_effect] = class_

    return side_effect_2_class, side_effect_2_name

def load_combo_side_effect(filepath = "bio-decagon-combo/bio-decagon-combo.csv"):
    df = pd.read_csv(filepath)
    print("Load Combination Side Effect Graph")
    combo_2_side_effect = defaultdict(set)
    side_effect_2_combo = defaultdict(set)
    side_effect_2_name = {}
    combo_2_stitch = {}

    stitch_ids_1 = df["STITCH 1"].tolist()
    stitch_ids_2 = df["STITCH 2"].tolist()
    side_effects = df["Polypharmacy Side Effect"].tolist()
    side_effect_names = df["Side Effect Name"].tolist()
    combos = (df["STITCH 1"] + "_" + df["STITCH 2"]).tolist()
    del df    
    stitch_set = set()
    items = zip(stitch_ids_1, stitch_ids_2, side_effects, side_effect_names, combos)
    for stitch_id_1, stitch_id_2, side_effect, side_effect_name, combo in items:
        combo_2_side_effect[combo].add(side_effect)
        side_effect_2_combo[side_effect].add(combo)
        side_effect_2_name[side_effect] = side_effect_name
        combo_2_stitch[combo] = [stitch_id_1, stitch_id_2]
        stitch_set.add(stitch_id_1)
        stitch_set.add(stitch_id_2)

    stitch_set = list(stitch_set)
    stitch_set.sort()
    stitch_2_idx = {}
    idx = 0
    for stitch in stitch_set:
        stitch_2_idx[stitch] = idx
        idx += 1

    num_interactions = sum(len(v) for v in combo_2_side_effect.values())
    print("Number of drug combinations: ", len(combo_2_stitch))
    print("Number of side effects: ", len(side_effect_2_name))
    print("Number of interactions: ", num_interactions)
    print()
    return combo_2_side_effect, side_effect_2_combo, side_effect_2_name, combo_2_stitch, stitch_2_idx

def remove_gene(side_effect_edge_types, keys_to_remove):
    for key in keys_to_remove:
        if key in side_effect_edge_types:
            side_effect_edge_types.remove(key)

def convert_combo_side_effect_to_edge_index_list(
        side_effect_2_combo, combo_2_stitch, stitch_2_idx): 
    edge_index_dict = defaultdict(list)
    for se in side_effect_2_combo.keys():
        for combo in side_effect_2_combo[se]:
            s1, s2 = combo_2_stitch[combo]
            edge_index_dict[("drug", se, "drug")].append([stitch_2_idx[s1], stitch_2_idx[s2]])

        if len(edge_index_dict[("drug", se, "drug")]) < 500:
            del edge_index_dict[("drug", se, "drug")]
        else:
            edge_index_dict[("drug", se, "drug")] = torch.tensor(edge_index_dict[("drug", se, "drug")]).long().T
    return stitch_2_idx, edge_index_dict

def load_mono_side_effect(filepath = "bio-decagon-mono/bio-decagon-mono.csv"):
    df = pd.read_csv(filepath)
    print("Load Mono Side Effect\n")
    stitch_ids = df["STITCH"]
    side_effects = df["Individual Side Effect"]
    side_effect_names = df["Side Effect Name"]

    items = zip(stitch_ids, side_effects, side_effect_names)

    stitch_2_side_effect = defaultdict(set)
    side_effect_2_name = {}

    for stitch_id, side_effect, side_effect_name in items:
        stitch_2_side_effect[stitch_id].add(side_effect)
        side_effect_2_name[side_effect] = side_effect_name

    return stitch_2_side_effect, side_effect_2_name

def generate_morgan_fingerprint(stitch_2_smile, stitch_2_idx):
    num_drugs = len(stitch_2_idx)
    x = np.identity(num_drugs)
    features = [0 for i in range(num_drugs)]
    for stitch, idx in stitch_2_idx.items():
        features[idx] = featurize.molconvert.smiles2ECFP4(stitch_2_smile[stitch])
    augment = np.array(features)
    return augment

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def combine_gene(edge_index_drug_gene, device, file_path):
    if os.path.isfile(file_path):
        print(f"File {file_path} exist,read in")
        drug_gene_pca = pd.read_csv(file_path)
        x_dict_pca = torch.tensor(drug_gene_pca.values, dtype=torch.float32)
    else:
        indices = torch.tensor(edge_index_drug_gene, dtype=torch.long, device="cpu")
        values = torch.squeeze(
            torch.ones(1, edge_index_drug_gene.shape[1], device="cpu"))
        size = (645, 19080)
        drug_gene_sparse_tensor = torch.sparse_coo_tensor(indices, values, size, device="cpu")

        drug_gene_sparse_tensor = drug_gene_sparse_tensor.coalesce()

        data_np = csr_matrix((drug_gene_sparse_tensor.values(), (drug_gene_sparse_tensor.indices()[0].numpy(),
                                                                 drug_gene_sparse_tensor.indices()[1].numpy())),
                             shape=drug_gene_sparse_tensor.shape)

        sparse_pca = SparsePCA(n_components=645)

        transformed_data = sparse_pca.fit_transform(data_np.toarray())


        x_dict_pca = torch.tensor(transformed_data, dtype=torch.float32)

        df = pd.DataFrame(transformed_data)

        df.to_csv(file_path, index=False)

        print(f"File {file_path} exist,write in")

    return x_dict_pca.to(device)


def handle_drug_edge_index(drug_drug_edge_index, args):
    keys_to_remove = [('gene', 'get_target', 'drug'), ('gene', 'interact', 'gene'),
                      ('drug', 'has_target', 'gene')]

    adj_orig_tiles = {}
    for key, edge_index in drug_drug_edge_index.items():
        size = ()
        src, relation, dst = key

        if relation == 'has_target':
            size = (args.num_drugs, args.num_genes)
        elif relation == 'get_target':
            size = (args.num_genes, args.num_drugs)
        elif relation == 'interact':
            size = (args.num_genes, args.num_genes)
        else:
            size = (args.num_drugs, args.num_drugs)

        indices = torch.tensor(edge_index, dtype=torch.long, device="cpu")
        values = torch.squeeze(
            torch.ones(1, edge_index.shape[1]))
        drug_drug_sparse_tensor = torch.sparse_coo_tensor(indices, values, size, device="cpu")
        drug_drug_dense_tensor = drug_drug_sparse_tensor.to_dense()

        adj_orig_tiles[key] = torch.tile(torch.unsqueeze(drug_drug_dense_tensor, dim=-1), [1, args.K])

        if relation in ['has_target', 'get_target']:
            drug_drug_edge_index[key] = drug_drug_sparse_tensor

    return drug_drug_edge_index, adj_orig_tiles
