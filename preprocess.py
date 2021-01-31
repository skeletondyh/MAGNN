import numpy as np
import scipy.sparse as sp
import pickle
import os
import sys
import utils.preprocess

nc_datasets = ["DBLP", "ACM", "IMDB"]
metapaths = {
    "DBLP" : [[1, 0, 1], [1, 0, 2, 0, 1]],
    "ACM"  : [[0, 1, 0], [0, 2, 0]],
    "IMDB" : [[0, 1, 0], [0, 2, 0]]
}

targets = {"DBLP" : 1, "ACM" : 0, "IMDB" : 0}

def nc(dataset):
    prefix = os.path.expanduser("~/POSE/Data/HIN/" + dataset)
    print(prefix)

    out_dir = os.path.join("data", dataset)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    edges = pickle.load(open(os.path.join(prefix, "edges.pkl"), "rb"))
    adj = sum(edges)

    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adj, node_types, metapaths[dataset])
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, node_types, targets[dataset])
    
    for G, metapath in zip(G_list, metapaths[dataset]):
        nx.write_adjlist(G, os.path.join(out_dir, '-'.join(map(str, metapath)) + '.adjlist'))
    
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(metapaths[dataset], all_edge_metapath_idx_array):
        np.save(os.path.join(out_dir, '-'.join(map(str, metapath)) + '_idx.npy'), edge_metapath_idx_array)

def lp(dataset):
    pass

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset in nc_datasets:
        nc(dataset)
    else:
        lp(dataset)