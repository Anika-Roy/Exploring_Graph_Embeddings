import scipy.io as sio
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import multiprocessing
import logging  # Setting up the loggings to monitor gensim
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# Train a one-vs-rest logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors

import concurrent.futures
import os
import sys
import argparse
import datetime

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

class Node2Vec():
        def __init__(
                        self, 
                        graph : 'dict[np.ndarray]', 
                        window_size : int, 
                        embedding_size : int, 
                        walks_per_vertex : int, 
                        walk_length : int,
                        p : float,
                        q : float
                ) -> None:
                """
                Initialize the DeepWalk model. This directly from the paper https://arxiv.org/pdf/1403.6652.pdf.

                Parameters
                ----------
                graph : list[list[int]]
                        The adjacency list to be embedded. This is a dict of a np.array. Where adj_list[vertex][0] gives the neighbours indices.
                        adj_list[vertex][1] = weights of edges.
                window_size : int
                        The window size for the skipgram model.
                embedding_size : int
                        The size of the embedding. The final output matrix will be of size |V| x embedding_size.
                walks_per_vertex : int
                        The number of walks to perform per vertex.
                walk_length : int
                        The length of each walk.
                p : float
                        The return parameter.
                q : float
                        The in-out parameter.

                Methods
                -------
                generate_n_walks()
                        Generate n walks from the graph.
                train()
                        Train the model.
                update()
                        Feed model new walks.
                get_embeddings()
                        Return the embeddings.
                """

                # DeepWalk parameters
                self.g = graph
                self.w = window_size
                self.d = embedding_size
                self.gamma = walks_per_vertex
                self.epochs = self.gamma
                self.t = walk_length
                self.n = len(graph)
                self.p = p
                self.q = q
                
                # Cutoffs for sampling
                self.p0 = 1. / self.p / max(1., self.p, self.q)
                self.p1 = 1. /  max(1., self.p, self.q)
                self.p2 = 1. / self.q / max(1., self.p, self.q)
        
        def get_random_neighbour(self, vertex : 'int') -> 'int':
                """
                Fetches a random neighbour of a given vertex
                by sampling on the basis of the edge weights
                
                Parameters
                ----------
                vertex : int
                        The vertex whose neighbour we will sample
                        
                Returns
                -------
                int
                        The neighbour that was sampled
                """

                # Sample a vertex with probability proportional 
                # to the weight of the edge joining it.
                try:
                    # Sample a vertex with probability proportional 
                    # to the weight of the edge joining it.
                    return int(np.random.choice(self.g[vertex][0], p=self.g[vertex][1]))
                except (ValueError, IndexError, KeyError):
                    # Handle the error, and return the original vertex
                    return vertex
        
        def binary_search(self, arr, target): 
                # Perform binary search
                index = np.searchsorted(arr, target)

                # Check if the value was found at the given index
                return index < len(arr) and arr[index] == target

        def second_order_biased_random_walk(
                        self,
                        adj_list, 
                        walk_len : 'int', 
                        start_node : 'int', 
                        return_parameter :'float', 
                        in_out_parameter : 'float'
                        ) -> np.array:
                """
                Return a walk based on a 2nd order Markov Chain like transition.

                Parameters
                ----------
                adj_mat : list[list[int]]
                        Adjacency matrix of the graph.
                walk_len : int
                        Length of the random walk.
                start_node : int
                        Starting node of the random walk.
                return_parameter : float
                        The value of the "p" parameter
                in_out_parameter : float
                        The value of the "q" parameter
                
                Returns
                -------
                np.array
                        List of nodes in the random walk.  
                
                """
                # Array to store the walk
                walk = [
                        start_node,
                        self.get_random_neighbour(start_node) # The prev_node is never Null
                ]

                # Generate the rest of the walk
                for i in range(2, walk_len):
                    # Variable to check whether we added to walk
                    found = False
                    new_node = None
                    
                    # Keep running until sampled in the red region
                    while not found:
                        new_node = self.get_random_neighbour(walk[-1])
                        r = np.random.rand()

                        # Check if we will go back to the same node
                        if new_node == walk[-2]:
                            if r < self.p0:
                                found = True
                        
                        # Check if we are going to move by a distance of 1
                        # elif self.g[walk[-2]][new_node]:
                        elif self.binary_search(self.g[walk[-2]][0], new_node):
                            if r < self.p1:
                                found = True
                                    
                        else: # So we are moving by a distance of 2
                            if r < self.p2:
                                found = True
                                
                    walk.append(new_node)
            
                return walk

        def generate_walks_for_iteration(self, args):
                gamma, g, t, p, q, iteration = args
                np.random.seed(iteration)
                
                walks = []

                # print(f"Started with Iteration #{iteration} at {datetime.datetime.now()}", file=sys.stdout)
                
                for vertex in range(self.n):
                        walks.append(self.second_order_biased_random_walk(g, t, vertex, p, q))

                print(f"Done with Iteration #{iteration} at {datetime.datetime.now()}", file=sys.stdout)

                filename = f"walks_{iteration}.txt"

                with open(filename, 'a') as file:
                        for walk in walks:
                                walk_str = ' '.join(str(node) for node in walk)
                                file.write(walk_str + '\n')

                walks = []

        def generate_n_walks_parallel(self, num_iters: int, num_cores):
                args_list = [(self.gamma, self.g, self.t, self.p, self.q, iteration) for iteration in range(self.gamma)]

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                        executor.map(self.generate_walks_for_iteration, args_list)
                # for args in args_list:
                #         self.generate_walks_for_iteration(args)


                # print(f"Done with {self.n * num_iters} nodes at {datetime.datetime.now()}", file=sys.stdout)

        def train(self, epochs : int, lr : float) -> None:
                """
                Train the model.

                Parameters
                ----------
                epochs : int
                        Number of epochs to train the model for.
                lr : float
                        Learning rate for the optimizer.                
                """         

                print("Start generating random walks", file=sys.stdout) 
                start = time.perf_counter()
                # Generate many walks
                walks = self.generate_n_walks_parallel(self.gamma, cores)
                end = time.perf_counter()
                print(f"Done generating random walks in {round(end - start, 2)} seconds", file=sys.stdout)

                print("Creating word2vec model", file=sys.stdout)

                walks = None

                # Initialize the model
                self.model = Word2Vec(
                        walks,
                        negative= 10,
                        sg=1,
                        alpha=0.05,
                        epochs=epochs, 
                        vector_size=self.d,        # embedding dimension
                        window=self.w,             # context window size
                        min_count=0,
                        workers=cores
                )

                for iteration in range(self.gamma):
                        walks_file = f"walks_{iteration}.txt"
                        if os.path.exists(walks_file):
                                walks = LineSentence(walks_file)

                                # Train the model incrementally
                                self.model.build_vocab(walks, update=(iteration != 0))
                                self.model.train(walks, total_examples=self.model.corpus_count, epochs=epochs)
                        
                        if (iteration % 10) == 0:
                                print(f"Done training the model on the {iteration}th iteration", file=sys.stdout)

                print("Done creating word2vec model", file=sys.stdout)

def load_mat(filename):
    data = sio.loadmat(filename)
    return data

def sparse_matrix_to_adjacency_list(sparse_matrix):
    G = nx.from_scipy_sparse_array(sparse_matrix)

    adjacency_list = {}
    for node in G.nodes:
        neighbors = np.array(list(G.neighbors(node)), dtype=int)
        weights = np.array([float(G[node][neighbor]['weight']) for neighbor in neighbors], dtype=float)

        # Normalize weights
        weights /= np.sum(weights)

        # Sort neighbors and weights by node number
        sort_order = np.argsort(neighbors)
        neighbors = np.array(neighbors[sort_order], dtype=int)
        weights = weights[sort_order]

        adjacency_list[node] = np.column_stack((neighbors, weights)).T

    return adjacency_list

# taken from https://github.com/lucashu1/link-prediction/blob/master/gae/preprocessing.py

# Convert sparse matrix to tuple
import scipy.sparse as sp
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Takes in adjacency matrix in sparse format
# Returns: adj_train, train_edges, val_edges, val_edges_false, 
# test_edges, test_edges_false
def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print ('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_array(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print ('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print ("WARNING: not enough removable edges to perform full train-test split!")
        print ("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print ("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print ('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print ('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    if verbose == True:
        print ('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print ('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print ('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print ('Done with train-test split!')
        print ('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false
    
# Generate bootstrapped edge embeddings (as is done in node2vec paper)
# Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list, node_embeddings):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = node_embeddings[node1]
        emb2 = node_embeddings[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description='Node2Vec Training Script')
    parser.add_argument('p', type=float, help='Parameter p for Node2Vec')
    parser.add_argument('q', type=float, help='Parameter q for Node2Vec')
    args = parser.parse_args()

    print("Number of cores present: ", cores, file=sys.stdout)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", level=logging.INFO, filename='PPI.log', filemode='w')

    data_np = load_mat('PPI.mat')
    sparse_matrix = data_np['network']

    # Perform train-test split
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = mask_test_edges(sparse_matrix, test_frac=.3, val_frac=.1,verbose=True)
    g_train = nx.from_scipy_sparse_array(adj_train) # new graph object with only non-hidden edges
    print("The graph is connected : ", nx.is_connected(g_train))

    # Inspect train/test split
    print ("Total nodes:", sparse_matrix.shape[0])
    print ("Total edges:", int(sparse_matrix.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
    print ("Training edges (positive):", len(train_edges))
    print ("Training edges (negative):", len(train_edges_false))
    print ("Validation edges (positive):", len(val_edges))
    print ("Validation edges (negative):", len(val_edges_false))
    print ("Test edges (positive):", len(test_edges))
    print ("Test edges (negative):", len(test_edges_false))

    adjacency_list = {}
    for node in g_train.nodes:
        neighbors = np.array(list(g_train.neighbors(node)), dtype=int)
        weights = np.array([float(g_train[node][neighbor]['weight']) for neighbor in neighbors], dtype=float)

        # Normalize weights
        weights /= np.sum(weights)

        # Sort neighbors and weights by node number
        sort_order = np.argsort(neighbors)
        neighbors = np.array(neighbors[sort_order], dtype=int)
        weights = weights[sort_order]

        adjacency_list[node] = np.column_stack((neighbors, weights)).T

    # Iterate over the adjacency list and add a self loop 
    # if the neighbour list is empty
    for node in adjacency_list:
        if adjacency_list[node].size == 0:
            adjacency_list[node] = np.array([[node, 1.0]])


    dw = Node2Vec(adjacency_list, 10, 128, 80, 40, args.p, args.q)
    print("Node2Vec object created", file=sys.stdout)

    master_start = time.time()

    print("Training started...", file=sys.stdout)
    dw.train(20, 0.05)
    print("Training completed", file=sys.stdout)

    print("Saving embeddings...", file=sys.stdout)
    dw.model.wv.save_word2vec_format(f'PPI_{args.p}_{args.q}.txt')
    print("Embeddings saved", file=sys.stdout)

    master_end = time.time()

    print("All done in ", master_end - master_start, " seconds", file=sys.stdout)

    print(f"STARTING ANALYSIS FOR p = {args.p}, q = {args.q}", file=sys.stdout)
    # Step 1: load the embeddings from word_2_vec format using keyedvectors
    model = KeyedVectors.load_word2vec_format(f'PPI_{args.p}_{args.q}.txt', binary=False)

    # populate the node embeddings in a numpy array
    node_embeddings = np.zeros((model.vectors.shape[0], model.vectors.shape[1]))
    
    for i in range(model.vectors.shape[0]):
        node_embeddings[i] = model[str(i)]


    print("Embedding matrix shape : ", node_embeddings.shape, file=sys.stdout)

    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(train_edges, node_embeddings)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false, node_embeddings)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    
    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
    
    # Val-set edge embeddings, labels
    pos_val_edge_embs = get_edge_embeddings(val_edges, node_embeddings)
    neg_val_edge_embs = get_edge_embeddings(val_edges_false, node_embeddings)
    val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
    val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])
    
    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges, node_embeddings)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false, node_embeddings)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
    
    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)
    
    # Predicted edge scores: probability of being of class "1" (real edge)
    val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    val_roc = roc_auc_score(val_edge_labels, val_preds)

    # Predicted edge scores: probability of being of class "1" (real edge)
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_roc = roc_auc_score(test_edge_labels, test_preds)

    print ('node2vec Validation ROC score: ', str(val_roc), file=sys.stdout)
    print ('node2vec Test ROC score: ', str(test_roc), file=sys.stdout)

if __name__ == "__main__":
    main()