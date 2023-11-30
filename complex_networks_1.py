# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt
import random
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""# Zebranie danych"""

# data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/datasets/higgs-mention_network.edgelist", sep=" ", header=None)
data = pd.read_csv("./higgs-mention_network.edgelist", sep=" ", header=None)
data.rename(
    columns={
        0: "user1",
        1: "user2",
        2: "count"
},
    inplace=True
)

G = nx.from_pandas_edgelist(data, "user1", "user2", "count")

"""# 3a"""

def find_path(L, start, end):
    print(f"########################### FIND PATH ... for {str(L)}")
    return nx.shortest_path(L, source=start, target=end)

print(find_path(G, 677, 1163))

"""# 3b"""

tries = 0
def generate_random_eulerian_graph(n, k, p, max_tries=10):
    print(f"########################### GENERATOR")
    global tries
    # random graph
    Q = nx.connected_watts_strogatz_graph(n, k, p)
    print(f"Q test: {Q}")

    if not nx.is_eulerian(Q):
        odd_nodes = [node for node, degree in dict(Q.degree()).items() if degree % 2 == 1]
        while len(odd_nodes) > 1:
            u, v = random.sample(odd_nodes, 2)
            Q.add_edge(u, v)
            odd_nodes.remove(u)
            odd_nodes.remove(v)

    if nx.is_eulerian(Q):
        tries = 0
        if (Q == None):
            raise Exception("Failed to make the graph Eulerian.")
        return Q
    elif (tries <= max_tries):
        Q = nx.eulerize(Q)
        if nx.is_eulerian(Q):
            return Q
        generate_random_eulerian_graph(n, k, p)
    else:
        tries = 0
        print("Failed to make the graph Eulerian.")
        raise Exception("Failed to make the graph Eulerian.")

P = generate_random_eulerian_graph(1000, 2, 0.5)
print()
print(P.number_of_nodes())
print(nx.is_eulerian(P))
print(nx.is_eulerian(G))

def euler(B):
    print(f"########################### EULER ... for {str(B)}")
    if nx.is_semieulerian(B):
        print("is_semieulerian")
    if nx.is_eulerian(B):
        print("is_eulerian")
    else:
        try:
            print("is_not_eulerian")
            print("Generating new graph...")
            B = generate_random_eulerian_graph(1000, 2, 0.5)
            print("is_eulerian")
        except Exception:
            print("Failed to get eulerian graph. Because:")
            print("Please, rerun task")
            return None

    if nx.has_eulerian_path(B):
        print("has_eulerian_path")
    else:
        try:
            print("has_not_eulerian_path")
            print("Generating new graph...")
            B = generate_random_eulerian_graph(1000, 2, 0.5)
            print("has_eulerian_path")
        except Exception:
            print("Failed to get eulerian graph. Because:")
            print("Please, rerun task")
            return None

euler(P)

# euler(G)

"""# 3c"""

def flows(A, start, end):
    print(f"########################### FLOW ... for {str(A)}")
    flow_value, flow_dict = nx.maximum_flow(A, start, end, capacity="count")
    return flow_value, flow_dict

v, gg = flows(G, 677, 1163)
print(v)
