# -*- coding: utf-8 -*-
"""complex_networks_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q3BSQnEAxNH7Avz6mt2FoHDu0zAZ6guo
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# !pip install -q pyvis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pyvis.network import Network

# /content/drive/MyDrive/Colab Notebooks/datasets/bn-mouse_visual-cortex_2.edges

# from google.colab import drive
# drive.mount('/content/drive')


# data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/datasets/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)

data.rename(
    columns={
        0: "link1",
        1: "link2"
},
    inplace=True
)

TEMP = nx.from_pandas_edgelist(data, "link1", "link2")

adj_matrix = nx.adjacency_matrix(TEMP).toarray()
inc_matrix = nx.incidence_matrix(TEMP).toarray()

edge_list_temp = data.to_numpy()

def edge_list_to_inc_matrix(edge_list):
    nodes = set()
    for edge in edge_list:
        nodes.add(edge[0])
        nodes.add(edge[1])

    node_to_index = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    num_edges = len(edge_list)
    incidence_matrix = [[0] * num_edges for _ in range(num_nodes)]

    for i, edge in enumerate(edge_list):
        start_node, end_node = edge
        start_index = node_to_index[start_node]
        end_index = node_to_index[end_node]
        incidence_matrix[start_index][i] = 1
        incidence_matrix[end_index][i] = 1
    return np.array(incidence_matrix)

def from_inc_to_adj(inc: np.ndarray):
    am = np.dot(inc, inc.T).astype(int)
    np.fill_diagonal(am, 0)
    return am

(adj_matrix == from_inc_to_adj(inc_matrix)).all()

A = nx.Graph()
I = nx.Graph()

A = nx.from_numpy_array(adj_matrix)
I = nx.from_numpy_array(from_inc_to_adj(inc_matrix))

B = nx.from_numpy_array(from_inc_to_adj(edge_list_to_inc_matrix(edge_list_temp)))

fig, ax = plt.subplots(figsize=(9, 6))
nx.draw(B, nx.kamada_kawai_layout(B), node_size=55, node_color="gold", node_shape="o")
ax.set_facecolor('deepskyblue')
fig.set_facecolor('deepskyblue')
ax.plot()
# fig.savefig("kawai.png")
plt.show()
"""# Draw"""

# plt.figure(figsize=(9, 6))

# nx.draw(A, nx.kamada_kawai_layout(A), node_size=55, node_color="skyblue", node_shape="s")
# plt.plot()

# nx.draw(I, nx.spring_layout(I), node_size=55, node_color="skyblue", node_shape="s")
# plt.plot()

fig, ax = plt.subplots(figsize=(9, 6))
nx.draw(A, nx.kamada_kawai_layout(A), node_size=55, node_color="gold", node_shape="*")
ax.set_facecolor('deepskyblue')
fig.set_facecolor('deepskyblue')
ax.plot()
# fig.savefig("kawai.png")
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
nx.draw(I, nx.spring_layout(I), node_size=55, node_color="gold", node_shape="*")
ax.set_facecolor('deepskyblue')
fig.set_facecolor('deepskyblue')
ax.plot()
# fig.savefig("spring.png")
plt.show()

"""# graphvis"""

net = Network(
    notebook = True,
    cdn_resources="remote",
    height="750px",
    width="100%",
    bgcolor="#222222",
    font_color="white",
    select_menu=True,
    filter_menu=True
)

net.from_nx(A)
net.show_buttons(filter_ = "physics")
net.show("nx.html")