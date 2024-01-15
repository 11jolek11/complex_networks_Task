import pandas as pd
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import modularity
from copy import copy
import matplotlib.pyplot as plt
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

from graph_tiger.attacks import Attack
from graph_tiger.graphs import graph_loader

def plot_results(graph, steps, results, title):
   plt.figure(figsize=(6.4, 4.8))

   for method, result in results.items():
      result = [r / len(graph) for r in result]
      plt.plot(list(range(steps)), result, label=method)

   plt.ylim(0, 1)
   plt.ylabel('LCC')
   plt.xlabel('N_rm / N')
   plt.title(title)
   plt.legend()

   save_dir = os.getcwd() + '/plots/'
   os.makedirs(save_dir, exist_ok=True)

   plt.savefig(save_dir + title + '.pdf')
   plt.show()
   plt.clf()


graph = graph_loader(graph_type='BA', seed=1)

params = {
     'runs': 1,
     'steps': 30,
     'seed': 1,

     'attack': 'rb_node',
     'attack_approx': int(0.1*len(graph)),

     'plot_transition': True,
     'gif_animation': True,
     'gif_snaps': True,

     'edge_style': None,
     'node_style': None,
     'fa_iter': 20
 }

print("Creating example visualization")
a = Attack(graph, **params)
a.run_simulation()



# if __name__ == "__main__":
#     data = pd.read_csv("./data/got-edges.csv", sep=",")
#     print(data.head())
#     data.drop(["Weight"], axis=1, inplace=True)
    
#     TEMP = nx.from_pandas_edgelist(data, "Source", "Target")

#     print(len(list(nx.connected_components(TEMP))))

#     data = pd.read_csv("./data/facebook_combined.txt", sep=" ", header=None)
#     data.rename(
#         columns={
#             0: "Source",
#             1: "Target"},
#         inplace=True)
    
#     TEMP = nx.from_pandas_edgelist(data, "Source", "Target")
