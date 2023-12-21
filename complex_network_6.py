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



def all_nodes_pagerank(G: nx.Graph) -> dict:
    return nx.pagerank(G)

def attack(G: nx.Graph, max_iters=100, max_removal_prop=1.0):
    G_copy = copy(G)
    G_size = G.degree()

    centr = nx.degree_centrality(G)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    for i in range(len(dcs)):
        # print(i)
        # if current_removal_propor >= max_removal_prop:
            # break
        if max_iters <= i:
            break

        G_copy.remove_node(dcs[i])
        print(f"Number of cc after removal: {len(list(nx.connected_components(G_copy)))}")
        diameter_hist.append(nx.diameter(G_copy))

        current_removal_propor = 1 - (G_copy.degree()/G_size)
        removal_propor.append(current_removal_propor)
    
    plt.plot(removal_propor, diameter_hist)
    plt.show()

def fail(G: nx.Graph, max_iters = 100,  max_removal_prop=1.0):
    G_copy = copy(G)
    G_size = G.degree()

    centr = nx.degree_centrality(G)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    current_iter = 0

    # while (current_removal_propor >= max_removal_prop) or max_iters <= current_iter:
    while max_iters <= current_iter:
        current_iter += 1

        G_copy.remove_node(dcs.pop(random.randrange(len(dcs))))
        diameter_hist.append(nx.diameter(G_copy))

        current_removal_propor = 1 - (G_copy.degree()/G_size)
        removal_propor.append(current_removal_propor)
    
    plt.plot(removal_propor, diameter_hist)
    plt.show()

def epidemy(G: nx.Graph, model_params: dict, n: int):
    model  = ep.SEIRModel(G)

    cfg = mc.Configuration()

    for item in model_params.items():
        cfg.add_model_parameter(item[0], item[1])
    
    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(n)

    trends = model.build_trends(iterations)
    viz = DiffusionTrend(model, trends)
    viz.plot("diffusion")


if __name__ == "__main__":
    data = pd.read_csv("./data/facebook_combined.txt", sep=" ", header=None)
    data.rename(
        columns={
            0: "link1",
            1: "link2"},
        inplace=True)
    
    TEMP = nx.from_pandas_edgelist(data, "link1", "link2")

    print(len(list(nx.connected_components(TEMP))))

    # attack(TEMP, max_removal_prop=0.2)
    # fail(TEMP, max_removal_prop=0.2)

    epidemy(TEMP,
            {
                'beta': 0.01,
                'lambda': 0.9,
                'gamma': 0.005,
                'alpha': 0.05,
                "fraction_infected": 0.05
            },
            500)
