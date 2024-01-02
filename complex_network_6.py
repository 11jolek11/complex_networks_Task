import pandas as pd
import networkx as nx
import numpy as np
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

def all_nodes_pagerank_to_csv(G: nx.Graph):
    my_dict = nx.pagerank(G)
    my_frame = pd.DataFrame({"Node": list(my_dict.keys()), "Rank": list(my_dict.values())})
    print(my_frame.shape)
    my_frame.index.name = "Index"
    my_frame.to_csv("page_rank_report.csv")
    return my_frame
        

def attack(G: nx.Graph, max_iters=20, max_removal_prop=1.0):
    G_copy = copy(G)
    G_size = G.number_of_nodes()

    centr = nx.degree_centrality(G)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    for i in range(len(dcs)):
        if i >= max_iters:
            break

        if dcs[i] in list(nx.articulation_points(G_copy)):
            continue

        # print(f"Iter {i}")
        
        # print(f"Node: {dcs[i]}")
        # print(f"Degree {G_copy.degree(dcs[i])}")
        # print(f"Articulation points no. :{len(list(nx.articulation_points(G_copy)))}")
        # print(f"Articulation points:{list(nx.articulation_points(G_copy))}")
        
        

        G_copy.remove_node(dcs[i])
        # print(f"Number of cc after removal: {len(list(nx.connected_components(G_copy)))}")
        diameter_hist.append(nx.diameter(G_copy))

        current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
        removal_propor.append(current_removal_propor)
    print(f"Diameter len {len(diameter_hist)}")
    print(f"Diameter len {len(removal_propor)}")
    
    plt.plot(removal_propor, diameter_hist)
    plt.savefig("attack_lol.jpg")
    # plt.show()

def fail(G: nx.Graph, max_iters = 20,  max_removal_prop=1.0):
    G_copy = copy(G)
    G_size = G.number_of_nodes()

    centr = nx.degree_centrality(G)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    current_iter = 0

    # while (current_removal_propor >= max_removal_prop) or max_iters <= current_iter:
    while max_iters >= current_iter:
        current_iter += 1

        node_for_removal = dcs.pop(random.randrange(len(dcs)))
        print(f"node_for_removal {node_for_removal}")

        if node_for_removal in list(nx.articulation_points(G_copy)):
            continue

        G_copy.remove_node(node_for_removal)
        diameter_hist.append(nx.diameter(G_copy))

        current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
        removal_propor.append(current_removal_propor)

    print(f"Diameter len {len(diameter_hist)}")
    print(f"Diameter len {len(removal_propor)}")
    
    plt.plot(removal_propor, diameter_hist)
    plt.savefig("fail_lol.jpg")
    # plt.show()

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


def degree_distribution(G:nx.Graph):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.savefig("GRAPH_CHECK.jpg")


if __name__ == "__main__":
    # data = pd.read_csv("./data/facebook_combined.txt", sep=" ", header=None)
    # data.rename(
    #     columns={
    #         0: "Source",
    #         1: "Target"},
    #     inplace=True)

    # data = pd.read_csv("./data/got-edges.csv", sep=",")
    # print(data.head())
    # data.drop(["Weight"], axis=1, inplace=True)
    
    # TEMP = nx.from_pandas_edgelist(data, "Source", "Target")

    # print(len(list(nx.connected_components(TEMP))))

    # attack(TEMP, max_removal_prop=0.2)
    # fail(TEMP, max_removal_prop=0.2)

    data = pd.read_csv("./data/facebook_combined.txt", sep=" ", header=None)
    data.rename(
        columns={
            0: "Source",
            1: "Target"},
        inplace=True)
    
    TEMP = nx.from_pandas_edgelist(data, "Source", "Target")
    all_nodes_pagerank_to_csv(TEMP)
    # degree_distribution(TEMP)

    # attack(TEMP, max_removal_prop=0.2)
    # fail(TEMP, max_removal_prop=0.2)

    # epidemy(TEMP,
    #         {
    #             'beta': 0.01,
    #             'lambda': 0.9,
    #             'gamma': 0.005,
    #             'alpha': 0.05,
    #             "fraction_infected": 0.05
    #         },
    #         500)
