import networkx as nx
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
from typing import *
from collections import defaultdict

# plt.style.use('dark_background')
# plt.style.use('seaborn-v0_8-dark')
# plt.style.use('seaborn-v0_8-poster')
# plt.style.use('seaborn-v0_8-pastel')
plt.style.use(['seaborn-v0_8-pastel', 'seaborn-v0_8-darkgrid'])


def get_info(G: nx.Graph):
    print(f"Is directed? - {nx.is_directed(G)}")
    print(f"Is weighted? - {nx.is_weighted(G)}")
    print(f"Is connected? - {nx.is_connected(G)}")
    print(f"Is planar? - {nx.is_planar(G)}")
    print(f"Is bipartite? - {nx.is_bipartite(G)}")
    print(f"Is complete? - {G.number_of_edges() == ((G.number_of_nodes() * (G.number_of_nodes() - 1))) // 2}")
    print(f"Is multigraph? - {isinstance(G, nx.MultiGraph)}")

    degree = G.number_of_nodes()

    print(f"Degree: {degree}")
    print(f"Size: {G.number_of_edges()}")
    print(f"Density: {nx.density(G)}")
    print(f"Diameter: {nx.diameter(G)}") # TODO uncomment
    print(f"Average shortest path length: {nx.average_shortest_path_length(G)}") # TODO uncomment

    # cliques = nx.find_cliques(G)
    # for clique in cliques:
    #     print(f" - {clique}")
    cliques = list(nx.enumerate_all_cliques(G))
    print("Cliques: ")
    MAX_NUMBER = 10
    for clique in cliques[-MAX_NUMBER:]:
        print(f" - {clique}")
        
    print(f"Max clique: {nx.approximation.max_clique(G)}")

    # Składowe spójne
    connected_components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(connected_components)}")
    print(f"Connected components: {connected_components}")

    k_components = nx.k_components(G)
    print(f"K-components: {k_components}")

    # Przeguby
    articulation_points = list(nx.articulation_points(G))
    print(f"Articulation points: {articulation_points}")

    bridges = list(nx.bridges(G))
    print(f"Bridges: {bridges}")

    degree = G.degree

    # FIXME(11jolek11): 
    # File "/home/jolek/University/cn1/complex_networks_3.py", line 49, in get_info
    # avg_degree = sum(dict(degree).values()) / len(degree)
    #                  ^^^^^^^^^^^^
    avg_degree = sum(dict(degree).values()) / len(degree)
    print(f"Avg degree: {avg_degree}")
    # print(f"###########3 {G.degree} -- {type(G.degree)}")

    degree_sequence = [d for n, d in G.degree()]
    is_scale_free = True
    # if not np.all(np.diff(sorted(degree_sequence)) >= 0) or not np.all(np.diff(np.diff(degree_sequence)) >= 0):
    #     is_scale_free = False

    # print(f"No scale graph: {is_scale_free}")

    # sns.set(style="dark")
    # plt.figure(figsize=(9, 4))
    # sns.histplot(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 1, 1), color='r')
    # plt.title("Nodes degree distribution")
    # plt.xlabel("Node degree")
    # plt.ylabel("Number of nodes")
    # plt.show()

# Napisać tak aby mozna coś z danych wyciągnąć (zadanie z analizy danych)
# analiza grafu

def get_edge_info(G, target_edge):
    print(f"Betweenness: {nx.edge_betweenness_centrality(G)[target_edge]}")

def get_node_info(G, target_node):
    print(f"Betweenness: {nx.betweenness_centrality(G)[target_node]}")
    print(f"Eigenvector centrality: {nx.eigenvector_centrality(G)[target_node]}") # Czy to powinno być tu?
    pagerank = nx.pagerank(G)
    print(f"PageRank: {pagerank[target_node]}")


def hist_nodes(graph: nx.Graph) -> None:
    centralities = [
        {'function': nx.degree, 'title': 'Stopień wierzchołka'},
        {'function': nx.closeness_centrality, 'title': 'Bliskość wierzchołka'},
        {'function': nx.betweenness_centrality, 'title': 'Pośrednictwo krawędzi'},
        {'function': nx.eigenvector_centrality, 'title': 'Centralność eigenvectora'},
        {'function': nx.pagerank, 'title': 'Pagerank'},
    ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    idx = 0
    for i in range(2):
        for j in range(3):
            if not (i == 1 and j == 1):
                current_centrality = centralities[idx]
                values = list(current_centrality['function'](graph).values()) if \
                    current_centrality['title'] != 'Stopień wierzchołka' else list(
                    dict(current_centrality['function'](graph)).values())
                median_value = np.median(values)
                mean_value = np.mean(values)

                n, bins, patches = axs[i, j].hist(values, bins=graph.number_of_nodes() // 10, edgecolor='black',
                                                  alpha=0.7)
                min_idx = np.digitize([min(bins)], bins)[0] - 1
                max_idx = np.digitize([max(bins)], bins)[0] - 1
                if min_idx < 0: min_idx = 0
                if max_idx > len(patches) - 1: max_idx = len(patches) - 1

                patches[min_idx].set_fc('blue')
                patches[max_idx].set_fc('orange')

                axs[i, j].axvline(median_value, color='green', linestyle='dashdot', linewidth=2,
                                  label=f'Mediana: {median_value}')
                axs[i, j].axvline(mean_value, color='blue', linestyle='dashdot', linewidth=2,
                                  label=f'Średnia: {mean_value}')
                axs[i, j].axvline(min(bins), color='red', linestyle='dashdot', linewidth=2,
                                  label=f'Min: {min(bins)}')
                axs[i, j].axvline(max(bins), color='yellow', linestyle='dashdot', linewidth=2,
                                  label=f'Max: {max(bins)}')

                axs[i, j].legend()

                axs[i, j].set_xlabel(current_centrality['title'])
                if current_centrality['title'] == 'Pośrednictwo krawędzi':
                    ylabel = "Liczba krawędzi"
                else:
                    ylabel = "Liczba wierzchołków"
                axs[i, j].set_ylabel(ylabel)

                axs[i, j].set_title(current_centrality["title"])
                idx += 1
            else:
                axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

    # plt.tight_layout()
    # plt.show()

def hist_nodes_2(graph: nx.Graph) -> None:
    centralities = [
        {'function': nx.degree, 'title': 'Rozkład stopni wierzchołka'},
        {'function': nx.closeness_centrality, 'title': 'Bliskość wierzchołka'},
        {'function': nx.betweenness_centrality, 'title': 'Pośrednictwo krawędzi'},
        {'function': nx.eigenvector_centrality, 'title': 'Eigenvector'},
        {'function': nx.pagerank, 'title': 'Pagerank'},
    ]

    # fig, axs = plt.subplots(figsize=(15, 10))
    idx = 0
    for i in range(2):
        for j in range(3):
            if not (i == 1 and j == 1):
                fig, axs = plt.subplots(figsize=(15, 10))
                current_centrality = centralities[idx]
                # list
                values = list(dict(current_centrality['function'](graph)).values()) if \
                    current_centrality['title'] != 'Stopień wierzchołka' else list(
                    dict(current_centrality['function'](graph)).values())
                median_value = np.median(values)
                mean_value = np.mean(values)

                n, bins, patches = axs.hist(values, bins=graph.number_of_nodes() // 10, edgecolor='black',
                                                  alpha=0.7)
                min_idx = np.digitize([min(bins)], bins)[0] - 1
                max_idx = np.digitize([max(bins)], bins)[0] - 1
                if min_idx < 0: min_idx = 0
                if max_idx > len(patches) - 1: max_idx = len(patches) - 1

                patches[min_idx].set_fc('blue')
                patches[max_idx].set_fc('orange')

                axs.axvline(median_value, color='red', linestyle='dashdot', linewidth=2,
                                  label=f'Med: {median_value:.3f}')
                axs.axvline(mean_value, color='green', linestyle='dashdot', linewidth=2,
                                  label=f'Avg: {mean_value:.3f}')
                axs.axvline(min(bins), color='blue', linestyle='dashdot', linewidth=2,
                                  label=f'Min: {min(bins):.3f}')
                axs.axvline(max(bins), color='orange', linestyle='dashdot', linewidth=2,
                                  label=f'Max: {max(bins):.3f}')

                axs.legend()

                axs.set_xlabel(current_centrality['title'])
                if current_centrality['title'] == 'Pośrednictwo krawędzi':
                    ylabel = "Liczba krawędzi"
                else:
                    ylabel = "Liczba wierzchołków"
                axs.set_ylabel(ylabel)

                axs.set_title(current_centrality["title"])
                idx += 1
            else:
                axs.axis('off')

            plt.plot()
            plt.tight_layout()
            plt.show()


def rest(graph: nx.Graph) -> None:
    connected = list(map(len, nx.connected_components(graph)))

    k_components = nx.k_components(graph)
    k_amounts = [0] * (len(k_components.keys()) + 1)
    for k, v in k_components.items():
        k_amounts[k] = len(v)

    bridges = list(nx.bridges(graph))
    bridgedict = defaultdict(int)
    for (u, v) in bridges:
        bridgedict[u] += 1
        bridgedict[v] += 1
    bridge_amounts = [0] * (max(bridgedict.values()) + 1)
    for k, v in bridgedict.items():
        bridge_amounts[v] += 1
    bridge_amounts[0] = graph.number_of_nodes() - sum(bridge_amounts)

    # print(f"Mosty: {bridges}")
    # print(f"Przeguby: {list(nx.articulation_points(graph))}")

    funcs = [
        # {'values': connected, 'title': 'Składowe spójne', 'x': 'Podgraf', 'y': 'Ilość wierzchołków'},
        {'values': k_amounts, 'title': 'K-spojność', 'x': 'k', 'y': 'Ilość podgrafów'},
        {'values': bridge_amounts, 'title': 'Rozkład wystepowania w mostach', 'x': 'Ilość w mostach', 'y': 'Ilość wierzchołków'}
    ]

    fig, axs = plt.subplots(1, len(funcs), figsize=(15, 10))
    for i in range(len(funcs)):
        values = funcs[i]
        mean = sum(values['values']) / len(values['values'])
        median = sum(values['values']) // 2
        temp = -1
        while median > 0:
            temp += 1
            median -= values['values'][temp]
        median = temp

        axs[i].bar([_ for _ in range(len(values['values']))], values['values'])
        # axs[i].axhline(y=mean, color='r', linestyle='--', label=f'Średnia ilości: {mean:.3f}')
        # axs[i].axvline(x=median, color='g', linestyle='-.', label=f'Mediana wielkości: {median:.3f}')
        axs[i].set_xlabel(values['x'])
        axs[i].set_xlabel(values['x'])
        axs[i].set_ylabel(values['y'])
        axs[i].set_title(values["title"])
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def cliq_info(graph: nx.Graph) -> None:
    cliques = nx.find_cliques(graph)
    cliques = [c for c in cliques]
    cliq_lens = [0] * (max(len(c) for c in cliques) + 1)
    for clique in cliques:
        cliq_lens[len(clique)] += 1

    node_cliques = nx.node_clique_number(graph)
    clique_number = [0] * (graph.number_of_nodes()+1)
    for k, v in node_cliques.items():
        print(f"k: {k} -- v: {v}")
        clique_number[k-1] = v

    node_maxs = [0] * (max(clique_number) + 1)
    for n in clique_number:
        node_maxs[n] += 1

    funcs = [
        {'values': cliq_lens, 'title': 'Wielkości klik i ich rozkład', 'x': 'Ilośc wierzchołków', 'y': 'Ilość podklik'},
        # {'values': clique_number, 'title': 'Największa klika z wierzchołkiem', 'x': 'Wierzchołek', 'y': 'Wielkość podkliki'},
        {'values': node_maxs, 'title': 'Ilość wierzchołków w klikach', 'x': 'Wielkość podkliki',
         'y': 'Ilość wierzchołków'}
    ]

    fig, axs = plt.subplots(1, len(funcs), figsize=(15, 10))
    for i in range(len(funcs)):
        values = funcs[i]
        mean = sum(values['values']) / len(values['values'])
        axs[i].bar([_ for _ in range(len(values['values']))], values['values'])
        # axs[i].axhline(y=mean, color='r', linestyle='--', label=f'Średnia ilości: {mean:.3f}')
        axs[i].set_xlabel(values['x'])
        axs[i].set_xlabel(values['x'])
        axs[i].set_ylabel(values['y'])
        axs[i].set_title(values["title"])
        axs[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
    
    # data = pd.read_csv("./data/higgs-mention_network.edgelist", sep=" ", header=None)

    data.rename(
        columns={
            0: "link1",
            1: "link2"},
        inplace=True)

    TEMP = nx.from_pandas_edgelist(data, "link1", "link2")

    # random_edge = random.choice(list(TEMP.edges))
    # random_node = random.choice(list(TEMP.nodes))
    # print(f"Random node >> {random_node}")
    # print(f"Random edge >> {random_edge}")


    print("####### Graph #######")
    get_info(TEMP)
    # print("####### Edge #######")
    # get_edge_info(TEMP, random_edge)
    # print("####### Node #######")
    # get_node_info(TEMP, random_node)
    # hist_nodes(TEMP)

    hist_nodes_2(TEMP)
    cliq_info(TEMP)
    rest(TEMP)
