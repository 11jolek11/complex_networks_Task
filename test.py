import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import *
from collections import defaultdict

matplotlib.use('TkAgg')


def parameters_of_graph(graph: nx.Graph) -> None:

    if isinstance(graph, nx.Graph):
        print(f"Jest to graf prosty, nieskierowany,", end=" ")
    elif isinstance(graph, nx.DiGraph):
        print(f"Jest to graf prosty, skierowany,", end=" ")
    elif isinstance(graph, nx.MultiGraph):
        print(f"Jest to multigraf, nieskierowany,", end=" ")
    else:
        print(f"Jest to multigraf, skierowany,", end=" ")

    if nx.is_weighted(graph):
        print(f"ważony,", end=" ")
    else:
        print(f"nieważony,", end=" ")

    if nx.is_connected(graph):
        print(f"spójny,", end=" ")
    else:
        print(f"niespójny,", end=" ")

    if nx.is_planar(graph):
        print(f"planarny,", end=" ")
    else:
        print(f"nieplanarny,", end=" ")

    if nx.is_bipartite(graph):
        print(f"dwudzielny.")
    else:
        print(f"niedwudzielny.")

    print()
    print(f"Rząd = {graph.number_of_nodes()},", end=" ")
    print(f"Rozmiar = {graph.number_of_edges()};", end=" ")
    if graph.number_of_edges() == \
            (graph.number_of_nodes() * (graph.number_of_nodes() - 1)) // 2:
        print(f"zatem jest to graf pełny.")
    else:
        print(f"zatem nie jest to graf pełny.")

    print(f"Gęstość = {nx.density(graph)}.")
    print(f"Średnica grafu = {nx.diameter(graph)},", end=" ")
    print(f"Średnia długość najkrótszej ścieżki = {nx.average_shortest_path_length(graph)}.")

    # wykresy1(graph)
    wykresy2(graph)
    # wykresy3(graph)
    # uproszczony_graf(graph, density_threshold=0.05)
    # uproszczony_graf(graph, density_threshold=0.1)
    # uproszczony_graf(graph, density_threshold=0.25)


def wykresy1(graph: nx.Graph) -> None:
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

                axs[i, j].axvline(median_value, color='red', linestyle='dashed', linewidth=2,
                                  label=f'Mediana: {median_value:.3f}')
                axs[i, j].axvline(mean_value, color='green', linestyle='dashed', linewidth=2,
                                  label=f'Średnia: {mean_value:.3f}')
                axs[i, j].axvline(min(bins), color='blue', linestyle='dashed', linewidth=2,
                                  label=f'Min: {min(bins):.3f}')
                axs[i, j].axvline(max(bins), color='orange', linestyle='dashed', linewidth=2,
                                  label=f'Max: {max(bins):.3f}')

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


def wykresy2(graph: nx.Graph) -> None:
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

    print(f"Mosty: {bridges}")
    print(f"Przeguby: {list(nx.articulation_points(graph))}")

    funcs = [
        {'values': connected, 'title': 'Składowe spójne', 'x': 'Podgraf', 'y': 'Ilość wierzchołków'},
        {'values': k_amounts, 'title': 'K-spójne podgrafów', 'x': 'k', 'y': 'Ilość podgrafów'},
        {'values': bridge_amounts, 'title': 'Spójność wierzchołka', 'x': 'Częstotliwosć występowania w mostach', 'y': 'Ilość wierzchołków'}
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    for i in range(3):
        values = funcs[i]
        mean = sum(values['values']) / len(values['values'])
        median = sum(values['values']) // 2
        temp = -1
        while median > 0:
            temp += 1
            median -= values['values'][temp]
        median = temp

        axs[i].bar([_ for _ in range(len(values['values']))], values['values'])
        axs[i].axhline(y=mean, color='r', linestyle='--', label=f'Średnia ilości: {mean:.3f}')
        axs[i].axvline(x=median, color='g', linestyle='-.', label=f'Mediana wielkości: {median:.3f}')
        axs[i].set_xlabel(values['x'])
        axs[i].set_xlabel(values['x'])
        axs[i].set_ylabel(values['y'])
        axs[i].set_title(values["title"])
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def wykresy3(graph: nx.Graph) -> None:
    cliques = nx.find_cliques(graph)
    cliques = [c for c in cliques]
    cliq_lens = [0] * (max(len(c) for c in cliques) + 1)
    for clique in cliques:
        cliq_lens[len(clique)] += 1

    node_cliques = nx.node_clique_number(graph)
    clique_number = [0] * (graph.number_of_nodes())
    for k, v in node_cliques.items():
        clique_number[k - 1] = v

    node_maxs = [0] * (max(clique_number) + 1)
    for n in clique_number:
        node_maxs[n] += 1

    funcs = [
        {'values': cliq_lens, 'title': 'Wielkości podklik', 'x': 'Ilośc wierzchołków', 'y': 'Ilość podklik'},
        {'values': clique_number, 'title': 'Największa klika z wierzchołkiem', 'x': 'Wierzchołek', 'y': 'Wielkość podkliki'},
        {'values': node_maxs, 'title': 'Ilość wierzchołków największych podklik', 'x': 'Wielkość podkliki',
         'y': 'Ilość wierzchołków'}
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    for i in range(3):
        values = funcs[i]
        mean = sum(values['values']) / len(values['values'])
        axs[i].bar([_ for _ in range(len(values['values']))], values['values'])
        axs[i].axhline(y=mean, color='r', linestyle='--', label=f'Średnia ilości: {mean:.3f}')
        axs[i].set_xlabel(values['x'])
        axs[i].set_xlabel(values['x'])
        axs[i].set_ylabel(values['y'])
        axs[i].set_title(values["title"])
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def split_graph_by_density(graph, density_threshold) -> List[nx.Graph]:
    nodes_sorted_by_degree = sorted(graph.nodes, key=lambda x: -graph.degree(x))
    visited = set()
    subgraphs = []

    for node in nodes_sorted_by_degree:
        if node not in visited:
            visited.add(node)
            subgraph = nx.Graph()
            subgraph.add_node(node)
            while True:
                if subgraph.number_of_nodes() == 1:
                    neighbors = list(graph.neighbors(node))
                    neighbors = [n for n in neighbors if n not in visited]
                    if neighbors:
                        new_node = max(neighbors, key=lambda x: graph.degree(x))
                        subgraph.add_node(new_node)
                        subgraph.add_edge(node, new_node)
                        visited.add(new_node)
                    else:
                        break
                else:
                    neighbors_in_subgraph = defaultdict(list)
                    for sub_node in subgraph.nodes:
                        neighbors = list(graph.neighbors(sub_node))
                        for n in neighbors:
                            if n not in visited:
                                neighbors_in_subgraph[n].append(sub_node)
                    if neighbors_in_subgraph:
                        key, value = max(neighbors_in_subgraph.items(), key=lambda x: len(x[1]))
                        if 2 * (subgraph.number_of_edges() + len(value)) \
                            / (subgraph.number_of_nodes() * (subgraph.number_of_nodes() + 1)) \
                            >= density_threshold:
                            subgraph.add_node(key)
                            visited.add(key)
                            for n in value:
                                subgraph.add_edge(key, n)
                        else:
                            break
                    else:
                        break
            subgraphs.append(subgraph)
    return subgraphs


def count_inter_subgraph_edges(original_graph, subgraphs) -> Dict[Tuple[int, int], int]:
    edge_counts = {}
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            subgraph1 = subgraphs[i]
            subgraph2 = subgraphs[j]
            count = 0
            for node1 in subgraph1.nodes:
                for node2 in subgraph2.nodes:
                    if original_graph.has_edge(node1, node2):
                        count += 1

            if count > 0: edge_counts[(i, j)] = count
    return edge_counts


def uproszczony_graf(graph: nx.Graph, density_threshold: float) -> None:
    new_graph = nx.Graph()
    subgraphs = split_graph_by_density(graph, density_threshold)
    edge_counts = count_inter_subgraph_edges(graph, subgraphs)

    for i, subgraph in enumerate(subgraphs):
        new_graph.add_node(i, node_count=subgraph.number_of_nodes())

    for (i, j), count in edge_counts.items():
        if count > 0:
            new_graph.add_edge(i, j, weight=count)

    node_count = nx.get_node_attributes(new_graph, 'node_count')
    weights = nx.get_edge_attributes(new_graph, 'weight')
    sizes = [node_count[node] * 20 for node in new_graph.nodes()]
    pos = nx.spring_layout(new_graph)

    nx.draw_networkx_nodes(new_graph, pos, node_color=[node_count[node] for node in new_graph.nodes()],
                           cmap=plt.get_cmap('viridis'), node_size=sizes)

    nx.draw_networkx_edges(new_graph, pos, width=[weights[edge] for edge in new_graph.edges()])

    node_labels = {node: str(node_count[node]) if node_count[node] != 1 else '' for node in new_graph.nodes()}
    nx.draw_networkx_labels(new_graph, pos, labels=node_labels, font_color='white')

    edge_labels = nx.get_edge_attributes(new_graph, 'weight')
    nx.draw_networkx_edge_labels(new_graph, pos, edge_labels=edge_labels)

    plt.show()


if __name__ == "__main__":
    edges: List[Tuple] = []
    with open("bio-diseasome.mtx", 'r') as f:
        while tmp := f.readline().split(' '):
            if not tmp[0]: break
            tmp[0], tmp[1] = int(tmp[0]), int(tmp[1])
            edges.append((tmp[0], tmp[1]))
    graph: nx.Graph = nx.from_edgelist(edges)
    parameters_of_graph(graph)

