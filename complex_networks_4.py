import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from matplotlib import cm as cm
from collections import defaultdict
from networkx.algorithms.community import modularity
from scipy.sparse.linalg import eigsh



def find_cliques(G: nx.Graph):
    # z1
    cliques = sorted(list(nx.find_cliques(G)), key=len, reverse=True)
    print(cliques)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos)
    colors = cm.get_cmap('hsv', len(cliques))
    for i, clique in enumerate(cliques):
        nx.draw_networkx_nodes(G, pos, nodelist=clique, node_color=colors(i), node_size=200)
    plt.show()

def modules(G: nx.Graph):
    # z1
    louvain_communities = community.louvain_communities(G)
    print("Louvain Communities:", louvain_communities)
    return louvain_communities

def agl_methods(G: nx.Graph):
    # z2
    # adj_matrix = nx.to_numpy_array(G)
    floyd = nx.floyd_warshall_numpy(G)
    linked = linkage(floyd, method="ward")
    # print(adj_matrix)
    print(floyd)
    # Dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top')
    # print(f"Agl: {modularity(G, linked)}")
    plt.show()

# def div_methods(G: nx.Graph):
#     # z3
#     # FIXME(11jolek11): KMeans is not split method?
#     distances = nx.floyd_warshall_numpy(G)
#     kmeans = KMeans(n_clusters=2, random_state=3214)
#     kmeans_labels = kmeans.fit_predict(distances)
#     print("K-Means Labels:", kmeans_labels)

#     pos = nx.spring_layout(G, seed=123)
#     nx.draw(G, pos, node_color=kmeans_labels, cmap=plt.cm.RdYlBu, with_labels=True)
#     plt.title("Network with Spectral Division")
#     plt.show()

def divisive(graph: nx.Graph, threshold=3):
    adj_matrix = nx.to_numpy_array(graph)
    modularity_div = {}

    meth = "complete"
    divisive_clusters = linkage(adj_matrix.T, method=meth)
    divisive_labels = fcluster(divisive_clusters, t=threshold, criterion='distance')

    for node, cluster in zip(graph.nodes, divisive_labels):
        graph.nodes[node]['div'] = cluster

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
    pos = nx.kamada_kawai_layout(graph)
    cmap = cm.get_cmap('hsv', max(divisive_labels))

    nx.draw(graph, pos, node_color=[cmap(i) for i in divisive_labels], with_labels=False, ax=ax1)
    ax1.set_title(f"Metoda podziałowa {meth}")

    divisive_dendrogram = dendrogram(divisive_clusters, ax=ax2, no_labels=True)
    ax2.set_title(f"Dendrogram {meth}")
    plt.show()

    modularity_div[meth] = modularity(graph, [{node for node, data in graph.nodes(data=True) if data['div'] == cluster} for cluster in set(divisive_labels)])
    print(f"Divisive: {modularity_div}")
    return modularity_div

# TODO(11jolek11): z3

def compare(G: nx.Graph):
    # z4
    louvain_communities = community.louvain_communities(G)
    girvan_newman_communities = list(community.girvan_newman(G))

    modularity = community.modularity(G, louvain_communities)
    print("Modularity for Louvain:", modularity)

    modularity = community.modularity(G, girvan_newman_communities[-1])
    print("Modularity for Girvan:", modularity)

    # znormalizowana korelacja pomiędzy dwoma podziałami (nakładanie się ze sobą)
    # nms = normalized_mutual_info_score(kmeans_labels, spectral_labels)
    # print(nms)

def spectral(G: nx.Graph):
    # z5
    distances = nx.floyd_warshall_numpy(G)
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=3214)
    spectral_labels = spectral.fit_predict(distances)
    print("Spectral Labels:", spectral_labels)

    pos = nx.spring_layout(G, seed=123)
    nx.draw(G, pos, node_color=spectral_labels, cmap=plt.cm.RdYlBu, with_labels=True)
    plt.title("Network with Spectral Division")
    plt.show()

# def compare_precision(G: nx.Graph):
#     # z6
#     distances = nx.floyd_warshall_numpy(G)
#     kmeans = KMeans(n_clusters=2, random_state=3214)
#     kmeans_labels = kmeans.fit_predict(distances)

#     # 5. Podział spektralny (niedeterministyczny algorytm, dlatego ustawiony random_state)
#     spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=3214)
#     spectral_labels = spectral.fit_predict(distances)
#     nms = normalized_mutual_info_score(kmeans_labels, spectral_labels)
#     print(nms)

#     clustering_coefficient = nx.average_clustering(G)
#     print(f"Clustering coefficient of the graph: {clustering_coefficient}")

#     density = nx.density(G)
#     print(f"Density of the graph: {density}")

#     degree_centrality = nx.degree_centrality(G)
#     print(f"Degree centrality: {degree_centrality}")

#     closeness_centrality = nx.closeness_centrality(G)
#     print(f"Closeness centrality: {closeness_centrality}")

# def z6(G: nx.Graph):
#     clustering_coefficient = nx.average_clustering(G)
#     print(f"Clustering coefficient of the graph: {clustering_coefficient}")

#     density = nx.density(G)
#     print(f"Density of the graph: {density}")

#     degree_centrality = nx.degree_centrality(G)
#     print(f"Degree centrality: {degree_centrality}")

#     closeness_centrality = nx.closeness_centrality(G)
#     print(f"Closeness centrality: {closeness_centrality}")

#     betweenness_centrality = nx.betweenness_centrality(G)
#     print(f"Betweenness centrality: {betweenness_centrality}")


if __name__ == "__main__":
    data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
    
    # data = pd.read_csv("./data/higgs-mention_network.edgelist", sep=" ", header=None)

    data.rename(
        columns={
            0: "link1",
            1: "link2"},
        inplace=True)

    TEMP = nx.from_pandas_edgelist(data, "link1", "link2")

    average_clustering = nx.average_clustering(TEMP)
    print(f"average_clustering: {average_clustering}")

    # cliques(TEMP)

    find_cliques(TEMP)
    modules(TEMP)
    agl_methods(TEMP)

    # split_methods(TEMP)

    compare(TEMP)
    divisive(TEMP, threshold=0)
    spectral(TEMP)
    # compare_precision(TEMP)
