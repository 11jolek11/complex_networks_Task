import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score


def find_cliques(G: nx.Graph):
    # z1
    return list(nx.find_cliques(G))

def modules(G: nx.Graph):
    # z1
    louvain_communities = community.louvain_communities(G)
    print("Louvain Communities:", louvain_communities)
    return louvain_communities

def hierarchy_analysis(G: nx.Graph):
    # z2
    distances = nx.floyd_warshall_numpy(G)
    linked = linkage(distances, 'single', metric='euclidean')
    print(distances)
    # Dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top')
    plt.show()

def split_methods(G: nx.Graph):
    # z3
    distances = nx.floyd_warshall_numpy(G)
    kmeans = KMeans(n_clusters=2, random_state=3214)
    kmeans_labels = kmeans.fit_predict(distances)
    print("K-Means Labels:", kmeans_labels)

    pos = nx.spring_layout(G, seed=123)
    nx.draw(G, pos, node_color=kmeans_labels, cmap=plt.cm.RdYlBu, with_labels=True)
    plt.title("Network with Spectral Division")
    plt.show()

def compare(G: nx.Graph):
    # z4
    louvain_communities = community.louvain_communities(G)
    girvan_newman_communities = list(community.girvan_newman(G))

    modularity = community.modularity(G, louvain_communities)
    print("Modularity:", modularity)

    modularity = community.modularity(G, girvan_newman_communities[-1])
    print("Modularity:", modularity)

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

def compare_precision(G: nx.Graph):
    # z6
    distances = nx.floyd_warshall_numpy(G)
    kmeans = KMeans(n_clusters=2, random_state=3214)
    kmeans_labels = kmeans.fit_predict(distances)

    # 5. Podział spektralny (niedeterministyczny algorytm, dlatego ustawiony random_state)
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=3214)
    spectral_labels = spectral.fit_predict(distances)
    nms = normalized_mutual_info_score(kmeans_labels, spectral_labels)
    print(nms)

    clustering_coefficient = nx.average_clustering(G)
    print(f"Clustering coefficient of the graph: {clustering_coefficient}")

    density = nx.density(G)
    print(f"Density of the graph: {density}")

    degree_centrality = nx.degree_centrality(G)
    print(f"Degree centrality: {degree_centrality}")

    closeness_centrality = nx.closeness_centrality(G)
    print(f"Closeness centrality: {closeness_centrality}")

def z6(G: nx.Graph):
    clustering_coefficient = nx.average_clustering(G)
    print(f"Clustering coefficient of the graph: {clustering_coefficient}")

    density = nx.density(G)
    print(f"Density of the graph: {density}")

    degree_centrality = nx.degree_centrality(G)
    print(f"Degree centrality: {degree_centrality}")

    closeness_centrality = nx.closeness_centrality(G)
    print(f"Closeness centrality: {closeness_centrality}")

    betweenness_centrality = nx.betweenness_centrality(G)
    print(f"Betweenness centrality: {betweenness_centrality}")


if __name__ == "__main__":
    data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
    
    # data = pd.read_csv("./data/higgs-mention_network.edgelist", sep=" ", header=None)

    data.rename(
        columns={
            0: "link1",
            1: "link2"},
        inplace=True)

    TEMP = nx.from_pandas_edgelist(data, "link1", "link2")

    modules(TEMP)
    hierarchy_analysis(TEMP)
    split_methods(TEMP)
    compare(TEMP)
    spectral(TEMP)
    compare_precision(TEMP)
