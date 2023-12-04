import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.algorithms import community
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

# e = pd.read_csv('edges.csv')
# h = pd.read_csv('hero-network.csv')
# n = pd.read_csv('nodes.csv')

# Thor = h[h['hero1']=='THOR/DR. DONALD BLAK'].iloc[:333]
# Cap = h[h['hero1']=='CAPTAIN AMERICA'].iloc[:333]
# IronMan = h[h['hero1'].str.contains('IRON MAN/TONY STARK')].iloc[:333]

# Subset = pd.concat([Thor,Cap,IronMan],axis = 0)

# G=nx.from_pandas_edgelist(Subset, 'hero1', 'hero2')

data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
    
    # data = pd.read_csv("./data/higgs-mention_network.edgelist", sep=" ", header=None)

data.rename(
    columns={
        0: "link1",
        1: "link2"},
    inplace=True)

G = nx.from_pandas_edgelist(data, "link1", "link2")

# nx.draw(G, with_labels=True, node_size = 8)
# plt.show()

# a. Kliki
cliques = list(nx.find_cliques(G))
print("Cliques:", cliques)

# b. Moduły
louvain_communities = community.louvain_communities(G)
print("Louvain Communities:", louvain_communities)

# 2. Analiza Hierarchii Skupień

# # Przygotowanie macierzy odległości (sąsiedztwa)
# distances = nx.to_numpy_array(G)
# print(distances)
#
# # Aglomeracyjna analiza hierarchii skupień
# linkage_matrix = linkage(distances, method='average', metric='euclidean')
# dendrogram(linkage_matrix)
# plt.title("Dendrogram")
# plt.show()

# Aglomeracyjna analiza hierarchii skupień - od podstaw do całości, w odległośći euklidesowskiej
distances = nx.floyd_warshall_numpy(G)
linked = linkage(distances, 'single', metric='euclidean')
print(distances)
# Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top')
plt.show()

# Divisive Method - od całego grafu usuwamy po węźle (tu: "najcenniejszy" węzeł związany z miarami centralności)
girvan_newman_communities = list(community.girvan_newman(G))

# 3. Metody Podziałowe

# Metoda podziałowa
kmeans = KMeans(n_clusters=2, random_state=3214)
kmeans_labels = kmeans.fit_predict(distances)
print("K-Means Labels:", kmeans_labels)

pos = nx.spring_layout(G, seed=123)
nx.draw(G, pos, node_color=kmeans_labels, cmap=plt.cm.RdYlBu, with_labels=True)
plt.title("Network with Spectral Division")
plt.show()

# 5. Podział spektralny (niedeterministyczny algorytm, dlatego ustawiony random_state)
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=3214)
spectral_labels = spectral.fit_predict(distances)
print("Spectral Labels:", spectral_labels)

pos = nx.spring_layout(G, seed=123)
nx.draw(G, pos, node_color=spectral_labels, cmap=plt.cm.RdYlBu, with_labels=True)
plt.title("Network with Spectral Division")
plt.show()

# 4. Porównanie Metod

# Modularność
modularity = community.modularity(G, louvain_communities)
print("Modularity:", modularity)

modularity = community.modularity(G, girvan_newman_communities[-1])
print("Modularity:", modularity)

# znormalizowana korelacja pomiędzy dwoma podziałami (nakładanie się ze sobą)
nms = normalized_mutual_info_score(kmeans_labels, spectral_labels)
print(nms)

# 6. Analiza Hierarchiczna (dendrogram + różne wskaźniki) (hierarchical models nodes with more links are expected to have a lower clustering coefficient)
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
