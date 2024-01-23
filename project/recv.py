import networkx as nx
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib
import seaborn as sns
import pandas as pd
from copy import copy
from statistics import mean
import random
import sys

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
import numpy as np
from networkx.algorithms.community import modularity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import cm

plt.style.use("seaborn-v0_8") # dark_background seaborn-v0_8

matplotlib.use("TkAgg")
import easygui

class ClientApp:
    def __init__(self):
        # Inicjalizacja pustego grafu
        self.graph = nx.Graph()
        # Pamięć dla nodów
        self.node_mem = []

        # Utworzenie obiektu figure i dwóch osi (ax1 i ax2)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Ustawienia dla ax1 (graf)
        self.ax1.set_title('')

        # Ustawienia dla ax2 (tekst)
        self.ax2.set_title('Stats')
        self.ax2.axis('off')  # Wyłączenie wyświetlania osi

        # Dodatkowy tekst na wykresie
        self.texts = {
            'order': self.ax2.text(0.1, 0.9, 'Order: ', fontsize=10),
            'size': self.ax2.text(0.1, 0.85, 'Size: ', fontsize=10),
            'density': self.ax2.text(0.1, 0.8, 'Density: ', fontsize=10),
            'diameter': self.ax2.text(0.1, 0.75, 'Diameter: ', fontsize=10),
            'radius': self.ax2.text(0.1, 0.7, 'Radius: ', fontsize=10),
            'avg_shortest_path': self.ax2.text(0.1, 0.65, 'Avg Shortest Path: ', fontsize=10),
            'avg_degree': self.ax2.text(0.1, 0.6, 'Avg Degree: ', fontsize=10),
            'clustering_coefficient': self.ax2.text(0.1, 0.55, 'Clustering Coefficient: ', fontsize=10),
            'connectivity_coefficient': self.ax2.text(0.1, 0.5, 'Connectivity Coefficient: ', fontsize=10),
            'avg_centrality': self.ax2.text(0.1, 0.45, 'Avg Centrality: ', fontsize=10),
            'degree_correlation_coefficient': self.ax2.text(0.1, 0.4, 'Degree Correlation Coefficient: ', fontsize=10),
            'avg_closeness': self.ax2.text(0.1, 0.35, 'Avg Closeness: ', fontsize=10),
            'transitivity': self.ax2.text(0.1, 0.3, 'Transitivity: ', fontsize=10),
        }

        self.weight = 0

        self.filter_node_closeness_centrality = 0
        self.filter_node_degree = 0
        self.filter_node_betweenes = 0
        self.filter_node_pagerank = 0

    def filter_graph(self):
        nodes_to_remove = set([node for node in self.graph.nodes if self.graph.nodes[node].get('weight', 0) < self.weight])

        degrees = {node: val for (node, val) in self.graph.degree()}
        pagerank = nx.pagerank(self.graph)
        betweenes = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)


        a = set([node for node in self.graph.nodes if degrees[node] < self.filter_node_degree])
        b = set([node for node in self.graph.nodes if pagerank[node] < self.filter_node_pagerank])
        c = set([node for node in self.graph.nodes if betweenes[node] < self.filter_node_betweenes])
        d = set([node for node in self.graph.nodes if closeness_centrality[node] < self.filter_node_closeness_centrality])

        nodes_to_remove.update(a)
        nodes_to_remove.update(b)
        nodes_to_remove.update(c)
        nodes_to_remove.update(d)

        nodes_to_remove = list(nodes_to_remove)

        return nodes_to_remove


    def fetch_graph_from_server(self):
        # Zapytanie do serwera
        response = requests.get("http://127.0.0.1:5000/get")

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Connection to server broken {response.status_code}")
            return {'nodes': [], 'edges': []}

    def update_graph(self, graph_data):
        # Wyczyszczenie istniejącego grafu
        self.graph.clear()

        # Dodanie wierzchołków i krawędzi do grafu
        for node_data in graph_data['nodes']:
            node_id = node_data['id']
            node_weight = node_data.get('weight', 0)  # Pobierz wagę, jeśli istnieje, w przeciwnym razie ustaw na 0
            self.graph.add_node(node_id, weight=node_weight)

        self.graph.add_edges_from(graph_data['edges'])

        # Usunięcie wierzchołków o wadze mniejszej niż self.weight
        # nodes_to_remove = [node for node in self.graph.nodes if self.graph.nodes[node].get('weight', 0) < self.weight]
        nodes_to_remove = self.filter_graph()
        self.graph.remove_nodes_from(nodes_to_remove)

    def animate(self, frame):
        # Pobranie grafu z serwera
        graph_data_from_server = self.fetch_graph_from_server()

        # Sprawdzenie, czy struktura grafu się zmieniła
        if graph_data_from_server['nodes'] != self.node_mem:

            # Zapisanie nowej struktury do pamięci
            self.node_mem = graph_data_from_server['nodes']

            # Zaktualizuj graf
            self.update_graph(graph_data_from_server)

            # Wyczyść ax1 i narysuj nowy graf
            self.ax1.clear()
            nx.draw(self.graph, with_labels=True, font_size=8, node_size=50, ax=self.ax1)

            # Aktualizacja tekstu w ax2
            graph_metrics = self.calculate_graph_metrics()
            for key, text_object in self.texts.items():
                text_object.set_text(f'{key.capitalize()}: {graph_metrics[key]}')

    def calculate_graph_metrics(self):
        # Rząd grafu (liczba wierzchołków)
        order = len(self.graph.nodes())

        # Rozmiar grafu (liczba krawędzi)
        size = len(self.graph.edges())

        # Gęstość grafu
        density = nx.density(self.graph)

        # Średnica grafu
        try:
            diameter = nx.diameter(self.graph)
        except nx.NetworkXError:
            diameter = float("inf")

        # Promień grafu
        try:
            radius = nx.radius(self.graph)
        except nx.NetworkXError:
            radius = float("inf")

        # Średnia najkrótsza ścieżka
        try:
            avg_shortest_path = nx.average_shortest_path_length(self.graph)
        except nx.NetworkXError:
            avg_shortest_path = float("inf")

        # Stopień grafu
        degree_values = list(dict(self.graph.degree()).values())
        avg_degree = sum(degree_values) / order

        # Współczynnik skupienia grafu
        clustering_coefficient = nx.average_clustering(self.graph)

        # Współczynnik spójności grafu
        connected_components = list(nx.connected_components(self.graph))
        connectivity_coefficient = len(connected_components)

        # Współczynnik centralizacji grafu
        centrality_coefficients = nx.degree_centrality(self.graph).values()
        avg_centrality = sum(centrality_coefficients) / order

        # Współczynnik korelacji stopniowej grafu
        try:
            degree_correlation_coefficient = nx.degree_pearson_correlation_coefficient(self.graph)
        except Exception:
            degree_correlation_coefficient = None

        # Współczynnik bliskości grafu
        closeness_coefficients = nx.closeness_centrality(self.graph).values()
        avg_closeness = sum(closeness_coefficients) / order

        transitivity_index = nx.transitivity(self.graph)

        # Zwróć obliczone parametry
        return {
            'order': order,
            'size': size,
            'density': density,
            'diameter': diameter,
            'radius': radius,
            'avg_shortest_path': avg_shortest_path,
            'avg_degree': avg_degree,
            'clustering_coefficient': clustering_coefficient,
            'connectivity_coefficient': connectivity_coefficient,
            'avg_centrality': avg_centrality,
            'degree_correlation_coefficient': degree_correlation_coefficient,
            'avg_closeness': avg_closeness,
            'transitivity': transitivity_index
        }

    def run(self, interval=10000):
        nx.draw(self.graph, with_labels=True, font_size=8, node_size=50, ax=self.ax1)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=interval)
        plt.show()
        sys.exit(0)

def define_text_boxes(app):
    pass
    # interval_box_ax = plt.axes([0.62, 0.30, 0.2, 0.05])
    # interval_box = widgets.TextBox(interval_box_ax, 'Interval:')
    # interval_box.on_submit(lambda text: interval_update(app, text))

def interval_update(app, text):
    app.ani.event_source.stop()
    app.run(int(text))

def weight_update(app, text):
    app.weight = float(text)

def all_nodes_pagerank(client):
    graph = copy(client.graph)
    my_dict = nx.pagerank(graph)
    my_frame = pd.DataFrame({"Node": list(my_dict.keys()), "Rank": list(my_dict.values())})
    my_frame.index.name = "Index"
    sns.displot(my_frame, x="Rank", kde=True)
    path = "pagerank.jpg"
    plt.savefig(path)
    easygui.msgbox(f"Saved in {path}", title="PageRank complete")

def epidemy(client, model_params: dict = {
                'beta': 0.01,
                'lambda': 0.9,
                'gamma': 0.005,
                'alpha': 0.05,
                "fraction_infected": 0.05
            }, n: int = 500):

    graph = copy(client.graph)

    model = ep.SEIRModel(graph)

    cfg = mc.Configuration()

    for item in model_params.items():
        cfg.add_model_parameter(item[0], item[1])

    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(n)

    trends = model.build_trends(iterations)
    viz = DiffusionTrend(model, trends)
    path = "diffusion.jpg"
    viz.plot(path)
    easygui.msgbox(f"Saved in {path}", title="Epidemy simulation complete")

def attack(client, max_iters=20):
    G_copy = copy(client.graph)
    G_size = G_copy.number_of_nodes()

    centr = nx.degree_centrality(G_copy)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    for i in range(len(dcs[:max_iters])):
        G_copy.remove_node(dcs[i])

        comps = list((G_copy.subgraph(c) for c in nx.connected_components(G_copy)))
        temp = [nx.diameter(comp.to_undirected()) for comp in comps]
        if temp:
            diameter_hist.append(mean(temp))
        else:
            diameter_hist.append(0)

        current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
        removal_propor.append(current_removal_propor)
    
    fig, ax = plt.subplots()

    # create the barchart
    sns.scatterplot(x=removal_propor, y=diameter_hist)
    # plt.set_title('Attack simulation')
    # plt.set_ylabel('Diameter')
    path = "attack.jpg"
    fig.savefig(path)
    easygui.msgbox(f"Saved in {path}", title="Attack simulation complete")


def fail(client, max_iters=20):
    G_copy = copy(client.graph)
    G_size = G_copy.number_of_nodes()

    centr = nx.degree_centrality(G_copy)
    dcs = pd.Series(centr)
    dcs.sort_values(ascending=False, inplace=True)

    removal_propor = []
    diameter_hist = []

    current_removal_propor = 0.0

    dcs = dcs.index.values.tolist()

    current_iter = 0

    while max_iters >= current_iter:
        print(f"Fail iter: {current_iter}")
        current_iter += 1

        # len(dcs )
        if len(dcs) >= 1:
            node_for_removal = dcs.pop(random.randrange(len(dcs)))
        else:
            break

        print(f"node_for_removal {node_for_removal}")

        G_copy.remove_node(node_for_removal)

        comps = list((G_copy.subgraph(c) for c in nx.connected_components(G_copy)))

        temp = [nx.diameter(comp.to_undirected()) for comp in comps]
        if temp:
            diameter_hist.append(mean(temp))
        else:
            diameter_hist.append(0)

        current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
        removal_propor.append(current_removal_propor)
    
    fig, ax = plt.subplots()

    sns.scatterplot(x=removal_propor, y=diameter_hist)
    # plt.set_title('Fail simulation')
    # plt.set_ylabel('Diameter')
    path = "fail.jpg"
    fig.savefig(path)
    easygui.msgbox(f"Saved in {path}", title="Fail simulation complete")

def degree_distribution(client):
    G_copy = copy(client.graph)
    degree_sequence = sorted((d for _, d in G_copy.degree()), reverse=True)

    fig = plt.figure("Degree of a random graph", figsize=(6, 4))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G_copy.subgraph(sorted(nx.connected_components(G_copy), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of self._graph_storage")
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

    path = "degree_distr.jpg"
    plt.savefig(path)
    easygui.msgbox(f"Saved in {path}", title="Complete")

def divisive(client, threshold=3, meth: str = "complete", criterion: str = "distance"):
    G_copy = copy(client.graph)
    adj_matrix = nx.to_numpy_array(G_copy)
    modularity_div = {}

    divisive_clusters = linkage(adj_matrix.T, method=meth)
    divisive_labels = fcluster(divisive_clusters, t=threshold, criterion=criterion)

    for node, cluster in zip(G_copy.nodes, divisive_labels):
        G_copy.nodes[node]['div'] = cluster

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
    pos = nx.kamada_kawai_layout(G_copy)
    cmap = cm.get_cmap('hsv', max(divisive_labels))

    nx.draw(G_copy, pos, node_color=[cmap(i) for i in divisive_labels], with_labels=False, ax=ax1)
    ax1.set_title(f"Metoda podziałowa {meth}")

    dendrogram(divisive_clusters, ax=ax2, no_labels=True)
    ax2.set_title(f"Dendrogram {meth}")
    path = "divisive.jpg"
    plt.savefig(path)

    modularity_div[meth] = modularity(G_copy, [{node for node, data in G_copy.nodes(data=True) if data['div'] == cluster} for cluster in set(divisive_labels)])
    print(f"Divisive: {modularity_div}")
    easygui.msgbox(f"Saved in {path}", title=f"Divisive complete. Score {modularity_div}")
# len([nx.diameter(comp.to_undirected()) for comp in comps]) == 0

def agl_methods(client, method: str = "ward"):
    G_copy = copy(client.graph)
    floyd = nx.floyd_warshall_numpy(G_copy)
    floyd[floyd == np.inf] = 999

    linked = linkage(floyd, method=method)
    # Dendrogram
    # plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    dendrogram(linked, orientation='top')
    path = "aglomerative.jpg"
    fig.savefig(path)
    easygui.msgbox(f"Saved in {path}", title=f"Divisive complete")

def triadic_census_info(client):
    G_copy = copy(client.graph)
    triadic_census = nx.triadic_census(G_copy)

    message = ""

    for key, value in triadic_census.items():
        message += f"{key}: {value} \n"

    easygui.msgbox(
        message,
        title="Report")
    
def refresh_forced(client):
    print("REFRESH")
    graph_data_from_server = client.fetch_graph_from_server()
    client.update_graph(graph_data_from_server)



if __name__ == "__main__":
    client_app = ClientApp()

    # interval_box_ax = plt.axes([0.6, 0.30, 0.2, 0.05])
    # interval_box = widgets.TextBox(interval_box_ax, 'Interval:')
    # interval_box.on_submit(lambda text: interval_update(client_app, text))

    # weight_box_ax = plt.axes([0.6, 0.20, 0.2, 0.05])
    # weight_box = widgets.TextBox(weight_box_ax, "Weight filter: ")
    # weight_box.on_submit(lambda text: weight_update(client_app, text))

    refresh_ax = plt.axes([0.6, 0.25, 0.2, 0.05])
    refresh_btn = widgets.Button(refresh_ax, "Force refresh")
    refresh_btn.on_clicked(lambda _: refresh_forced(client_app))

    pagerank_box_ax = plt.axes([0.6, 0.20, 0.2, 0.05])
    pagerank_btn = widgets.Button(pagerank_box_ax, "PageRank")
    pagerank_btn.on_clicked(lambda _ :all_nodes_pagerank(client_app))

    epidemy_box_ax = plt.axes([0.6, 0.15, 0.2, 0.05])
    epidemy_btn = widgets.Button(epidemy_box_ax, "Epidemy simulation")
    epidemy_btn.on_clicked(lambda _ :epidemy(client_app))

    attack_box_ax = plt.axes([0.6, 0.1, 0.2, 0.05])
    attack_btn = widgets.Button(attack_box_ax, "Attack simulation")
    attack_btn.on_clicked(lambda _ :attack(client_app))

    fail_box_ax = plt.axes([0.6, 0.05, 0.2, 0.05])
    fail_btn = widgets.Button(fail_box_ax, "Fail simulation")
    fail_btn.on_clicked(lambda _ :fail(client_app))

    degree_distribution_box_ax = plt.axes([0.8, 0.15, 0.2, 0.05])
    degree_distribution__btn = widgets.Button(degree_distribution_box_ax, "Visualize degree distribution")
    degree_distribution__btn.on_clicked(lambda _ :degree_distribution(client_app))

    divisive_box_ax = plt.axes([0.8, 0.1, 0.2, 0.05])
    divisive_btn = widgets.Button(divisive_box_ax, "Modularity: Divisive method")
    divisive_btn.on_clicked(lambda _ :divisive(client_app))

    agl_methods_box_ax = plt.axes([0.8, 0.05, 0.2, 0.05])
    agl_methods_btn = widgets.Button(agl_methods_box_ax, "Modularity: Aglomerative method")
    agl_methods_btn.on_clicked(lambda _ :agl_methods(client_app))

    # triadic_census_info_ax = plt.axes([0.8, 0.00, 0.2, 0.05])
    # triadic_census_info_btn = widgets.Button(triadic_census_info_ax, "Triadic Census")
    # triadic_census_info_btn.on_clicked(lambda _ :triadic_census_info(client_app))

    client_app.run()
    sys.exit(0)

