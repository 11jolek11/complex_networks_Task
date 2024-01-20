import networkx as nx
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib
matplotlib.use("TkAgg")

class ClientApp:
    def __init__(self):
        # Inicjalizacja pustego grafu
        self.graph = nx.Graph()
        # Pamięć dla nodów
        self.node_mem = []

        # Utworzenie obiektu figure i dwóch osi (ax1 i ax2)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Ustawienia dla ax1 (graf)
        self.ax1.set_title('Graph Animation')

        # Ustawienia dla ax2 (tekst)
        self.ax2.set_title('Graph Metrics')
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
        }

        self.weight = 0

    def fetch_graph_from_server(self):
        # Zapytanie do serwera
        server_url = "http://127.0.0.1:5000/get"  # Aktualizuj, jeśli serwer działa na innym hoście lub porcie
        response = requests.get("http://127.0.0.1:5000/get")

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Failed to fetch graph from server.")
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
        nodes_to_remove = [node for node in self.graph.nodes if self.graph.nodes[node].get('weight', 0) < self.weight]
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
            'avg_closeness': avg_closeness
        }

    def run(self, interval=10000):
        nx.draw(self.graph, with_labels=True, font_size=8, node_size=50, ax=self.ax1)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=interval)
        plt.show()

def define_text_boxes(app):
    interval_box_ax = plt.axes([0.62, 0.30, 0.2, 0.05])
    interval_box = widgets.TextBox(interval_box_ax, 'Interval:')
    interval_box.on_submit(lambda text: interval_update(app, text))

def interval_update(app, text):
    app.ani.event_source.stop()
    app.run(int(text))

def weight_update(app, text):
    app.weight = float(text)

if __name__ == "__main__":
    client_app = ClientApp()

    interval_box_ax = plt.axes([0.6, 0.30, 0.2, 0.05])
    interval_box = widgets.TextBox(interval_box_ax, 'Interval:')
    interval_box.on_submit(lambda text: interval_update(client_app, text))

    weight_box_ax = plt.axes([0.6, 0.25, 0.2, 0.05])
    weight_box = widgets.TextBox(weight_box_ax, "Weight: ")
    weight_box.on_submit(lambda text: weight_update(client_app, text))

    client_app.run()