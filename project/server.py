from flask import Flask, jsonify
import networkx as nx
import random
import time
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("../data/facebook_combined.txt", sep=" ", header=None)
data.rename(
    columns={
        0: "Source",
        1: "Target"},
    inplace=True)
    
G1 = nx.from_pandas_edgelist(data, "Source", "Target")

# Inicjalizacja pustego grafu G2
G2 = nx.Graph()

# Dodanie losowego podgrafu do G2
random_nodes = random.sample(list(G1.nodes()), 10)
G2.add_nodes_from(random_nodes)
for node in random_nodes:
    G2.add_node(node, weight=1)  # Waga ustawiona na 1 dla początkowych 10 węzłów
G2.add_edges_from(G1.subgraph(random_nodes).edges())


def update_external_graph():
    # Metoda zewnętrzna - zwraca listę wierzchołków, krawędzi i wag G2
    nodes_data = list(G2.nodes(data=True))
    edges_data = list(G2.edges())

    # Przygotowanie słownika z danymi wierzchołków, krawędzi i wag
    graph_data = {
        'nodes': [{'id': node, 'weight': data.get('weight', 0)} for node, data in nodes_data],
        'edges': edges_data
    }

    return jsonify(graph_data)


def update_internal_graph():
    # Metoda wewnętrzna - co 60 sekund
    while True:
        time.sleep(5)

        # Zmiana wag przed dodaniem nowego wierzchołka
        for node_G2 in G2.nodes:
            G2.nodes[node_G2]['weight'] = random.uniform(max(0, G2.nodes[node_G2]['weight'] - 0.01),
                                                         min(1, G2.nodes[node_G2]['weight'] + 0.01))

        # Wybierz losowy wierzchołek z G2
        random_node_G2 = random.choice(list(G2.nodes()))

        # Wybierz losowego sąsiada tego wierzchołka z G1
        neighbors_G1 = list(G1.neighbors(random_node_G2))
        if neighbors_G1:
            random_neighbor_G1 = random.choice(neighbors_G1)

            # Sprawdź, czy sąsiad jest już w G2
            if random_neighbor_G1 in G2:
                # Jeśli tak, usuń go z G2
                G2.remove_node(random_neighbor_G1)
            else:
                # Jeśli nie, dodaj go do G2 wraz z krawędziami
                G2.add_node(random_neighbor_G1, weight=random.uniform(max(0, G2.nodes[random_node_G2]['weight'] - 0.05),
                                                                      min(1, G2.nodes[random_node_G2]['weight'] + 0.05)))
                G2.add_edges_from(G1.subgraph(G2.nodes()).edges())

        # Usunięcie wierzchołków o wadze zerowej z G2
        zero_weight_nodes = [node for node in G2.nodes if G2.nodes[node]['weight'] == 0]
        G2.remove_nodes_from(zero_weight_nodes)


# Uruchomienie wątku dla metody wewnętrznej
import threading

threading.Thread(target=update_internal_graph, daemon=True).start()


@app.route('/get', methods=['GET'])
def get():
    return update_external_graph()


if __name__ == '__main__':
    app.run(debug=False)
