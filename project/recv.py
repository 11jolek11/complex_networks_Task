from os import stat
import random
from copy import copy
from textwrap import fill

import requests
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from statistics import mean
import seaborn as sns
from networkx.algorithms.community import modularity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)


import tkinter as tk
import tkinter.ttk as ttk


matplotlib.use('TkAgg')


class GraphEngine():
    def __init__(self) -> None:
        self._graph_storage = nx.Graph()

    @staticmethod
    def __get_figure():
        return Figure(figsize=(6, 4), dpi=100)

    def add_data(self, new_data):
        self._graph_storage.add_edges_from(new_data)

    def all_nodes_pagerank(self):
        my_dict = nx.pagerank(self)
        my_frame = pd.DataFrame({"Node": list(my_dict.keys()), "Rank": list(my_dict.values())})
        my_frame.index.name = "Index"
        sns.displot(my_frame, x="Rank", kde=True)
        return None, None

    def attack(self, max_iters=20):
        G_copy = copy(self._graph_storage)
        G_size = self._graph_storage.number_of_nodes()

        centr = nx.degree_centrality(self._graph_storage)
        dcs = pd.Series(centr)
        dcs.sort_values(ascending=False, inplace=True)

        removal_propor = []
        diameter_hist = []

        current_removal_propor = 0.0

        dcs = dcs.index.values.tolist()

        for i in range(len(dcs[:max_iters])):
            G_copy.remove_node(dcs[i])

            comps = list((G_copy.subgraph(c) for c in nx.connected_components(self._graph_storage)))
            diameter_hist.append(mean([nx.diameter(comp.to_undirected()) for comp in comps]))

            current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
            removal_propor.append(current_removal_propor)

        # create a figure
        figure = self.__get_figure()

        # create axes
        axes = figure.add_subplot()

        # create the barchart
        axes.plot(removal_propor, diameter_hist)
        axes.set_title('Attack simulation')
        axes.set_ylabel('Diameter')

        return figure, None


    def fail(self, max_iters=20):
        G_copy = copy(self._graph_storage)
        G_size = self._graph_storage.number_of_nodes()

        centr = nx.degree_centrality(self._graph_storage)
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

            node_for_removal = dcs.pop(random.randrange(len(dcs)))
            print(f"node_for_removal {node_for_removal}")

            G_copy.remove_node(node_for_removal)

            comps = list((G_copy.subgraph(c) for c in nx.connected_components(self._graph_storage)))

            te = "Size of cc: "
            for comp in comps:
                te += str(comp.number_of_nodes()) + " "
            print(te)

            diameter_hist.append(mean([nx.diameter(comp.to_undirected()) for comp in comps]))

            current_removal_propor = 1 - (G_copy.number_of_nodes()/G_size)
            removal_propor.append(current_removal_propor)

        figure = self.__get_figure()

        # create axes
        axes = figure.add_subplot()

        # create the barchart
        axes.plot(removal_propor, diameter_hist)
        axes.set_title('Fail simulation')
        axes.set_ylabel('Diameter')

        return figure, None


    def epidemy(self, model_params: dict, n: int):
        model = ep.SEIRModel(self._graph_storage)

        cfg = mc.Configuration()

        for item in model_params.items():
            cfg.add_model_parameter(item[0], item[1])

        model.set_initial_status(cfg)
        iterations = model.iteration_bunch(n)

        trends = model.build_trends(iterations)
        viz = DiffusionTrend(model, trends)
        # TODO(11jolek11): Test!
        print(type(viz))
        print(issubclass(viz.__class__, Figure))
        viz.plot("diffusion")
        plt.show()

        return None, None

    def degree_distribution(self):
        degree_sequence = sorted((d for _, d in self._graph_storage.degree()), reverse=True)

        fig = plt.figure("Degree of a random graph", figsize=(6, 4))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = self._graph_storage.subgraph(sorted(nx.connected_components(self._graph_storage), key=len, reverse=True)[0])
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
        return fig, None

    def agl_methods(self, method: str):
        floyd = nx.floyd_warshall_numpy(self._graph_storage)
        linked = linkage(floyd, method=method)
        print(floyd)
        # Dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top')
        plt.show()

        return None, {"floyd": floyd}

    def plot_graph(self):
        self._graph_storage.draw()
        plt.plot()
        plt.show()

        return None, None

    def divisive(self, threshold=3, meth: str = "complete", criterion: str = "distance"):
        adj_matrix = nx.to_numpy_array(self._graph_storage)
        modularity_div = {}

        divisive_clusters = linkage(adj_matrix.T, method=meth)
        divisive_labels = fcluster(divisive_clusters, t=threshold, criterion=criterion)

        for node, cluster in zip(self._graph_storage.nodes, divisive_labels):
            self._graph_storage.nodes[node]['div'] = cluster

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
        pos = nx.kamada_kawai_layout(self._graph_storage)
        cmap = cm.get_cmap('hsv', max(divisive_labels))

        nx.draw(self._graph_storage, pos, node_color=[cmap(i) for i in divisive_labels], with_labels=False, ax=ax1)
        ax1.set_title(f"Metoda podziałowa {meth}")

        dendrogram(divisive_clusters, ax=ax2, no_labels=True)
        ax2.set_title(f"Dendrogram {meth}")
        plt.show()

        modularity_div[meth] = modularity(self._graph_storage, [{node for node, data in self._graph_storage.nodes(data=True) if data['div'] == cluster} for cluster in set(divisive_labels)])
        print(f"Divisive: {modularity_div}")

        return None, {"modularity": modularity_div}


class Gui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.plot_area = ttk.Frame(self.root)
        self.user_area = ttk.Frame(self.root)
        self.figure_canvas = FigureCanvasTkAgg(None, self.plot_area)
        self.plot_subframe = ttk.Frame(self.plot_area)


    # Create/update plot_area
    def create_plot_and_properties_area(self):
        # create FigureCanvasTkAgg object
        self.figure_canvas = FigureCanvasTkAgg(None, self.plot_area)

        # create the toolbar
        NavigationToolbar2Tk(self.figure_canvas, self.plot_area)

        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        self.plot_subframe.pack()

    def update_plot_and_properties_area(self, figure: Figure, data: dict):
        # create FigureCanvasTkAgg object
        self.figure_canvas = FigureCanvasTkAgg(figure, self.plot_area)

        # create the toolbar
        NavigationToolbar2Tk(self.figure_canvas, self.plot_area)

        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        for child in self.plot_subframe.winfo_children():
            child.destroy()

        for prop, value in data:
            temp = ttk.Label(self.plot_subframe, text=f"{prop}: {value}")
            temp.pack()

        self.plot_area.update()

    def create_user_area(self):
        pass

    def build(self):
        self.create_user_area()
        self.create_plot_and_properties_area()
        self.user_area.pack(side=tk.LEFT, fill=tk.Y, expand=1)
        self.plot_area.pack(side=tk.LEFT)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ge = GraphEngine()

    for _ in range(10):
        resp = requests.get("http://127.0.0.1:8000/")

        if resp.status_code != 200:
            raise Exception("Failed connection...")
        ge.add_data(resp.json()["data"])

        ge.degree_distribution()

