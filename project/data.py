import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


class DataSource:
    def __init__(self, G: nx.Graph, sample_sizes: list[int], default_sample_size: int = 1) -> None:
        self.graph = list(G.edges)
        self.sample_sizes = sample_sizes
        self._default_sample_size = default_sample_size

        self._generator = self._generator()

    def _generator(self):
        while self.graph:
            if not self.sample_sizes:
                sample_size = self._default_sample_size
            else:
                sample_size = self.sample_sizes.pop()
            
            yield [self.graph.pop(random.randrange(len(self.graph))) for _ in range(sample_size)]
    
    def get_data(self):
        return next(self._generator)


if __name__ == "__main__":
    data = pd.read_csv("../data/facebook_combined.txt", sep=" ", header=None)
    data.rename(
        columns={
            0: "Source",
            1: "Target"},
        inplace=True)
        
    TEMP = nx.from_pandas_edgelist(data, "Source", "Target")

    handle = DataSource(TEMP, [100, 50, 30], default_sample_size=20)

    # G_sample = nx.Graph()

    # for _ in range(200):
    #     G_sample.add_edges_from(handle.get_data())

    #     # https://brandonrozek.com/blog/networkx-random-sample-graph/

    #     nx.draw(G_sample, with_labels=True)

    #     # plt.show()
    # plt.plot()
    # plt.show()

    # print(nx.utils.graphs_equal(TEMP, G_sample))
    # print(nx.algorithms.graph_edit_distance(TEMP, G_sample))
    # plt.plot(test_ged)
    # plt.show()
