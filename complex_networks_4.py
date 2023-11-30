import pandas as pd
import networkx as nx


def find_cliques(G: nx.Graph):
    return list(nx.enumerate_all_cliques(G))

if __name__ == "__main__":
    data = pd.read_csv("./data/bn-mouse_visual-cortex_2.edges", sep=" ", header=None)
    
    # data = pd.read_csv("./data/higgs-mention_network.edgelist", sep=" ", header=None)

    data.rename(
        columns={
            0: "link1",
            1: "link2"},
        inplace=True)

    TEMP = nx.from_pandas_edgelist(data, "link1", "link2")