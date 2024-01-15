from fastapi import FastAPI
import pandas as pd
import networkx as nx



app = FastAPI()


@app.get("/")
def index():
    pass



if __name__ == "__main__":
    data = pd.read_csv("../data/facebook_combined.txt", sep=" ", header=None)
    data.rename(
        columns={
            0: "Source",
            1: "Target"},
        inplace=True)
        
    TEMP = nx.from_pandas_edgelist(data, "Source", "Target")