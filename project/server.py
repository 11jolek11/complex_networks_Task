from fastapi import FastAPI
from data import DataSource
import pandas as pd
import networkx as nx



app = FastAPI()

data = pd.read_csv("../data/facebook_combined.txt", sep=" ", header=None)
data.rename(
    columns={
        0: "Source",
        1: "Target"},
    inplace=True)
    
TEMP = nx.from_pandas_edgelist(data, "Source", "Target")
handle = DataSource(TEMP, [100, 50, 30], default_sample_size=20)


@app.get("/")
def index():
    return {"data": handle.get_data()}
#  uvicorn server:app --reload