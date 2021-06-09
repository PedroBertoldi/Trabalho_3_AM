import pandas
import numpy as np

def GetData(n_lines=None):
    data_set = pandas.read_csv("pokemon (1).csv",sep=";")

    colunas = list(data_set.columns)
    colunas.pop()

    X = data_set[colunas].to_numpy()

    temp_dict = {}
    counter = 0
    for tipo in data_set["type1"].unique():
        temp_dict[tipo]=counter
        counter+=1

    data_set["type1"] = data_set["type1"].apply(lambda x: temp_dict[x])
    y = data_set["type1"].to_numpy()
    return X,y

if __name__ == "__main__":
    GetData()