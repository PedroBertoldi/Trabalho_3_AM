import pandas
import numpy as np

def GetData(n_lines=None):
    data_set = pandas.read_csv("Pokemon.csv")

    drop_cols = ['Type 2', 'Generation', 'Legendary', '#']

    tipos = data_set["Type 1"].isin(["Grass","Fire","Rock","Water","Psychic"])
    data_set = data_set[tipos].drop(columns=drop_cols)


    X = data_set[["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]].to_numpy()

    temp_dict = {}
    counter = 0
    for tipo in data_set["Type 1"].unique():
        temp_dict[tipo]=counter
        counter+=1

    data_set["Type 1"] = data_set["Type 1"].apply(lambda x: temp_dict[x])
    y = data_set["Type 1"].to_numpy()
    return X,y

if __name__ == "__main__":
    GetData()