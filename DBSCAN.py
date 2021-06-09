from typing import List
from numpy.ma import count
from sklearn.cluster import DBSCAN
import Data2
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import homogeneity_score

def GetModelName():
    name = os.path.basename(__file__).split(".")
    return name[0]
    
def run():
    X,y = Data2.GetData()
    #==============================================================================================
    params = [{"leaf_size" : 30 , "eps" : 3 , "min_samples":5},
              {"leaf_size" : 30 , "eps" : 5 , "min_samples":5},
              {"leaf_size" : 30 , "eps" : 20 , "min_samples":5},
              {"leaf_size" : 30 , "eps" : 20 , "min_samples":10},
              {"leaf_size" : 30 , "eps" : 20 , "min_samples":15},]
    #===============================================================================================
    count = 0
    name = GetModelName()
    if not os.path.exists(name):
        os.makedirs(name)
    for param in params:
        count += 1
        
        #============================================================================================
        labels = DBSCAN(leaf_size=param["leaf_size"],eps=param["eps"],min_samples=param["min_samples"]).fit_predict(X)
        #============================================================================================

        number_of_colors = len(list(set(labels)))
        color_samples = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]

        color_vet = []
        for label in labels:
            color_vet.append(color_samples[label])

        plotx = (X[:,18] + X[:,19] + X[:,20])
        ploty = (X[:,21] + X[:,22])

        plt.figure(figsize=(9,9))
        plt.title("Plot for " + name + " Set nº" + str(count))


        first_note = ""
        for key in param.keys():
            first_note += str(key) + ": " + str(param[key]) + "\n"
        plt.figtext(x=0.01,y=0.9,s=first_note)
        note = "Davies Bouldin Score: " + str(davies_bouldin_score(X,labels)) + "\n"
        note += "Silhouette Score: " + str(silhouette_score(X,labels)) + "\n"
        note += "Mutual Score: " + str(mutual_info_score(y,labels)) + "\n"
        note += "homogeneity Score: " + str(homogeneity_score(y,labels))
        plt.figtext(x=0.01,y=0.001,s=note)
        plt.scatter(plotx,ploty,c=color_vet)
        plt.savefig(name + "/" + name + "_set_" + str(count) + ".png", format="png", pad_inches = 0)
        plt.show()

if __name__ == "__main__":
    run()
