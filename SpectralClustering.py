from sklearn.cluster import SpectralClustering
import Data2
import matplotlib.pyplot as plt
import random
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import homogeneity_score

X,y = Data2.GetData()

kmeans = SpectralClustering(n_clusters=8).fit(X)
labels = kmeans.labels_

number_of_colors = len(list(set(labels)))
color_samples = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]

color_vet = []
for label in labels:
    color_vet.append(color_samples[label])

plotx = (X[:,18] + X[:,19] + X[:,20])
ploty = (X[:,21] + X[:,22])

print("Davies Bouldin Score: ",str(davies_bouldin_score(X,labels)))
print("Silhouette Score: ",str(silhouette_score(X,labels)))
print("Mutual Score: ",str(mutual_info_score(y,labels)))
print("homogeneity Score: ",str(homogeneity_score(y,labels)))

plt.figure(figsize=(9,9))
note = "Davies Bouldin Score: " + str(davies_bouldin_score(X,labels)) + "\n"
note += "Silhouette Score: " + str(silhouette_score(X,labels)) + "\n"
note += "Mutual Score: " + str(mutual_info_score(y,labels)) + "\n"
note += "homogeneity Score: " + str(homogeneity_score(y,labels))
plt.figtext(x=0.01,y=0.001,s=note)
plt.scatter(plotx,ploty,c=color_vet)
plt.show()
