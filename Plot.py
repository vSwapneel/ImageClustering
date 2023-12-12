import sys
import umap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statistics import median
import cv2
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

np.set_printoptions(threshold=np.inf)

def k_means(data, k, centroid, max_iters=100):

    centroids = centroid

    for iter in range(max_iters):
        print("Iteration No.: ", iter)
        labels=np.full(data.shape[0], -1)
        for elem_iter, elements in enumerate(data) :
            distances = np.linalg.norm(elements - centroids, axis=1)
            label = np.argmin(distances, axis=0)
            labels[elem_iter] = label
        
        new_centroids = np.array([data[np.array(labels) == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
            
    return centroids, labels


with open('Test Data.txt', 'r', encoding='utf-8') as file:
    _beta_raw_data = file.read()

raw_data = _beta_raw_data.splitlines()

image_vector_array =[]
for images in raw_data :
    parts = [int(value) for value in images.split(',')]
    image_vector_array.append(parts)
    
mean_value_update=[]
for images in image_vector_array:
    mean_value = sum(images) / len(images)
    new_list = [1 if value >= mean_value else 0 for value in images]
    mean_value_update.append(new_list)    

blured_images=[]
for image in image_vector_array :
    blurred_image = gaussian_filter(np.array(image), sigma=2)
    blured_images.append(blurred_image)


scaler = MinMaxScaler()
images_normalised = scaler.fit_transform(blured_images)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=200)
images_reduced = reducer.fit_transform(images_normalised)

silhouette_scores = []

centroids = images_reduced[np.random.choice(range(images_reduced.shape[0]), size=20, replace=False)]

for k in range(2, 21):
    clusters, cluster_labels = k_means(images_reduced, k, centroids[:k], 25)
    silhouette = silhouette_score(images_normalised, cluster_labels)
    silhouette_scores.append(silhouette)

print(silhouette_scores)

plt.plot(range(2, 21), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Values of K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

