import sys
import umap
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import cv2
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

np.set_printoptions(threshold=np.inf)

def k_means(data, k, max_iters=100):
    centroids = data[np.random.choice(range(data.shape[0]), size=k, replace=False)]

    with open("Initial Centroid.txt", 'w') as output_file:
        sys.stdout = output_file
        print(centroids)
        sys.stdout = sys.__stdout__

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

# blured_images=[]
# for image in image_vector_array :
#     blurred_image = gaussian_filter(np.array(image), sigma=2)
#     blured_images.append(blurred_image)

scaler = MinMaxScaler()
images_normalised = scaler.fit_transform(mean_value_update)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=100)
image_reduced = reducer.fit_transform(images_normalised)

clusters, cluster_labels = k_means(np.array(image_reduced),10, 10)

silhouette_avg = silhouette_score(image_reduced, cluster_labels)

print(cluster_labels)

plt.scatter(image_reduced[:, 0], image_reduced[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(clusters[:, 0], clusters[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title(f'Custom K-means Clustering for Digit Recognition\nBest Silhouette Score: {silhouette_avg:.2f}')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend()
plt.show()

with open('Pred Output.dat', 'w') as pred_output_file_path:
    sys.stdout = pred_output_file_path 
    for i, num in enumerate(cluster_labels):
        if i != len(cluster_labels)-1:
            pred_output_file_path.write(str(num+1) + "\n")
        else:
            pred_output_file_path.write(str(num+1))
    sys.stdout = sys.__stdout__

