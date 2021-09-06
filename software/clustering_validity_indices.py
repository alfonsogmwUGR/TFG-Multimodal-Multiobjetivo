import numpy as np
import random
from sklearn.metrics import pairwise_distances



# Índice de conectividadde una partición de clustering (dadas sus etiquetas)
# MINIMIZACIÓN
def connectedness(data, labels, data_neighborhoods):
	connectedness = 0
	neighborhood_size = data_neighborhoods.shape[1]
	
	for i in range(labels.size):
		for j in range(neighborhood_size):
			if(labels[i] != labels[int(data_neighborhoods[i,j])]):
				connectedness += 1/(j+1)
				
	return connectedness




# Índice de Davies–Bouldin
def davies_bouldin_index(data, labels):
	# Inicializamos la distancia media de las instancias de los clusters
	mean_distances = []
	mean_distances_sum = 0
	#Obtenemos el numero de clusters del clustering actual
	nb_clusters = len(set(labels))

	# Para cada cluster en el clustering actual
	for j in set(labels):
		# Obtener las instancias asociadas al cluster
		clust = data[labels == j, :]

		if clust.shape[0] > 1:

			#Obtenemos la distancia media intra-cluster
			tot = 0.
			for k in range(clust.shape[0] - 1):
				tot += ((((clust[k + 1:] - clust[k]) ** 2).sum(1)) ** .5).sum()

			avg = tot / ((clust.shape[0] - 1) * (clust.shape[0]) / 2.)

			# Acumular la distancia media
			mean_distances.append(avg)
			mean_distances_sum += avg

		else:

			mean_distances.append(0)


	#overall_deviation = mean_distances_sum / nb_clusters

	#Calculo de la medida de Davis-Bouldin
	centroids = np.empty((nb_clusters, data.shape[1]))
	labels_list = list(set(labels))

	for i in range(len(labels_list)):

		clust = data[labels == labels_list[i], :]

		centroids[i, :] = np.mean(clust, axis = 0)

	centroid_distances = pairwise_distances(centroids, Y=None, metric='euclidean')

	R_sum = 0

	for i in range(centroids.shape[0]):

		R_max = -1

		for j in range(centroids.shape[0]):

			if i != j:

				if(centroid_distances[i,j] == 0):
					centroid_distances[i,j] = 0.0001

				R = (mean_distances[i] + mean_distances[j]) / centroid_distances[i,j]

				if R > R_max:
					R_max = R

		R_sum += R_max

	DB = R_sum / nb_clusters
	return DB



# Número de restricciones must-link y cannot-link incumplidas en una partición de clustering
# (dadas sus etiquetas)
# MINIMIZACIÓN
def clustering_infeasibility(data, labels, must_link_constrs, cannot_links_constrs):
	infeasibility = 0
	
	# Calculamos el numero de restricciones must-link que no se satisfacen
	for c in range(len(must_link_constrs)):
		if labels[must_link_constrs[c][0]] != labels[must_link_constrs[c][1]]:
			infeasibility += 1
	
	# Calculamos el numero de restricciones cannot-link que no se satisfacen
	for c in range(len(cannot_links_constrs)):
		if labels[cannot_links_constrs[c][0]] == labels[cannot_links_constrs[c][1]]:
			infeasibility += 1
	
	return infeasibility



# Número de restricciones must-link y cannot-link cumplidas
# MAXIMIZACIÓN
def satisfied_constraints(data, labels, must_link_constrs, cannot_links_constrs):
	satisfied = 0
	
	# Calculamos el numero de restricciones must-link que sí se satisfacen
	for c in range(len(must_link_constrs)):
		if labels[must_link_constrs[c][0]] == labels[must_link_constrs[c][1]]:
			satisfied += 1
	
	# Calculamos el numero de restricciones cannot-link que sí se satisfacen
	for c in range(len(cannot_links_constrs)):
		if labels[cannot_links_constrs[c][0]] != labels[cannot_links_constrs[c][1]]:
			satisfied += 1
	
	return int(satisfied)