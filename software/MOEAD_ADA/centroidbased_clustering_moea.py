import numpy as np
import random
from MOEAD_ADA.labelbased_clustering_moea import LabelBasedClusteringMOEA
#from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

#############################################################################
#############################################################################
# Clase para MOEA con esquema de representación basado en centroides (reales)
#############################################################################
#############################################################################

class CentroidBasedClusteringMOEA(LabelBasedClusteringMOEA):
	
	def __init__(self, data, num_clusts, ml_constraints, cl_constraints,
	num_subproblems, neighborhood_size, objective_functions, obj_functs_args,
	prob_mutation, prob_crossover = 0.5, ref_z = None,
	crossover_operator="uniform", decomposition_method="Tchebycheff",
	maximization_problem=True, alpha=0.5):
		
		LabelBasedClusteringMOEA.__init__(self, data, num_clusts, ml_constraints,
		cl_constraints, num_subproblems, neighborhood_size, objective_functions,
		obj_functs_args, prob_mutation, prob_crossover, ref_z, crossover_operator,
		decomposition_method, maximization_problem)
		
		self._data_dimensionality = data.shape[1]  # data_dimensionality =! dimensionality
		
		# Parámetro alpha para el cruce BLX-alpha
		self._alpha = alpha
		
		
###############################################################################
###############################################################################

	# Asignar instancias a cada uno de los centroides, en base a su cercanía
	def assign_labels(self):
		labels = np.zeros((self._population.shape[0], self._dimensionality))
		for i in range(self._dimensionality):
			# Calcular las distancias de cada punto del dataset
			# con todos los centroides de la población
			data_point = np.reshape(self._data[i],(1,self._data[i].size))
			distances_to_centroids = euclidean_distances(data_point, self._population)
			# Etiqueta (label) corresponde al centroide más cercano
			labels[i] = np.argmin(distances_to_centroids)
		
		return labels

###############################################################################
###############################################################################

	
	# Cruce lineal o aritmético
	def arithmetical_crossover(self, parent_a, parent_b):
		# Se generan tres descendientes
		#(1/2)p1 + (1/2)p2, (3/2)p1 - (1/2)p2, (-1/2)p1+(3/2)p2
		offspring_a = 0.5*parent_a + 0.5*parent_b
		offspring_b = 1.5*parent_a - 0.5*parent_b
		offspring_c = -0.5*parent_a + 1.5*parent_b
		
		# SELECIÓN DE DESCENDIENTE DETERMINISTA:
		# se elige al descendiente que mejor valor objetivo (f-value) tenga
		best_offspring = np.copy(offspring_a)
		best_fv = self.f(offspring_a)
		
		fv_b = self.f(offspring_b)
		fv_c = self.f(offspring_c)
		
		if self._maximization:
			if best_fv > fv_b:
				best_offspring = np.copy(offspring_b)
				best_fv = fv_b
			if best_fv > fv_c:
				best_offspring = np.copy(offspring_c)
		else:
			if best_fv < fv_b:
				best_offspring = np.copy(offspring_b)
				best_fv = fv_b
			if best_fv < fv_c:
				best_offspring = np.copy(offspring_c)
		
		return best_offspring
				
		# SELECIÓN DE DESCENDIENTE NO DETERMINISTA:
		# se elige aleatoriamente a uno de los tres descendientes
		"""
		return random.choice([offspring_a,offspring_b,offspring_c])
		"""
		
		
###############################################################################
###############################################################################			
			
	
	# BLX-alpha
	def blend_crossover(self, parent_a, parent_b):
		h_max = np.maximum(parent_a, parent_b)
		h_min = np.minimum(parent_a, parent_b)
		
		interval = h_max - h_min
		
		low_vals = h_min - interval*self._alpha
		high_vals = h_max + interval*self._alpha
		
		return np.uniform(low_vals, high_vals, parent_a.size)
	
	
###############################################################################
###############################################################################
	

	# Operador de cruce (se ejecutará uno de los definidos previamente)
	def crossover(self, parent_a, parent_b):
		if self._crossover_operator == "blx-alpha" or self._crossover_operator == "blend":
			return self.blend_crossover(parent_a, parent_b)
		elif self._crossover_operator == "arithmetical" or self._crossover_operator == "arithmetic" or self._crossover_operator == "linear":
			return self.arithmetical_crossover(parent_a, parent_b)
		# Por defecto: BLX-alpha
		return self.blend_crossover(parent_a, parent_b)
	
	
###############################################################################
###############################################################################
	
	# Operador de mutación
	def mutation(self, chromosome):
		pass


###############################################################################
###############################################################################


	def initialization(self):
		LabelBasedClusteringMOEA.initialization()
		# Volvemos a definir a la población y la matrix para los cromosomas de la EP
		self._population = np.random.uniform(0, self._dimemsionality, (self._population_size, self._data_dimensionality))
		self._EP_chromosomes = np.empty((0,self._data_dimensionality))

###############################################################################
###############################################################################
	
	# Obtiene los labes de todos los centroides (cromosomas)
	# que hay en la población externa (EP)
	def get_EP_labels(self):
		EP_labels = np.zeros((self._EP.shape[0], self._data_dimensionality))
		for i in range(self._EP.shape[0]):
			EP_labels[i] = self.assign_labels(self._EP_chromosomes[i])
		return EP_labels