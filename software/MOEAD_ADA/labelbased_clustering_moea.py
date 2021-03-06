import numpy as np
import random
from sklearn.metrics import pairwise_distances
#from sklearn.metrics.pairwise import euclidean_distances
from math import floor
from sklearn.cluster import KMeans
import copy as cp



# Normaliza una matriz de datos para que los elementos de cada fila sumen 1
def normalize(dataset):
	norm_dataset = np.zeros(dataset.shape)
	for i in range(dataset.shape[0]):
		norm_dataset[i] = dataset[i] / np.sum(dataset[i])
	return norm_dataset



#############################################################################
#############################################################################
# Clase para MOEA con esquema de representación basado en etiquetas (enteros)
#############################################################################
#############################################################################


class LabelBasedClusteringMOEA:
	
	"""
	def __init__(self, data, num_clusts, ml_constraints, cl_constraints,
	num_subproblems, lambda_neighborhood_size, objective_functions, obj_functs_args, 
	prob_mutation, prob_crossover, ref_z, crossover_operator,
	decomposition_method, maximization_problem, random_seed):
	"""	
	def __init__(self, data, num_clusts, ml_constraints, cl_constraints,
	num_subproblems, lambda_neighborhood_size, objective_functions, obj_functs_args,
	prob_mutation, prob_crossover = 0.5, ref_z = None, kmeans_init_ratio=0.0, theta_penalty=0.1,
	crossover_operator="uniform", decomposition_method="Tchebycheff",
	maximization_problem=False, random_seed=None):
		
		self._data = data
		self._ml = ml_constraints
		self._cl = cl_constraints
		self._decomposition_method = decomposition_method
		self._objective_functions = objective_functions
		self._obj_functs_args = obj_functs_args
		self._prob_mutation = prob_mutation
		self._prob_crossover = prob_crossover
		self._num_clusts = num_clusts
		self._maximization = maximization_problem
		self._crossover_operator = crossover_operator
		
		# N
		self._population_size = num_subproblems
		# D (número de instancias en el dataset)
		self._dimensionality = data.shape[0]
		# m
		self._num_objectives = len(objective_functions) # objective_functions.size
		# T (tamaño del vecindario de lambdas)
		self._lambda_neighborhood_size = lambda_neighborhood_size
		
		# Vector ideal z
		self._z = ref_z
		
		# Proporción de la población inicializada con k-means
		self._kmeans_init_ratio = kmeans_init_ratio
		
		# Parámetro de penalización de Boundary Intersection
		self._theta_penalty = theta_penalty # NO USADO
		
		# Semilla aleatoria
		self._random_seed = random_seed
	
	
###############################################################################
###############################################################################


	# Obtener vector de funciones objetivo (f-values) de una solución x
	def f(self, x):
		if len(set(x)) < 2:
			if self._maximization:
				fill_value = -9999.999
			else:
				fill_value = 9999.999
			return np.full(len(self._objective_functions),fill_value)
		
		self._evals += 1
		# Cada función objetivo debe ejecutarse con su correspondiente tupla de parámetros
		return np.array([obj_function(self._data, x, *funct_args) for obj_function,funct_args in zip(self._objective_functions,self._obj_functs_args)])
	

###############################################################################
###############################################################################
	
	
	# Método de descomposición de suma ponderada con vector lambda
	def weighted_sum_approach(self, f_values, weights_vector):
		return np.dot(f_values, weights_vector)

###############################################################################
###############################################################################
	
	
	# Método de descomposición de Tchebycheff
	def tchebycheff_approach(self, f_values, weights_vector):
		
		# Z NO NORMALIZADO
		z_dif = np.abs(f_values-self._z)
		dist = weights_vector * z_dif

		return np.amax(dist)
	
	
###############################################################################
###############################################################################

	def boundary_intersection_approach(self, f_values, weights_vector):
		if self._maximization:
			d1 = np.linalg.norm((self._z-f_values)*weights_vector)/np.linalg.norm(weights_vector)
			d2 = np.linalg.norm(f_values-(self._z-d1*weights_vector))

		else:
			d1 = np.linalg.norm((f_values-self._z)*weights_vector)/np.linalg.norm(weights_vector)
			d2 = np.linalg.norm(f_values-(self._z+d1*weights_vector))
		return d1 + self._theta_penalty*d2


###############################################################################
###############################################################################


	def scalarizing_function(self, fx, weights):
		if self._decomposition_method == "Tchebycheff" or self._decomposition_method == "tchebycheff" or self._decomposition_method == "te":
			return self.tchebycheff_approach(fx, weights)
		elif self._decomposition_method == "Weighted Sum" or self._decomposition_method == "weighted_sum" or self._decomposition_method == "weighted-sum" or self._decomposition_method == "ws":
			return self.weighted_sum_approach(fx, weights)
		elif self._decomposition_method == "Boundary Intersection" or self._decomposition_method == "boundary_intersection" or self._decomposition_method == "bi":
			return self.boundary_intersection_approach(fx, weights)
		else:
			print("MÉTODO DE DESCOMPOSICIÓN INCORRECTO. USANDO TCHEBYCHEFF")
			return self.tchebycheff_approach(fx, weights)




###############################################################################
###############################################################################
	
	
	# Operador de cruce en un punto
	def one_point_crossover(self, parent_a, parent_b):
		point = random.choice(range(parent_a.size))
		offspring_1 = np.copy(parent_a)
		offspring_2 = np.copy(parent_a)
		
		offspring_1[point:] = parent_b[point:]
		offspring_2[0:point] = parent_b[0:point]
		
		return random.choice([offspring_1, offspring_2])
	
	
###############################################################################
###############################################################################
	
	
	# Operador de cruce uniforme
	def uniform_crossover(self, parent_a, parent_b):
		probabilities = np.random.rand(parent_a.size) # np.random.uniform(size=parent_a.size)
		return np.where(probabilities > self._prob_crossover, parent_a, parent_b)
		


###############################################################################
###############################################################################

	"""
	def clustering_oriented_crossover_operator(self, parent1, parent2):

		nb_copied_clusters = 1
		nb_copied_clusters += np.random.randint(0, np.ceil(len(set(parent1)) / 2))

		copied_clusters_labels = random.sample(list(set(parent1)), nb_copied_clusters)

		indices = []
		affected_clusters = []
		offspring = np.zeros(self._dimensionality, dtype = np.int8)

		for i in range(len(parent1)):

			if parent1[i] in copied_clusters_labels:

				offspring[i] = parent1[i]
				indices.append(i)
				affected_clusters.append(parent2[i])

		not_affected_clusters = set(parent2).difference(set(affected_clusters))

		if len(not_affected_clusters) > 0:

			for i in not_affected_clusters:
				not_affected_cluster_indices = np.where(parent2 == i)
				offspring[not_affected_cluster_indices] = -i

		offspring_clusters_labels = list(set(offspring))
		centroids = np.zeros((len(set(offspring)), self._data.shape[1]))

		for i in range(len(offspring_clusters_labels)):

			cluster_instances = self._data[np.where(offspring == offspring_clusters_labels[i])[0]]
			centroids[i,:] = np.mean(cluster_instances, axis = 0)

		unasigned_instances = np.where(offspring == 0)[0]

		for i in unasigned_instances:

			distances = np.sqrt(np.sum((centroids - self._data[i,:])**2, axis = 1))
			offspring[i] = offspring_clusters_labels[np.argmin(distances)]

		offspring_labels = list(set(offspring))
		final_offspring = np.zeros(len(offspring), dtype = np.int8)

		for i in range(len(offspring_labels)):

			final_offspring[offspring == offspring_labels[i]] = i + 1

		return np.array(final_offspring, dtype = np.int8)	
	
	"""
###############################################################################
###############################################################################
	
	
	# Operador de cruce (elige entre los que hay disponibles???)
	def crossover(self, parent_a, parent_b):
		if self._crossover_operator == "uniform":
			return self.uniform_crossover(parent_a, parent_b)
		elif self._crossover_operator == "one_point" or self._crossover_operator == "one point" or self._crossover_operator == "one-point":
			return self.one_point_crossover(parent_a, parent_b)
		#elif self._crossover_operator == "clustering_oriented" or self._crossover_operator == "clustering oriented" or self._crossover_operator == "clustering-oriented":
		#	return self.clustering_oriented_crossover_operator(parent_a, parent_b)
		else:
			print("AVISO: EL OPERADOR DE CRUCE NO ES VÁLIDO")
			print("UTILIZANDO CRUCE UNIFORME")
			return self.uniform_crossover(parent_a, parent_b)
	
	
###############################################################################
###############################################################################

	
	# Operador de mutación (basado en etiquetas)
	def mutation(self, chromosome):
		
		# 1ª variante:
		#  - Se obtiene un valor entre 0 y 1 por cada gen (posición) del cromosoma (array)
		#  - Se determinan las posiciones cuyo valor sea menor que el valor de probabilidad de mutación (es un atributo de la clase)
		#  - Se generan nuevos valores de etiquetas (genes) aleatorios, tantos como valores menores que la probabilidad de mutación
		#  - A las posiciones cuyo correspondiente valor entre 0 y 1 era menor que la prob. de mutación, se le asignan los nuevos genes generados
		"""
		probs = np.random.uniform(size=chromosome.size)
		mutate = probs < self._prob_mutation
		new_genes = np.random.randint(0, self._num_clusts, np.sum(mutate))
		
		mutated_chromosome = np.copy(chromosome)
		mutated_chromosome[mutate] = new_genes
		"""
		
		# 2ª variante:
		#  - Se obtienen aleatoriamente las n posiciones que se mutarán, siendo n = x prob. de mutación
		#  - Se obtienen directamente los nuevos genes, generando n valores aleatorios 
		#  - Se asignan los n genes aleatorios generados a las n posiciones aleatorias
		mutate = random.sample(range(0, self._dimensionality), int(self._dimensionality * self._prob_mutation))
		new_genes = np.random.randint(0, self._num_clusts, (1, int(self._dimensionality * self._prob_mutation)))

		mutated_chromosome = np.copy(chromosome)
		mutated_chromosome[mutate] = new_genes
		
		
		return mutated_chromosome
	

###############################################################################
###############################################################################
	
	
	# Actualización de la población externa:
	# Comprueba si un nuevo individuo (cromosoma + valores objetivo) 
	# es dominado por alguno de los que ya están en la población externa.
	# El nuevo individuo es añadido si no es dominado por ninguno.
	# Además, todos los dominados por el nuevo son eliminados previamente
	def update_external_population(self, chromosome, f_values):
		######################## EP COMO LISTA ################################
		"""
		# Eliminar soluciones dominadas por la nueva
		dominated_sols = []
		for solution,i in zip(self._EP,len(self._EP)):
			if self.dominates(f_values, solution):
				dominated_sols.append(i)
		for index in dominated_sols[::-1]:
			self._EP.pop(index)
			self._EP_chromosomes.pop(index)
			
		
		# Añadir nueva solución a EP si no es dominada
		# por ninguna de las ya existentes
		if not any(self.dominates(solution,f_values) for solution in self._EP):
			self._EP.append(f_values)
			self._EP_chromosomes.append(chromosome)
		"""
		######################## EP COMO ARRAY ################################
		if self._EP.size == 0:
			# Añadir nueva solución a EP
			self._EP = np.vstack((self._EP, f_values))
			self._EP_chromosomes = np.vstack((self._EP_chromosomes, chromosome))
			self._last_EP_update_eval = self._evals
			self._last_EP_update_epoch = self._epochs

		else:
			# Comprobar si la nueva solución ya estaba presente en EP,
			# comparando los cromosomas
			eq_chrom_count = np.sum(self._EP_chromosomes == chromosome, axis = 1)
			eq_chrom_flags = eq_chrom_count == self._dimensionality
					
			already_exists = eq_chrom_flags.any()
			
			if not already_exists:
			
				if self._maximization:
					# Detectar primero si la solución ya está en EP (tanto fv como cromosoma)
					
					# Eliminar soluciones dominadas por la nueva
					ge_count = np.sum(f_values >= self._EP, axis = 1)
					gt_count = np.sum(f_values > self._EP, axis = 1)
	
					ge_flags = ge_count == self._num_objectives
					gt_flags = gt_count >= 1
					
					dominated = np.logical_and(ge_flags, gt_flags)
					
					self._EP = np.delete(self._EP, dominated, 0)
					self._EP_chromosomes = np.delete(self._EP_chromosomes, dominated, 0)
					
					# Añadir nueva solución a EP si no es dominada o no pertenece ya a la población
					ge_count = np.sum(self._EP >= f_values, axis = 1)
					gt_count = np.sum(self._EP > f_values, axis = 1)
					
					ge_flags = ge_count == self._num_objectives
					gt_flags = gt_count >= 1
					
					is_dominated = np.logical_and(ge_flags, gt_flags).any()
					
					if not is_dominated:
						# Añadir nueva solución a EP
						self._EP = np.vstack((self._EP, f_values))
						self._EP_chromosomes = np.vstack((self._EP_chromosomes, chromosome))
						self._last_EP_update_eval = self._evals
						self._last_EP_update_epoch = self._epochs
						
				else:
					# Detectar primero si la solución ya está en EP (tanto fv como cromosoma)
					
					# Eliminar soluciones dominadas por la nueva
					le_count = np.sum(f_values <= self._EP, axis = 1)
					lt_count = np.sum(f_values < self._EP, axis = 1)
					
					le_flags = le_count == self._num_objectives
					lt_flags = lt_count >= 1
					
					dominated = np.logical_and(le_flags, lt_flags)
					
					self._EP = np.delete(self._EP, dominated, 0)
					self._EP_chromosomes = np.delete(self._EP_chromosomes, dominated, 0)
					
					# Añadir nueva solución a EP si no es dominada
					le_count = np.sum(self._EP <= f_values, axis = 1)
					lt_count = np.sum(self._EP < f_values, axis = 1)
					
					le_flags = le_count == self._num_objectives
					lt_flags = lt_count >= 1
					
					is_dominated = np.logical_and(le_flags, lt_flags).any()
					
					if not is_dominated:
						# Añadir nueva solución a EP
						self._EP = np.vstack((self._EP, f_values))
						self._EP_chromosomes = np.vstack((self._EP_chromosomes, chromosome))
						self._last_EP_update_eval = self._evals
						self._last_EP_update_epoch = self._epochs
	
	


###############################################################################
###############################################################################

	
	# Compara vectores solución teniendo en cuenta el método de descomposición
	# y si el problema de optimización consiste en maximizar o minimizar
	def compare_decomposition_values(self, a, b, lambda_weight_vector):
		if self._decomposition_method == "Tchebycheff" or self._decomposition_method == "tchebycheff" or self._decomposition_method == "te":
			return self.tchebycheff_approach(a, lambda_weight_vector) <= self.tchebycheff_approach(b, lambda_weight_vector)
		elif self._decomposition_method == "Weighted Sum" or self._decomposition_method == "weighted_sum" or self._decomposition_method == "weighted-sum" or self._decomposition_method == "ws":
			return self.weighted_sum_approach(a, lambda_weight_vector) >= self.weighted_sum_approach(b, lambda_weight_vector)
		elif self._decomposition_method == "Boundary Intersection" or self._decomposition_method == "boundary_intersection" or self._decomposition_method == "bi":
			return self.boundary_intersection_approach(a, lambda_weight_vector) <= self.boundary_intersection_approach(b, lambda_weight_vector)
		else:
			# Por defecto: Tchebycheff
			#print("Por defecto: Tchebycheff")
			return self.tchebycheff_approach(a, lambda_weight_vector) <= self.tchebycheff_approach(b, lambda_weight_vector)
	

###############################################################################
###############################################################################

	
	
	# Inicializar población
	def initialization(self):
		# Número de evaluaciones/épocas
		self._epochs = 0
		self._evals = 0
		self._last_EP_update_eval = -1
		self._last_EP_update_epoch = -1
		
		# Inicializar semilla aleatoria
		if self._random_seed is not None:
			random.seed(self._random_seed)	
			np.random.seed(self._random_seed)
			#print("Semilla {}".format(self._random_seed)) # DEBUG
		
		# Población externa
		self._EP = np.empty(shape=(0, self._num_objectives))
		self._EP_chromosomes = np.empty(shape=(0, self._dimensionality))

		
		# Población de individuos
		self._population = np.random.randint(0, self._num_clusts, size=(self._population_size, self._dimensionality))  # COMPROBADO 
		if self._kmeans_init_ratio > 0.0:
			initialized_with_kmeans = random.sample(range(0, self._population_size), floor(self._population_size * self._kmeans_init_ratio))
	
			for i in initialized_with_kmeans:
				self._population[i] = KMeans(n_clusters = self._num_clusts, max_iter = np.random.randint(10,20)).fit_predict(self._data)#.labels_

		
		
		# Vectores solución (f-values) de todos los individuos de la población	
		self._FV = np.empty((self._population_size, self._num_objectives))
		for i in range(self._population_size):
			self._FV[i] = self.f(self._population[i])
		
		
		# Punto de referencia z (mejor valor obtenido con cada función objetivo)
		if self._z is None:
			if self._maximization:
				self._z = np.amax(self._FV, axis=0)
			else:
				self._z = np.amin(self._FV, axis=0)
		
		# Punto de referencia z-worst: el peor valor obtenido con cada función objetivo
		if self._maximization:
			self._z_worst = np.amin(self._FV, axis=0)
		else:
			self._z_worst = np.amax(self._FV, axis=0)

				
		# Vectores de pesos lambda
		# SUS ELEMENTOS DEBEN SUMAR 1
		self._lambdas = normalize(np.random.randint(low = 1, high = 99,
									size = (self._population_size, self._num_objectives)))
		
		# INICIALIZACIÓN DEL VECINDARIO
		# Matriz de distancias de los vectores lambda
		lambdas_distances = pairwise_distances(self._lambdas, Y=None, metric='euclidean')
		# Vecindario de cada vector de pesos lambda-i
		self._lambda_neighborhood = lambdas_distances.argsort(axis = 1)[:,0:self._lambda_neighborhood_size] # COMPROBADO
		

###############################################################################
###############################################################################

	# Actualizar vectores de referencia z y z_worst
	def update_z(self, f_vector):
		if self._maximization:
			self._z = np.maximum(self._z, f_vector)
			self._z_worst = np.minimum(self._z_worst, f_vector)
		else:
			self._z = np.minimum(self._z, f_vector)
			self._z_worst = np.maximum(self._z_worst, f_vector)

###############################################################################
###############################################################################
	
	
	def normalize_fv(self, new_obj_vector):
		# Cambia la forma en que se calcula según sea maximización o minimización
		# Primero, se debe actualizar vect. de referencia z
		
		if self._maximization:
			norm_obj_vector = (new_obj_vector-self._z_worst)/(self._z-self._z_worst)
			norm_FV = (self._FV-self._z_worst)/(self._z-self._z_worst)

		else:
			norm_obj_vector = (new_obj_vector-self._z)/(self._z_worst-self._z)
			norm_FV = (self._FV-self._z)/(self._z_worst-self._z)

		return norm_obj_vector, norm_FV
	
	

###############################################################################
###############################################################################

	def normalize_vector(self, new_obj_vector):
		# Cambia la forma en que se calcula según sea maximización o minimización
		# Primero, se debe actualizar vect. de referencia z
		
		if self._maximization:						
			norm_obj_vector = (new_obj_vector-self._z_worst)/(self._z-self._z_worst)
		else:
			norm_obj_vector = (new_obj_vector-self._z)/(self._z_worst-self._z)
		
		return norm_obj_vector


###############################################################################
###############################################################################



