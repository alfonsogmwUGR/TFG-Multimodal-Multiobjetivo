import numpy as np
import random
from MOEAD_ADA.labelbased_clustering_moea import LabelBasedClusteringMOEA
#from sklearn.metrics import pairwise_distances
#from sklearn.metrics.pairwise import euclidean_distances


# Clase para MOEA con esquema de representación basado en grafos (enteros)
# Hereda directamente de la clase

class GraphBasedClusteringMOEA(LabelBasedClusteringMOEA):
	
	def __init__(self, data, num_clusts, ml_constraints, cl_constraints,
	num_subproblems, neighborhood_size, objective_functions, obj_functs_args,
	prob_mutation, prob_crossover = 0.5, ref_z = None,
	crossover_operator="uniform", decomposition_method="Tchebycheff",
	maximization_problem=True):
		LabelBasedClusteringMOEA.__init__(self, data, num_clusts, ml_constraints,
		cl_constraints, num_subproblems, neighborhood_size, objective_functions,
		obj_functs_args, prob_mutation, prob_crossover, ref_z, crossover_operator,
		decomposition_method, maximization_problem)
		
		self._num_subproblems = num_subproblems
		
		
###############################################################################
###############################################################################


	def decode_graph(self, graph_list):

		graph_list = np.array(graph_list, dtype = np.int16)
		visited = np.zeros(self._dim, dtype = np.bool)
		labels = np.array([-1]*self._dim, dtype = np.int16)
		label = 0

		for i in range(self._dim):

			if not visited[i]:

				current = i
				subcluster = []

				while not visited[current]:

					subcluster.append(current)
					visited[current] = True
					current = graph_list[current]

				if labels[current] == -1:

					labels[subcluster] = label
					label += 1

				else:

					labels[subcluster] = labels[current]

		return labels
	
	
###############################################################################
###############################################################################
	

	# Operador de cruce (elige entre los que hay disponibles)
	def crossover(self, parent_a, parent_b):
		if self._crossover_operator == "uniform":
			return self.uniform_crossover(parent_a, parent_b)
		elif self._crossover_operator == "one_point" or self._crossover_operator == "one point" or self._crossover_operator == "one-point":
			return self.one_point_crossover(parent_a, parent_b)
		else:
			print("AVISO: EL OPERADOR DE CRUCE NO ES VÁLIDO")
			print("UTILIZANDO CRUCE UNIFORME")
			return self.uniform_crossover(parent_a, parent_b)
		
###############################################################################
###############################################################################
	
	def f(self, x):
		decoded_labels = self.decode_graph(x)
		f_values = LabelBasedClusteringMOEA.f(decoded_labels)
		return f_values

###############################################################################
###############################################################################
	
	def initialization(self):
		LabelBasedClusteringMOEA.initialization()
		# Volvemos a definir a la población y la matrix para los cromosomas de la EP
		self._population = np.random.randint(0, self._dimemsionality, (self._population_size, self._dimemsionality))
		self._EP_chromosomes = np.empty((0,self._dimemsionality))

###############################################################################
###############################################################################

	# Decodifica todos los grafos (cromosomas) que hay en la población externa (EP)
	def get_EP_decoded_labels(self):
		EP_decoded_labels = np.zeros((self._EP.shape[0], self._dimemsionality))
		for i in range(self._EP.shape[0]):
			EP_decoded_labels[i] = self.decode_graph(self._EP_chromosomes[i])
		return EP_decoded_labels
			