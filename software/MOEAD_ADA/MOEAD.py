"""
	Clase del algoritmo MOEA/D
	(Repr. basada en etiquetas)
"""


import numpy as np
import random
from sklearn.metrics import pairwise_distances
from MOEAD_ADA.labelbased_clustering_moea import LabelBasedClusteringMOEA
import time





class MOEAD(LabelBasedClusteringMOEA):
	
	def __init__(self, data, num_clusts, ml_constraints, cl_constraints,
	num_subproblems, neighborhood_size, objective_functions, obj_functs_args,
	prob_mutation, prob_crossover = 0.5, ref_z = None,
	crossover_operator="uniform", decomposition_method="Tchebycheff",
	maximization_problem=True):
		LabelBasedClusteringMOEA.__init__(self, data, num_clusts, ml_constraints,
		cl_constraints, num_subproblems, neighborhood_size, objective_functions,
		obj_functs_args, prob_mutation, prob_crossover, ref_z, crossover_operator,
		decomposition_method, maximization_problem)
		
		
	

	
###############################################################################
###############################################################################
	
	"""
	# Comprueba si el vector a domina al vector b
	def dominates(self,a,b):
		if self._maximization_problem:
			return (a >= b).all() and (a > b).any()  # COMPROBADO
		else:
			return (a <= b).all() and (a < b).any()  # COMPROBADO
	"""	

	
###############################################################################
###############################################################################
	
	
	# Ejecutar el algoritmo MOEA/D
	def run(self, max_evals):
		# Inicialización
		self.initialization()
		
		# Para controlar tiempo de ejecución
		epoch_times = []
		total_evals_per_epoch = []
		
		while self._evals < max_evals:
			start_epoch = time.time()
			
			# Actualización
			for i in range(self._population_size):
				# Seleccionar aleatoriamente dos vectores del vecindario del
				# i-ésimo vector de pesos (lambda-i)
				#print(i)
				parents = np.random.choice(self._lambda_neighborhood[i], size=2, replace=False)
				#print(parents)
				#print("----------------------------------------------")
				
				# Cruzas los dos vectores obtenidos para generar descendencia
				y = self.crossover(self._population[parents[0]], self._population[parents[1]])
				
				# Mutar el vector resultante
				# (según la probabilidad de mutación)
				y = self.mutation(y)
				
				# Mejora/reparación de y ???
				
				# Actualización del vector referencia z
				f_y = self.f(y)    # Obtener valores objetivo (FV) de y
				if self._maximization:
					self._z = np.maximum(self._z, f_y)
				else:
					self._z = np.minimum(self._z, f_y)
				
				# Actualización de soluciones vecinas
				for n in range(self._neighborhood_size):
					j = self._lambda_neighborhood[n]
					f_xj = self._FV[j]
					lambda_j = self._lambdas[j]
					#if self.decomposition(f_y, lambda_j) < self.decomposition(f_xj, lambda_j):
					if self.compare_decomposition_values(f_y, f_xj, lambda_j):
						self._population[j] = np.copy(y)
						self._FV[j] = np.copy(f_y)
				
				# Actualización de la población externa
				self.update_external_population(y, f_y)
			
			self._epochs += 1
			epoch_times.append(time.time() - start_epoch)
			total_evals_per_epoch.append(self._evals)
			
		#return self._EP, self._EP_chromosomes
		return self._EP, self._EP_chromosomes, self._lambdas, total_evals_per_epoch, np.median(epoch_times) * self._epochs, self._last_EP_update_eval


###############################################################################
###############################################################################




	