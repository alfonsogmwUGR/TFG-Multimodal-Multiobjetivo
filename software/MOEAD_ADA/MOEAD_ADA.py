"""
	Clase del algoritmo MOEA/D adaptado bajo el framework ADA
	(Repr. basada en etiquetas)
"""



import numpy as np
import random
import time
#from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from MOEAD_ADA.labelbased_clustering_moea import LabelBasedClusteringMOEA






class MOEAD_ADA(LabelBasedClusteringMOEA):

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
	
	
	def normalize_fv(self, new_obj_vector):
		# ¿Cambia la forma en que se calcula según sea maximización o minimización? SÍ
		# Primero, se debe actualizar vect. de referencia z
		
		if self._maximization:
			
			self._z = np.maximum(self._z, new_obj_vector)
			self._z_worst = np.minimum(self._z_worst, new_obj_vector)
						
			norm_obj_vector = (new_obj_vector-self._z_worst)/(self._z-self._z_worst)
				
			norm_FV = (self._FV-self._z_worst)/(self._z-self._z_worst) # COMPROBADO

			
		else:
			
			self._z = np.minimum(self._z, new_obj_vector)
			self._z_worst = np.maximum(self._z_worst, new_obj_vector)
						
			norm_obj_vector = (new_obj_vector-self._z)/(self._z_worst-self._z)
			
			norm_FV = (self._FV-self._z)/(self._z_worst-self._z) # COMPROBADO

		
		return norm_obj_vector, norm_FV
	
	
###############################################################################
###############################################################################

	# Asignar al individuo 'x' el j-ésimo subproblema
	def assign_to_jth_subproblem(self, f_x):
		

		if self._decomposition_method == "Tchebycheff":
			scalar_values = np.array([self.tchebycheff_approach(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
			j = np.argmin(scalar_values)
		elif self._decomposition_method == "Weighted Sum Approach" or self._decomposition_method == "weighted_sum_approach":
			scalar_values = np.array([self.weighted_sum_approach(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
			j = np.argmax(scalar_values)
		else:
			scalar_values = np.array([self.tchebycheff_approach(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
			j = np.argmin(scalar_values)
		return j

	
###############################################################################
###############################################################################



	
	# Ejecutar el algoritmo MOEA/D-ADA
	def run(self, max_evals):
		# INICIALIZACIÓN
		self.initialization()
		
		# Punto de referencia z-worst: el peor valor obtenido con cada función objetivo
		if self._maximization:
			self._z_worst = np.amin(self._FV, axis=0)
		else:
			self._z_worst = np.amax(self._FV, axis=0)
		
		# Array con asignaciones de subproblemas.
		# assignated_subproblems[i] = j ---> al i-ésimo individuo de la población
		# se le ha asignado el j-ésimo subproblema (j-ésimo vector de pesos)
		self._assignated_subproblems = np.zeros(self._num_subproblems)
		
		# Asignar subproblema a cada individuo
		for i in range(self._num_subproblems):
			#scalar_values = np.array([self.scalarizing_function(self._FV[i], self._lambdas[k]) for k in range(self._num_subproblems)])
			#j = np.argmin(scalar_values)
			j = self.assign_to_jth_subproblem(self._FV[i])
			self._assignated_subproblems[i] = j
		
		# Para controlar tiempo de ejecución
		epoch_times = []
		total_evals_per_epoch = []
		
		while self._evals < max_evals:
			start_epoch = time.time()
			
			# Actualizar tamaño de la poblacón
			self._population_size = self._population.shape[0]
		
			#print("mu -> {}".format(self._population_size))
			
			#print("Assigned subproblems:")
			#print(self._assignated_subproblems)
			#print()
			
			#print("F-values:")
			#print(self._FV)
			#print()
			
			#print("Valores negativos en FV? -> {}".format((self._FV<0).any()))
			
			# Seleccionar aleatoriamente dos vectores de toda la población
			parents = np.random.choice(np.arange(self._population_size), size=2, replace=False)


			# Cruzas los dos vectores obtenidos para generar descendencia 'u'
			u = self.crossover(self._population[parents[0]], self._population[parents[1]])
			#print("Offspring:")
			#print(u)
			#print()
			
			# Mutar el vector resultante
			# (según la probabilidad de mutación)
			u = self.mutation(u)
			
			
			# Obtener valores objetivo f(u) del individuo descendiente generado
			# (individuo 'u')
			f_u = self.f(u)
			
			
			# Normalizar f(u) y el resto de vectores objetivo de la población
			norm_f_u, norm_FV = self.normalize_fv(f_u)
			
			
			# ASIGNACIÓN
			# Asignar el individuo 'u' al j-ésimo subproblema
			j = self.assign_to_jth_subproblem(norm_f_u)
			
			
			# Generar subconjunto X a partir de los individuos de P (población)
			# que cumplan las dos condiciones siguientes:
			#    - Están asignados al j-ésimo subproblema
			#    - Se encuentran en el vecindario de u (en espacio de soluciones)

			reshaped_u = np.reshape(u, (1,u.size))
			u_distances = euclidean_distances(reshaped_u, self._population).flatten()
			u_neigborhood = np.argsort(u_distances)[0:self._neighborhood_size]
			X = [i for i in range(self._population_size) if (self._assignated_subproblems[i] == j) and (i in u_neigborhood)]
			#print("X:")	# DEBUG
			#print(X)		# DEBUG
			#print()		# DEBUG
			
			# Flags booleanos del operador de adición
			b_explorer = len(X) == 0
			b_winner = False
			#print("Explorer: {}".format(b_explorer))	# DEBUG
			#print()									# DEBUG
			
			# ELIMINACIÓN
			# Eliminamos todos los individuos de 'X' superados por 'u'
			worse_than_u = np.array([x_index for x_index in X if self.compare_decomposition_values(norm_f_u, norm_FV[x_index], self._lambdas[j])])
			#print("Delete indices:")		# DEBUG
			#print(delete_indices)	 	# DEBUG
			
			b_winner = worse_than_u.size > 0
			#print("Winner: {}".format(b_winner))
			if b_winner:
				self._population = np.delete(self._population, worse_than_u, axis=0)
				self._FV = np.delete(self._FV, worse_than_u, axis=0)
				self._assignated_subproblems = np.delete(self._assignated_subproblems, worse_than_u)
			
			
			#print("f(u):")
			#print(f_u)
			
			#print("-------------------------------------")
			
			# ADICIÓN
			if b_winner or b_explorer:
				self._population = np.vstack((self._population, u))
				self._FV = np.vstack((self._FV, f_u))
				self._assignated_subproblems = np.append(self._assignated_subproblems, j)
			
			# Actualización de la población externa
			self.update_external_population(u, f_u)
			
			self._epochs += 1
			epoch_times.append(time.time() - start_epoch)
			total_evals_per_epoch.append(self._evals)
			
		# Actualización de la población externa (a partir de la población interna final)
		#for i in range(self._population.shape[0]):
			#self.update_external_population(self._population[i], self._FV[i])
		
		return self._EP, self._EP_chromosomes, self._lambdas, total_evals_per_epoch, np.median(epoch_times) * self._epochs, self._last_EP_update_eval
		#return self._EP, self._EP_chromosomes

