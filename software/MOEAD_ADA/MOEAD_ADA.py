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
			
			norm_FV = (self._FV-self._z_worst)/(self._z-self._z_worst)
			#for i in range(self._FV.shape[0]):
			#	self._FV[i] = (self._FV[i]-self._z_worst)/(self._z-self._z_worst)
			
		else:
			self._z = np.minimum(self._z, new_obj_vector)
			self._z_worst = np.maximum(self._z_worst, new_obj_vector)
			
			norm_obj_vector = (new_obj_vector-self._z)/(self._z_worst-self._z)
			
			norm_FV = (self._FV-self._z)/(self._z_worst-self._z)
			#for i in range(self._FV.shape[0]):
			#	self._FV[i] = (self._FV[i]-self._z)/(self._z_worst-self._z)
		
		return norm_obj_vector, norm_FV
	
	
###############################################################################
###############################################################################

	# Asignar al individuo 'x' el j-ésimo subproblema
	# PRECONDICIÓN: F(x) debe estar normalizado (valores entre 0 y 1)
	def assign_to_jth_subproblem(self, f_x):
		
		#scalar_values = np.array([self.scalarizing_function(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
		if self._decomposition_method == "Tchebycheff":
			scalar_values = np.array([self.tchebycheff_approach(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
			j = np.argmin(scalar_values)
		elif self._decomposition_method == "Weighted Sum Approach":
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
		# subproblem_allocation[i] = j ---> al i-ésimo individuo de la población
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
			
			# Actualizar tamaño de la poblacón (¿hace falta?)
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
			#print("Parents:")
			#print(self._population[parents[0]])
			#print("-----")
			#print(self._population[parents[1]])
			#print()

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
			#scalar_values = np.array([self.scalarizing_function(norm_f_u, self._lambdas[k]) for k in range(self._num_subproblems)])
			#if self._maximization:
			#	j = np.argmax(scalar_values)
			#else:
			#	j = np.argmin(scalar_values)
			j = self.assign_to_jth_subproblem(norm_f_u)
				
			#print("Scalar values for offspring:")
			#print(scalar_values)
			#print("Best scalar value (j-th): {} ({})".format(j,scalar_values[j]))
			
			
			# Generar subconjunto X a partir de los individuos de P (población)
			# que cumplan las dos condiciones siguientes:
			#    - Están asignados al j-ésimo subproblema
			#    - Se encuentran en el vecindario de u (en espacio de soluciones)

			reshaped_f_u = np.reshape(norm_f_u, (1,norm_f_u.size))
			u_distances = euclidean_distances(reshaped_f_u, self._FV)
			u_neigborhood = np.argsort(u_distances)[:,1:self._neighborhood_size+1]
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
			delete_indices = np.array([x_index for x_index in X if self.compare_scalar_values(norm_f_u, norm_FV[x_index], self._lambdas[j])])
			#print("Delete indices:")		# DEBUG
			#print(delete_indices)	 	# DEBUG
			
			b_winner = delete_indices.size > 0
			#print("Winner: {}".format(b_winner))
			if b_winner:
				self._population = np.delete(self._population, delete_indices, axis=0)
				self._FV = np.delete(self._FV, delete_indices, axis=0)
				self._assignated_subproblems = np.delete(self._assignated_subproblems, delete_indices)
			
			"""
			for x_index in X:
				if self.compare_scalar_values(u,self._population[x_index]):
					# Eliminar de la población al individuo de X
					self._population = np.delete()
					b_winner = True
			"""
			
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
			
		return self._EP, self._EP_chromosomes, self._lambdas, total_evals_per_epoch, np.median(epoch_times) * self._epochs, self._last_EP_update_eval
		#return self._EP, self._EP_chromosomes



"""
	PRUEBAS
"""

if __name__ == "__main__":
	
	random.seed(123)
	np.random.seed(123)
	
	print("Distancia entre un vector y las filas de una matriz")
	m = np.array([[0.0,0.0],
			   [1.0,0.0],
			   [1.0,1.0],
			   [2.0,0.0]])
	v = np.array([[1.0,0.0]])
	dists = euclidean_distances(v,m)
	dists_1d = dists.flatten()
	dist_rep = np.reshape(dists_1d,(1,dists_1d.size))
	print(m)
	print(v)
	print(dists)
	print()
	
	
	
	print("Delete test")
	vl = np.random.randint(0,99,10)
	mask = vl < 60
	num_mask = np.array([i for i in range(vl.size) if vl[i] < 60])
	reduc_vl = np.delete(vl,mask)
	reduc_vl_num = np.delete(vl,num_mask)
	
	print(vl)
	print(mask)
	print(num_mask)
	print(reduc_vl)
	print(reduc_vl_num)
	
	
	print("Restar un vector a una matriz")
	m2 = np.array([[1.0,1.0],
			   [2.0,2.0],
			   [3.0,3.0],
			   [4.0,4.0]])
	v2 = np.array([1.0,1.0])
	rest2 = m2-v2
	print(m2)
	print(v2)
	print(rest2)
	print()
	
	
	
	print("Mínimo entre dos vectores")
	v3 = np.array([1,65,3,66,5,2])
	v4 = np.array([2,4,99,4,1,1])
	v_min = np.minimum(v3,v4)
	print(v3)
	print(v4)
	print(v_min)
	print()