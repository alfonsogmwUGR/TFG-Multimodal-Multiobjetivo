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



"""
	Pruebas
"""


if __name__ == "__main__":
	
	random.seed(123)
	np.random.seed(123)
	
	obj_functs_params = [[2,3],
						 [4,5]]
	tst = np.random.randint(0, 5, size=(10, 20))
	
	
	# Comprueba si el vector a domina al vector b
	def dominates(a,b):
		return (a <= b).all() and (a < b).any()
	
	
	x = np.array([[1,2,3],
				  [1,3,3],
				  [2,3,4],
				  [1,2,2],
				  [3,5,1],
				  [9,9,9],
				  [3,1,6],
				  [8,8,8]])
	
	distances = pairwise_distances(x, Y=None, metric='euclidean')
	neighborhood = distances.argsort(axis = 1)[:,1:5] # COMPROBADO
	
	y = np.array([6,5,4,3,1])
	
	print(dominates(x[0],x[1]))
	
	# Eliminar varios elementos de lista dados los índices de los elementos a eliminar
	print("Eliminar elementos")
	test_list = ['a','b','c','d','e','f']
	test_list_2 = ['a','b','c','d','e','f']
	indices = [1,3,5]
	for index in indices:
		#test_list.pop(index)
		pass
	for index in indices[::-1]:
		test_list_2.pop(index)
	
	print(test_list)
	print(test_list_2)
	
	# DOMINANCIA
	print("Dominated rows")
	x = np.array([[1,2,3],
				  [1,2,3],
				  [2,3,4],
				  [1,2,2],
				  [1,3,3]])
	print(x)

	y = np.array([1,3,2])
	
	print(x > y)
	print(y < x)
	greater_equal_matrix = x >= y
	
	greater_equal_count = np.sum(x >= y, axis = 1)
	greater_strict_count = np.sum(x > y, axis = 1)
	
	greater_equal_flags = greater_equal_count == x.shape[1]
	greater_strict_flags = greater_strict_count >= 1
	

	
	dominated = np.logical_and(greater_equal_flags, greater_strict_flags)
	
	x = np.delete(x,dominated,axis=0)
	print(x)
	print(dominated.any())
	
	pru = np.empty((0,3))
	print(pru)
	print(pru.size)
	pru = np.vstack((pru,y))
	print(pru)
	print(pru.size)
	
	
	
	# Prueba cruce uniforme
	a = np.array([1,2,3,4,5,6])
	b = np.array([3,4,2,1,7,2])
	
	probabilities = np.random.uniform(size=a.size)
	offspring = np.where(probabilities > 0.5, a, b)
	
	print("CROSSOVER UNIFORME")
	print(a)
	print(b)
	print(offspring)
	
	
	# Prueba mutación
	print("Mutación")
	probabilities = np.random.uniform(size=a.size)
	prob_of_mutation = 0.2
	mutate = probabilities < prob_of_mutation
	new_genes = np.random.randint(0, 7, sum(mutate))
	
	
	mutated_chromosome = np.copy(offspring)
	mutated_chromosome[mutate] = new_genes
	print(offspring)
	print(mutated_chromosome)
	
	# Vectores lambda sumando 1
	#print("Vectores lambda sumando 1")
	#vects = normalize(np.random.randint(low = 1, high = 99, size = (4, 3)))
	#print(vects)
	
	
	# Generar vectores lambda válidos
	print("GENERAR VECTORES LAMBDA VÁLIDOS")
	lambda_list = []
	upper_bound = 1.0
	num_elems_lambda = 4
	for i in range(num_elems_lambda):
		new_value = random.uniform(0,upper_bound)
		lambda_list.append(new_value)
		upper_bound -= new_value
	
	print(lambda_list)
	print(sum(lambda_list))
	remaining = 1.0-sum(lambda_list)
	min_index = np.argmin(np.array(lambda_list))
	lambda_list[min_index] += remaining
	print(lambda_list)
	print(sum(lambda_list))
	
	

	print("NUEVO APPROACH PARA LAMBDAS")
	lambda_list = []
	upper_bound = 1.0
	num_elems_lambda = 4
	for i in range(num_elems_lambda):
		new_value = random.uniform(0,upper_bound)
		lambda_list.append(new_value)
		
	print(lambda_list)
	print(sum(lambda_list))
	print()
	
	surplus = (sum(lambda_list) - 1.0)/num_elems_lambda
	min_val = min(lambda_list)
	prod_val = sum(lambda_list)/num_elems_lambda
	print((sum(lambda_list) - 1.0))
	print(surplus)
	print()
	
	new_lambda_list = [l for l in lambda_list]
	print(new_lambda_list)
	print(sum(new_lambda_list))
	print()
	
	
	# MÉTODO NUMÉRICO FARA FORZAR SUMATORIA IGUAL A 1
	# !!!
	while (abs(sum(lambda_list)-1) > 0.0000000000000001):
		mean = sum(lambda_list)/num_elems_lambda
		num_exceding_elements = len([l for l in lambda_list if l>mean])
		surp = (sum(lambda_list) - 1.0)/num_exceding_elements
		new_lambda_list = []
		for l in lambda_list:
			if l > surp:
				new_lambda_list.append(l-surp)
			else:
				new_lambda_list.append(l)
		lambda_list = new_lambda_list
		
	print(lambda_list)
	print(sum(lambda_list))
	print()

	
	
	print("NUEVO APPROACH PARA LAMBDAS V2")
	lambda_list = []
	upper_bound = 1.0
	num_elems_lambda = 4
	for i in range(num_elems_lambda):
		new_value = random.uniform(0,upper_bound)
		lambda_list.append(new_value)
		
	print(lambda_list)
	print(sum(lambda_list))
	print()
	
	while (sum(lambda_list) > 1.00000):
		mean = sum(lambda_list)/num_elems_lambda
		upper_bound = max(min(lambda_list),0.1)
		substract_val = random.uniform(0, upper_bound)
		rand_index = random.randint(0, num_elems_lambda-1)
		if lambda_list[rand_index] >= substract_val:
			lambda_list[rand_index] = lambda_list[rand_index] - substract_val
	
	
	print(lambda_list)
	print(sum(lambda_list))
	print()
	
	
	# Máximo de cada columna
	print("Máximo de cada columna")
	x = np.array([[1,2,3],
			  [7,2,3],
			  [2,3,4],
			  [1,2,2],
			  [1,9,3]])
	maxs_x = np.amax(x,axis=0)
	print(x)
	print(maxs_x)
	print()
	
	
	
	print("Máximos de dos arrays")
	a = np.array([3,6,5,9,2])
	b = np.array([2,7,1,8,6])
	m = np.maximum(a,b)
	print(a)
	print(b)
	print(m)
	print()
	
	
	print("Probando Tchebycheff")
	l = np.array([0.15,0.2,0.25,0.3,0.1])
	z_dif = np.abs(a-m)
	dist = l * z_dif
	tch = np.amax(dist)
	print("x")
	print(a)
	print("z")
	print(m)
	print("z-dif")
	print(z_dif)
	print("lambdas")
	print(l)
	print("zdif * lambdas")
	print(dist)
	print("max (tcheb.)")
	print(tch)
	print()
	
	
	print("one-point crossover")
	parent_a = np.array(range(7))
	parent_b = np.array(range(0,70,10))
	point = 5
	offspring_1 = np.copy(parent_a)
	offspring_2 = np.copy(parent_a)
	
	offspring_1[point:] = parent_b[point:]
	offspring_2[0:point] = parent_b[0:point]
	
	print(parent_a)
	print(parent_b)
	print("--offsp--")
	print(offspring_1)
	print(offspring_2)
	print()
	
	

	
	r = np.array([5,2,7,4])
	r_not = np.array([23,77,44,99])
	mtrx = np.array([[1,2,3,4],
					[99,55,3,2],
					[5,2,7,4],
					[22,77,44,99]])
	

	