import numpy as np
import random
import time
#from sklearn.metrics import pairwise_distances
from MOEAD_ADA.MOEAD_ADA_PD import MOEAD_ADA_PD
import math





class MOEAD_ADA_PD_SSPS(MOEAD_ADA_PD):
	
	
	# Ejecutar el algoritmo MOEA/D-ADA
	def run(self, max_evals):
		# INICIALIZACIÓN
		self.initialization()
		

		# Array con asignaciones de subproblemas.
		# assignated_subproblems[i] = j ---> al i-ésimo individuo de la población
		# se le ha asignado el j-ésimo subproblema (j-ésimo vector de pesos)
		self._assignated_subproblems = np.zeros(self._num_subproblems, dtype=np.int)

		# Asignar subproblema a cada individuo
		for i in range(self._num_subproblems):
			j = self.assign_to_jth_subproblem(self._FV[i])
			self._assignated_subproblems[i] = j
			


		self._last_added_solution_eval = 0
		
		# Para controlar tiempo de ejecución
		epoch_times = []
		total_evals_per_epoch = []
		
		while self._evals < max_evals:
			start_epoch = time.time()
			
			# Actualizar tamaño de la poblacón
			self._population_size = self._population.shape[0]
			self._neighborhood_size = math.floor(self._population.shape[0]*self._neighborhood_size_ratio)
		
			
			# Seleccionar dos padres que estén asignados al mismo subproblema
			same_subprob_parents = []
			while len(same_subprob_parents) < 2:
				random_j = random.randint(0,self._num_subproblems-1)
				same_subprob_parents = [i for i in range(self._population_size) if self._assignated_subproblems[i] == random_j]
			parents = np.random.choice(np.array(same_subprob_parents), size=2, replace=False)


			# Cruzas los dos vectores obtenidos para generar descendencia 'u'
			u = self.crossover(self._population[parents[0]], self._population[parents[1]])
			
			# Mutar el vector resultante
			# (según la probabilidad de mutación)
			u = self.mutation(u)
			
			
			# Obtener valores objetivo f(u) del individuo descendiente generado
			# (individuo 'u')
			f_u = self.f(u)

			
			# Normalizar f(u) y el resto de vectores objetivo de la población
			self.update_z(f_u)	
					

			# ASIGNACIÓN
			# Asignar el individuo 'u' al j-ésimo subproblema
			#j = self.assign_to_jth_subproblem(norm_f_u)
			j = self.assign_to_jth_subproblem(f_u)
				
			# Generar subconjunto X a partir de los individuos de P (población)
			# que cumplan las dos condiciones siguientes:
			#    - Están asignados al j-ésimo subproblema
			#    - Se encuentran en el vecindario de u (en espacio de soluciones)

			u_distances = np.array([np.linalg.norm(u-self._population[i]) for i in range(self._population_size)])
			u_neigborhood = np.argsort(u_distances)[0:self._neighborhood_size]

			
			X = [i for i in u_neigborhood if self._assignated_subproblems[i] == j]

			
			# Flags booleanos del operador de adición
			b_explorer = len(X) == 0
			b_winner = False
			
			
			# Individuos de 'X' superados por 'u'
			worse_than_u = np.array([x_index for x_index in X if self.compare_decomposition_values(f_u, self._FV[x_index], self._lambdas[j]) ])
			
			b_winner = worse_than_u.size > 0
			
			
			# ELIMINACIÓN
			if b_winner:
				self._population = np.delete(self._population, worse_than_u, axis=0)
				self._FV = np.delete(self._FV, worse_than_u, axis=0)
				self._assignated_subproblems = np.delete(self._assignated_subproblems, worse_than_u)
			
			
			# ADICIÓN
			if b_winner or b_explorer:
				self._population = np.vstack((self._population, u))
				self._FV = np.vstack((self._FV, f_u))
				self._assignated_subproblems = np.append(self._assignated_subproblems, j)
				self._last_added_solution_eval = self._evals
			
			# Actualización de la población externa
			self.update_external_population(u, f_u)
			
			self._epochs += 1
			epoch_times.append(time.time() - start_epoch)
			total_evals_per_epoch.append(self._evals)
			

		
		return self._EP, self._EP_chromosomes, self._lambdas, total_evals_per_epoch, np.median(epoch_times) * self._epochs, self._last_EP_update_eval #self._last_added_solution_eval

	