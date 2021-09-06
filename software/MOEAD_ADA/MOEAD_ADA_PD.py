



import numpy as np
#import random
import time
#from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from MOEAD_ADA.MOEAD_ADA import MOEAD_ADA
import math





class MOEAD_ADA_PD(MOEAD_ADA):


	
###############################################################################
###############################################################################


	def perpendicular_distance(self, f_x, lambda_weights):
		
		if self._maximization:
			d1 = np.linalg.norm((self._z-f_x)*lambda_weights)/np.linalg.norm(lambda_weights)
			d2 = np.linalg.norm(f_x-(self._z-d1*lambda_weights))
		else:
			d1 = np.linalg.norm((f_x-self._z)*lambda_weights)/np.linalg.norm(lambda_weights)
			d2 = np.linalg.norm(f_x-(self._z+d1*lambda_weights))
			
		return d2
	


			
###############################################################################
###############################################################################



	# Asignar al individuo 'x' el j-ésimo subproblema
	def assign_to_jth_subproblem(self, f_x):
		
		# DISTANCIA PERPENDICULAR
		scalar_values = np.array([self.perpendicular_distance(f_x, self._lambdas[k]) for k in range(self._num_subproblems)])
		return np.argmin(scalar_values)

	