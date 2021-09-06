
import numpy as np
from functions import *
from sklearn.metrics import adjusted_rand_score
import sys
#import telegram_send as ts
#import gc
#import time
import collections
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial import distance
#import random
from math import sqrt

# MOEAD-ADA y variantes
from MOEAD_ADA.MOEAD_ADA import MOEAD_ADA
from MOEAD_ADA.MOEAD_ADA_Norm import MOEAD_ADA_Norm
from MOEAD_ADA.MOEAD_ADA_PD import MOEAD_ADA_PD
from MOEAD_ADA.MOEAD_ADA_PD_Norm_A import MOEAD_ADA_PD_Norm_A
from MOEAD_ADA.MOEAD_ADA_PD_Norm_B import MOEAD_ADA_PD_Norm_B
from MOEAD_ADA.MOEAD_ADA_PD_SSPS import MOEAD_ADA_PD_SSPS


# Funciones Minimización
from clustering_validity_indices import connectedness, clustering_infeasibility#, davies_bouldin_index
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
# Funciones Maximización
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from clustering_validity_indices import satisfied_constraints





def apply(name, algorithm_name, data_set, const_matrix, labels, method_params, rnd_seed):

	print("Procesando " + name + " dataset")

	data_set = normalize(data_set)
	labels = labels
	nb_clusters = len(set(labels))
	n_instances = data_set.shape[0]

	ml_const, cl_const = get_const_list(const_matrix)
	
	
	if method_params.maximization_problem:
		obj_functs = [silhouette_score, calinski_harabasz_score, satisfied_constraints]
		obj_functs_arguments = [[],[],[ml_const, cl_const]]
	else:
		# Vecindario de datos (usado con función objetivo 'connectedness')
		data_neighb_size = int(sqrt(n_instances)) #int(n_instances)
		distances = pairwise_distances(data_set, Y=None, metric='euclidean')
		neighborhoods = distances.argsort(axis = 1)[:,0:data_neighb_size] 
	
		obj_functs = [davies_bouldin_score, connectedness, clustering_infeasibility]
		obj_functs_arguments = [[],[neighborhoods],[ml_const, cl_const]]

	

			   
	if algorithm_name == "MOEAD_ADA":


		algorithm = MOEAD_ADA(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
	elif algorithm_name == "MOEAD_ADA_Norm":

		algorithm = MOEAD_ADA_Norm(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
	elif algorithm_name == "MOEAD_ADA_PD":

		algorithm = MOEAD_ADA_PD(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
	elif algorithm_name == "MOEAD_ADA_PD_Norm_A":

		algorithm = MOEAD_ADA_PD_Norm_A(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
	elif algorithm_name == "MOEAD_ADA_PD_Norm_B":

		algorithm = MOEAD_ADA_PD_Norm_B(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)		
	
		
	elif algorithm_name == "MOEAD_ADA_PD_SSPS":

		algorithm = MOEAD_ADA_PD_SSPS(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.0,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
		
		
	elif algorithm_name == "MOEAD_ADA_PD_KM":

		algorithm = MOEAD_ADA_PD(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.2,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		
	elif algorithm_name == "MOEAD_ADA_KM":

		algorithm = MOEAD_ADA(data_set, nb_clusters, ml_const, cl_const,
			   num_subproblems = method_params.num_subproblems,
			   neighborhood_size_ratio = method_params.neighborhood_size_ratio,
			   objective_functions = obj_functs,
			   obj_functs_args = obj_functs_arguments,
			   prob_mutation = method_params.prob_mutation,
			   prob_crossover = method_params.prob_crossover,
			   decomposition_method = method_params.decomposition_method,
			   crossover_operator = method_params.crossover_operator,
			   ref_z = method_params.ref_z,
			   kmeans_init_ratio = 0.2,
			   maximization_problem = method_params.maximization_problem,
			   random_seed = rnd_seed)
		


	else:
		print("Nombre del algoritmo incorrecto")
		return None

	EP, EP_chromosomes, lambdas, total_evals, execution_time, last_ep_update = algorithm.run(method_params.max_eval)

	best_ars = -1
	min_dist = float("inf")
	avg_k = 0

	for k in range(EP_chromosomes.shape[0]):

		ars = adjusted_rand_score(labels, EP_chromosomes[k,:])
		avg_k += len(list(set(EP_chromosomes[k,:])))

		if ars > best_ars:

			best_ars = ars
			best_moead_assign = EP_chromosomes[k,:]


	# Nº de clusters promedio
	avg_k /= EP_chromosomes.shape[0]

	# Soluciones más cercanas (en espacio de soluciones) al origen de coordenadas
	# Para problemas de MINIMIZACIÓN
	for k in range(EP.shape[0]):
		
		if distance.euclidean(EP[k, :], [0,0,0]) < min_dist:

			min_dist = distance.euclidean(EP[k, :], [0,0,0])
			closest_moead_assign = EP_chromosomes[k, :]
			
	
	# Soluciones más lejanas (en espacio de soluciones) al origen de coordenadas
	# Para problemas de MAXIMIZACIÓN
	max_dist = 0.0
	for k in range(EP.shape[0]):
		
		if distance.euclidean(EP[k, :], [0,0,0]) > max_dist:

			max_dist = distance.euclidean(EP[k, :], [0,0,0])
			furthest_moead_assign = EP_chromosomes[k, :]

	return tuple((best_moead_assign, closest_moead_assign, execution_time, last_ep_update, avg_k, best_ars, EP, lambdas, total_evals, EP_chromosomes, furthest_moead_assign))
	
def main():

	if len(sys.argv) != 3:
		print("Numero de argumentos incorrecto")
		#return -1

	algorithm_name = sys.argv[1]
	constr_percent = float(sys.argv[2])
	

	random_seeds = [55,123,111,9,12345]

	nb_runs = 5

	moead_params = collections.namedtuple("MOEAD_Parameters", "max_eval, num_subproblems,\
	lambda_neighborhood_size, neighborhood_size_ratio, decomposition_method,\
	prob_mutation, prob_crossover, crossover_operator,\
	ref_z, maximization_problem")

	parameters = moead_params(max_eval = 300000,
	   num_subproblems = 100, 
	   lambda_neighborhood_size = 10,
	   neighborhood_size_ratio = 0.1,
	   prob_mutation = 0.1,
	   prob_crossover = 0.5,
	   decomposition_method = "Tchebycheff", 
	   crossover_operator = "uniform", 
	   ref_z = np.array([0.0,0.0,0.0]), 
	   maximization_problem = False) 
	
	if algorithm_name == "MOEAD-ADA":
		algorithm_name = "MOEAD_ADA"


	datasets_folder = "./Datasets/Reales"
	constraints_folder = "./Constraints/Reales"
	#names = ["appendicitis", "balance", "banana_undersmpl", "breast_cancer", "bupa", "contraceptive", 
	#"ecoli", "glass", "haberman", "hayes_roth", "heart", "ionosphere", "iris", "led7digit", "monk2", 
	#"movement_libras", "newthyroid", "page_blocks_undersmpl", "phoneme_undersmpl", "pima", "saheart", 
	#"segment_undersmpl", "sonar", "soybean", "spambase_undersmpl", "spectfheart", "tae", "texture_undersmpl", 
	#"titanic_undersmpl", "vehicle", "vowel", "wdbc", "wine", "yeast", "zoo"]
	
	#names = ["appendicitis", "hayes_roth", "heart", "iris", "spectfheart", "zoo"] # 6
	names = ["appendicitis", "balance", "banana_undersmpl", "bupa", "ecoli", "glass", "haberman", "hayes_roth", "heart", "iris", "led7digit", "monk2", "newthyroid", "pima", "saheart", "soybean", "tae", "titanic_undersmpl", "wine", "zoo"] # 20


	#process_name = algorithm_name + " Reales" + str(int(100 * constr_percent))
	best_out_file = open("Results/" + algorithm_name + "/" + algorithm_name + "_Results_Reales_Best" + str(int(100 * constr_percent)) + ".res", "w+")
	closest_out_file = open("Results/" + algorithm_name + "/" + algorithm_name + "_Results_Reales_Closest" + str(int(100 * constr_percent)) + ".res", "w+")
	PFs_folder = "Results/" + algorithm_name + "/PF_Reales/"
	Lambdas_folder = "Results/" + algorithm_name + "/Lambdas_Reales/"
	TotalEvals_folder = "Results/" + algorithm_name + "/TotalEvals_Reales/"
	
	furthest_out_file = open("Results/" + algorithm_name + "/" + algorithm_name + "_Results_Reales_Furthest" + str(int(100 * constr_percent)) + ".res", "w+")
	extra_info_out_file = open("Results/" + algorithm_name + "/" + algorithm_name + "_Extra_info_Reales" + str(int(100 * constr_percent)) + ".res", "w+")
	Labels_folder = "Results/" + algorithm_name + "/Labels_Reales/"

	
	names, datasets, labels = load_datasets(names, datasets_folder)
	const_percent_vector = [constr_percent]
	const_array = load_constraints(names, const_percent_vector, constraints_folder)

	
	params_string = "# max_eval = " + str(parameters.max_eval) + \
	"# number of subproblems (init. population size) = " + str(parameters.num_subproblems) +\
	"# ref_z = " + str(parameters.ref_z) +\
	"# mutation_probability = " + str(parameters.prob_mutation) +\
	"# neighborhood_size_ratio = " + str(parameters.neighborhood_size_ratio) +\
	"# decomposition_method = " + str(parameters.decomposition_method) +\
	"# crossover_operator = " + str(parameters.crossover_operator) +\
	"# maximization_problem = " + str(parameters.maximization_problem)  + "\n"
	#"# objective_functions = davies_bouldin_index, connectedness, clustering_infeasibility" + "\n"
	#"# objective_functions = silhouette_score, calinski_harabasz_score, satisfied_constraints" + "\n"
	#"# connect_neigh_size = int(sqrt(n_instances))" +\

	best_out_file.write(params_string)
	best_out_file.write("# Datasets: " + str(len(names)) + ", Runs: " + str(nb_runs) + "\n")
	best_out_file.write(str(len(names)) + "," + str(nb_runs) + "\n")
	save_to_file(best_out_file, names)

	best_out_file.write("# ------------------ True Labels ------------------\n")

	closest_out_file.write(params_string)
	closest_out_file.write("# Datasets: " + str(len(names)) + ", Runs: " + str(nb_runs) + "\n")
	closest_out_file.write(str(len(names)) + "," + str(nb_runs) + "\n")
	save_to_file(closest_out_file, names)

	closest_out_file.write("# ------------------ True Labels ------------------\n")
	
	furthest_out_file.write(params_string)
	furthest_out_file.write("# Datasets: " + str(len(names)) + ", Runs: " + str(nb_runs) + "\n")
	furthest_out_file.write(str(len(names)) + "," + str(nb_runs) + "\n")
	save_to_file(furthest_out_file, names)

	furthest_out_file.write("# ------------------ True Labels ------------------\n")
	
	extra_info_out_file.write(params_string)
	extra_info_out_file.write("# Datasets: " + str(len(names)) + ", Runs: " + str(nb_runs) + "\n")
	extra_info_out_file.write(str(len(names)) + "," + str(nb_runs) + "\n")
	save_to_file(extra_info_out_file, names)

	for i in range(len(names)):
		save_to_file(best_out_file, labels[i])
		save_to_file(closest_out_file, labels[i])
		save_to_file(furthest_out_file, labels[i])
		
	
	
	extra_info_out_file.write("# Format\n")
	extra_info_out_file.write("# (execution_time, last_EP_update, avg_k, best_ARI, EP_size)\n")

	
	print("{} runs".format(nb_runs))
	print("{} datasets".format(len(names)))
	
	#n_jobs = multiprocessing.cpu_count() - 2
	# with Parallel(n_jobs = multiprocessing.cpu_count() - 2) as parallel:
	#with Parallel(n_jobs = 35) as parallel:
	with Parallel(n_jobs = 7) as parallel:

		for i in range(nb_runs):
			print("Run semilla {}".format(random_seeds[i]))

			best_out_file.write("# ------------------ Run " + str(i) + " ------------------\n")
			closest_out_file.write("# ------------------ Run " + str(i) + " ------------------\n")
			furthest_out_file.write("# ------------------ Run " + str(i) + " ------------------\n")
			extra_info_out_file.write("# ------------------ Run " + str(i) + " ------------------\n")
			
			results = parallel(delayed(apply)(names[j], algorithm_name, normalize(datasets[j]), const_array[0][j], labels[j], parameters, random_seeds[i]) 
				for j in range(len(names)))

			for j in range(len(results)):

				# Ficheros cromosomas para: mejor ARI, más cercano al origen, más alejado del origen
				save_to_file(best_out_file, results[j][0])
				save_to_file(closest_out_file, results[j][1])
				save_to_file(furthest_out_file, results[j][10])

				# Frentes de Pareto
				np.savetxt(PFs_folder + names[j] + str(int(100*constr_percent)) + "_" + str(i) + ".pf", results[j][6], delimiter = ",", fmt = "%15.10f")

				# Lambdas
				np.savetxt(Lambdas_folder + names[j] + str(int(100*constr_percent)) + "_" + str(i) + ".lambdas", 
					results[j][7], delimiter = ",", fmt = "%15.10f")

				# Total evaluaciones
				np.savetxt(TotalEvals_folder + names[j] + str(int(100*constr_percent)) + "_" + str(i) + ".te", 
					results[j][8], delimiter = ",", fmt = "%15.10f")

				# Info extra 
				save_to_file(extra_info_out_file, np.array([results[j][2], results[j][3], results[j][4], results[j][5], results[j][6].shape[0]]))
				
				# Labels (cromosomas)
				np.savetxt(Labels_folder + names[j] + str(int(100*constr_percent)) + "_" + str(i) + ".label", 
					results[j][9], delimiter = ",", fmt = "%d")
				

			#ts.send(messages=["Mensaje de " + process_name + ": Terminada ejecución número " + str(i)], conf="./configuracion_telegram.txt")
	
	best_out_file.close()
	closest_out_file.close()
	

if __name__ == "__main__": main()
