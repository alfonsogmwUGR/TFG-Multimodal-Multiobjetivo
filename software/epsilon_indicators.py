import sys
#from evaluation_measures import epsilon_indicator
from os import mkdir
from os.path import isdir
from results_files_functs import read_pareto_front_file
import numpy as np
from jmetal.core.quality_indicator import EpsilonIndicator


def main():
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print("Numero de argumentos incorrecto")
		return -1
	

	algorithm_name_a = sys.argv[1]
	algorithm_name_b = sys.argv[2]
	
	if len(sys.argv) == 3:
		maximization = False
	elif len(sys.argv) == 4:
		if sys.argv[3] == "min":
			maximization = False
		elif sys.argv[3] == "max":
			maximization = True
		else:
			print("Tercer argumento incorrecto. Por favor, introduzca \"max\" o \"min\".")
			return -1
	
	# DEBUG
	if maximization:
		print("MAX")
	else:
		print("MIN")
	
	constr_percentages = [10, 15, 20]
	num_runs = 5
	
	
	# Datasets reales
	datasets_names  = ["appendicitis", "hayes_roth", "heart", "iris", "newthyroid", "sonar", "soybean", "spectfheart", "wine", "zoo"]
	
		
	pf_files_dir_a = "./Results/"+ algorithm_name_a + "/PF_Reales/"
	pf_files_dir_b = "./Results/"+ algorithm_name_b + "/PF_Reales/"
	
	comp_file = "./Results/" + algorithm_name_a + "_vs_" + algorithm_name_b + "/" 
	epsilons_dir = comp_file + "EpsilonsReales/"
	
	if not isdir(comp_file):
		mkdir(comp_file)
	
	if not isdir(epsilons_dir):
		mkdir(epsilons_dir)
	
	
	for name in datasets_names:
		for constr in constr_percentages:
			epsilon_list_a_b = []
			epsilon_list_b_a = []
			# En cada fichero irán los indicadores epsilon de cada run (ejecución)
			for run in range(num_runs):
				pf_filename = name + str(constr) + "_" + str(run) + ".pf"
				
				pareto_front_a = np.array(read_pareto_front_file(pf_files_dir_a + pf_filename))
				pareto_front_b = np.array(read_pareto_front_file(pf_files_dir_b + pf_filename))
				
				if maximization:
					pareto_front_a = pareto_front_a * -1.0
					pareto_front_b = pareto_front_b * -1.0
				
				#epsilon_a_b = epsilon_indicator(pareto_front_a, pareto_front_b)
				#epsilon_b_a = epsilon_indicator(pareto_front_b, pareto_front_a)
				
				# I_e+(A,B)
				epsilon_a_b = EpsilonIndicator(pareto_front_b).compute(pareto_front_a)
				# I_e+(B,A)
				epsilon_b_a = EpsilonIndicator(pareto_front_a).compute(pareto_front_b)
				
				epsilon_list_a_b.append(epsilon_a_b)
				epsilon_list_b_a.append(epsilon_b_a)
				
			# Guardar todos los epsilons de las diferentes runs en el mismo fichero
			# (1 fichero por cada dataset y cada porcentaje de restricciones)
			epsilons_a_b_filename = name + str(constr) + "_" + algorithm_name_a + "_vs_" + algorithm_name_b + ".epsilon"
			epsilons_b_a_filename = name + str(constr) + "_" + algorithm_name_b + "_vs_" + algorithm_name_a + ".epsilon"
			
			epsilons_a_b_file = epsilons_dir + epsilons_a_b_filename
			epsilons_b_a_file = epsilons_dir + epsilons_b_a_filename
			
			np.savetxt(epsilons_a_b_file, np.array(epsilon_list_a_b))
			np.savetxt(epsilons_b_a_file, np.array(epsilon_list_b_a))
			
	
	# Datasets artificiales
	
	datasets_names = ["circles", "moons", "rand", "spirals"]
	
	
if __name__ == "__main__":
	main()
