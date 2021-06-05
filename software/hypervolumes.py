import numpy as np
import sys
from os import mkdir, system
from os.path import isdir
from results_files_functs import read_pareto_front_file
from jmetal.core.quality_indicator import HyperVolume


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
	hv_files_dir = comp_file + "HypervolumesReales/"
	
	if not isdir(comp_file):
		mkdir(comp_file)
		
	if not isdir(hv_files_dir):
		mkdir(hv_files_dir)
	
		
	for name in datasets_names:
		for constr in constr_percentages:
			hv_list_a = []
			hv_list_b = []
			for run in range(num_runs):
				pf_filename = name + str(constr) + "_" + str(run) + ".pf"
				
				# Frentes de pareto de ambos algoritmos
				# (para el dataset, el porcentaje de restricciones y el número de ejecución dados)
				pareto_front_a = np.array(read_pareto_front_file(pf_files_dir_a + pf_filename))
				pareto_front_b = np.array(read_pareto_front_file(pf_files_dir_b + pf_filename))
				
				if maximization:
					pareto_front_a = pareto_front_a * -1.0
					pareto_front_b = pareto_front_b * -1.0
				
				# Obtenemos los vectores de referencia (valores máximos) de ambos frentes de pareto
				max_pf_a = np.amax(pareto_front_a, axis=0)
				max_pf_b = np.amax(pareto_front_b, axis=0)
				
				
				# Punto de referencia: máximos de los dos vectores anteriores (más 1 a cada coordenada)
				ref_point = np.maximum(max_pf_a, max_pf_b)+1.0
				#print(ref_point) # DEBUG
				
				#DEBUG
				#print(np.minimum(np.amin(pareto_front_a, axis=0), np.amin(pareto_front_b, axis=0)))

				# Llamada al ejecutable hv
				hv_a = HyperVolume(ref_point).compute(pareto_front_a)
				hv_b = HyperVolume(ref_point).compute(pareto_front_b)
				
				# Guardamos los hipervolúmenes de cada run
				hv_list_a.append(hv_a)
				hv_list_b.append(hv_b)
				
			# Guardar todos los epsilons de las diferentes runs en el mismo fichero
			# (1 fichero por cada dataset y cada porcentaje de restricciones)
			hv_a_filename = name + str(constr) + "_" +  algorithm_name_a + ".hv"
			hv_b_filename = name + str(constr) + "_" +  algorithm_name_b + ".hv"
			
			np.savetxt(hv_files_dir + hv_a_filename, np.array(hv_list_a))
			np.savetxt(hv_files_dir + hv_b_filename, np.array(hv_list_b))
	
	# Datasets artificiales
	datasets_names = ["circles", "moons", "rand", "spirals"]
	



if __name__ == "__main__":
	main()
