import numpy as np
#from sklearn import datasets
import itertools
import random
from sklearn.metrics import adjusted_rand_score
import datetime
import os


def generate_data_2D(centers, sigmas, numb_data):

	xpts = np.zeros(1)
	ypts = np.zeros(1)
	labels = np.zeros(1)
	for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
		xpts = np.hstack((xpts, np.random.standard_normal(numb_data) * xsigma + xmu))
		ypts = np.hstack((ypts, np.random.standard_normal(numb_data) * ysigma + ymu))
		labels = np.hstack((labels, np.ones(numb_data) * i))

	X = np.zeros((len(xpts) - 1, 2))
	X[:, 0] = xpts[1:]
	X[:, 1] = ypts[1:]

	y = labels[1:]

	return X, y


def gen_rand_const(labels, nb_const):

	pairs = np.array(list(itertools.combinations(range(0, len(labels)), 2)))
	ind = random.sample(range(0, len(pairs)), nb_const)
	const = pairs[ind]

	const_matrix = np.identity(len(labels))

	for i in const:

		if labels[i[0]] == labels[i[1]]:

			const_matrix[i[0], i[1]] = 1
			const_matrix[i[1], i[0]] = 1

		else:

			const_matrix[i[0], i[1]] = -1
			const_matrix[i[1], i[0]] = -1

	return const_matrix


def get_const_list(m):
	ml = []
	cl = []

	for i in range(np.shape(m)[0]):
		for j in range(i + 1, np.shape(m)[0]):
			if m[i, j] == 1:
				ml.append((i, j))
			if m[i, j] == -1:
				cl.append((i, j))

	return ml, cl


def twospirals(n_points, noise=.5):
	n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
	d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
	d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise

	return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
			np.hstack((np.zeros(n_points),np.ones(n_points))))

def get_usat_percent_list(ml, cl, clustering):

	unsat = 0

	for i in range(len(ml)):

		if clustering[ml[i][0]] != clustering[ml[i][1]]:
			unsat += 1

	for i in range(len(cl)):

		if clustering[cl[i][0]] == clustering[cl[i][1]]:
			unsat += 1

	return (unsat / (len(ml) + len(cl))) * 100


def normalize(dataset):
	dataset = np.array(np.matrix(dataset))
	# mins = np.min(dataset, axis = 0)
	# max_minus_min = np.max(dataset, axis = 0) - mins
	# print(np.where(max_minus_min == 1)[0])
	# max_minus_min[np.where(max_minus_min == 1)[0]] = 1

	ptp = dataset.ptp(0)
	ptp[np.where(ptp == 0)[0]] = 1.0

	return (dataset - dataset.min(0)) / ptp

#Generar nuevas restricciones para bupa, breast_cancer, banana, iris, wine
def load_datasets(names, folder):

	names = np.sort(names)
	datasets_array = []
	labels_array = []
	names_array = []

	for i in range(len(names)):

		data = np.loadtxt(folder + "/" + names[i] + ".dat", delimiter = ",", dtype=str, comments = "@")
		data_set = np.asarray(data[:, :-1].astype(float))
		data_labels = np.asarray(data[:, -1])
		datasets_array.append(data_set)
		labels_array.append(data_labels)
		names_array.append(names[i])

	return names_array, datasets_array, labels_array


def load_constraints(names, const_percent_array, folder):

	const_array = [[] for _ in range(len(const_percent_array))]
	const_array_index = 0

	for label_percent in const_percent_array:

		print("Cargando restricciones en porcentaje: " + str(label_percent))

		for name in names:
			const = np.loadtxt(folder + "/" + str(name) + "(" + str(label_percent) + ").txt", dtype=np.int8)
			const_array[const_array_index].append(const)

		const_array_index += 1

	return const_array

def print_data_info(names, datasets_list, labels):

	print("Dataset & No. Classes & Features\\\\")
	
	for i in range(len(names)):

		n = np.shape(datasets_list[i])[0]
		nb_class = len(set(labels[i]))
		features = np.shape(datasets_list[i])[1]

		print(names[i].title() + " & " + str(n) + " & " + str(nb_class) + " & " + str(features) + " \\\\")

def print_constraints_info(names, folder):

	const_percent_vector = [0.1, 0.15, 0.2]
	const_array = load_constraints(names, const_percent_vector, folder)

	for i in range(len(names)):

		print(names[i].title() + " & ", end='')

		for l in range(len(const_percent_vector)):

			c = const_array[l][i]
			total = np.count_nonzero(c)
			total = (total - np.shape(c)[0]) / 2
			unique, counts = np.unique(c, return_counts=True)
			d = dict(zip(unique, counts))
			ml = (d[1] - np.shape(c)[0]) / 2
			cl = d[-1] / 2

			if l < len(const_percent_vector) - 1:
				print("%d & %d && " % (ml, cl), end='')
			else:
				print("%d & %d" % (ml, cl), end='')

		print(" \\\\")

def save_to_file(file, line):

	for i in range(len(line)-1):

		file.write(str(line[i]) + ",")

	file.write(str(line[len(line)-1]))
	file.write("\n")

def gen_and_save_const_matrix(names, labels, folder):

	const_percent_vector = np.array([0.1, 0.15, 0.2])

	for label_percent in const_percent_vector:

		for i in range(len(names)):

			name = names[i]
			labels_set = labels[i]
			# set_size = datasets[i].shape[0]
			set_size = len(labels[i])
			set_percent = np.ceil(set_size * label_percent)
			nb_const = int((set_percent * (set_percent - 1)) / 2)
			const = gen_rand_const(labels_set, nb_const)
			np.savetxt(folder + "/" + str(name) + "(" + str(label_percent) + ").txt", const, fmt='%5d')

#names: lista con los nombres de los datasets
#labels: lista que contiene listas de etiquetas verdaderas en correspondencia con names
#general_results: lista que contiene listas con los resultados de las ejecuciones, que consisten en listas de etiquetas en correspondencia con names,
#la longitud de esta lista es igual al numero de ejecuciones para las medias, la longitud de las listas internas es igual al numero de datasets, 
#la longitud de las listas de etiquetas es igual al número de instancias del dataset al que corresponden

def read_results_file(input_file):

	#Define separator
	sep = ","

	#Load all lines of result file
	file_lines = input_file.readlines()

	#Get the info line containing number of datasets and number of runs
	info_line = file_lines[2]
	info_line = info_line[:-1].split(sep)
	nb_datasets = int(info_line[0])
	nb_runs = int(info_line[1])

	#Get the name of the datasets
	names = file_lines[3][:-1].split(sep)

	#Get true labels of the datasets
	raw_labels = file_lines[5:5 + nb_datasets]
	labels = [i[:-1].split(sep) for i in raw_labels]
	general_results = [[] for i in range(nb_runs)]

	for i in range(nb_runs):

		run_results = [j[:-1].split(sep) for j in file_lines[4 + (nb_datasets + 1) * (i+1):4 + (nb_datasets + 1) * (i+2)]]
		run_results = run_results[1:]
		general_results[i] = run_results

	input_file.close()

	return names, labels, general_results

def read_results_dir(input_dir):

	#Define separator
	sep = ","

	results_filenames = os.listdir(input_dir)
	results_filenames.sort()
	names = []
	labels = []
	nb_runs_array = []
	nb_datasets = len(results_filenames)

	for i in range(nb_datasets):

		input_file = open(input_dir + "/" + results_filenames[i])
		file_lines = input_file.readlines()
		info_line = file_lines[2]
		info_line = info_line[:-1].split(sep)
		nb_runs_array.append(int(info_line[1]))
		input_file.close()

	if len(set(nb_runs_array)) > 1:

		print("nb_runs mismatch")
		return 0

	else:
		nb_runs = nb_runs_array[0]
		general_results = [[] for i in range(nb_runs)]

	for i in range(nb_datasets):

		input_file = open(input_dir + "/" + results_filenames[i])
		#Load all lines of result file
		file_lines = input_file.readlines()

		#Get the name of the datasets
		names.append(file_lines[3][:-1])#[:-1].split(sep)

		#Get true labels of the datasets
		labels.append(file_lines[5][:-1].split(sep))

		for i in range(nb_runs):

			run_results = file_lines[7 + i][:-1].split(sep)
			general_results[i].append(run_results)

		input_file.close()

	return names, labels, general_results


def print_mean_times(names, results, nb_additional_results = 0):

	nb_runs = len(results)
	nb_datasets = len(names)
	times = np.zeros((len(names), nb_runs))

	for i in range(nb_datasets):

		for j in range(nb_runs):

			times[i][j] = float(results[j][i][-(1 + nb_additional_results)]) if results[j][i][0] != "None" else 'inf'

	print("Dataset & Mean & Std & Var \\\\")
	for i in range(nb_datasets):

		print(names[i].title() + " & %.3f" % (np.mean(times[i,:])) + 
			" & %.3f" % (np.std(times[i,:])) + 
			" & %.3f" % (np.var(times[i,:])) + 
			" \\\\")

	print("----------------------- Time in Deltatime Format -----------------------")
	print("Dataset & Mean & Std & Var \\\\")
	for i in range(nb_datasets):

		print(names[i].title() + " & " + str(datetime.timedelta(seconds = np.mean(times[i,:]))) + 
			" & %.3f" % (np.std(times[i,:])) + 
			" & %.3f" % (np.var(times[i,:])) + 
			" \\\\")


def print_mean_ari(names, results, labels, nb_additional_results = 0):

	nb_runs = len(results)
	nb_datasets = len(names)
	aris = np.zeros((len(names), nb_runs))

	for i in range(nb_datasets):

		for j in range(nb_runs):

			aris[i][j] = adjusted_rand_score(labels[i], results[j][i][:-(1 + nb_additional_results)]) if results[j][i][0] != "None" else -1

	print("Dataset & Mean & Std & Var \\\\")
	for i in range(nb_datasets):

		print(names[i].title() + " & %.3f" % (np.mean(aris[i,:])) + 
			" & %.3f" % (np.std(aris[i,:])) + 
			" & %.3f" % (np.var(aris[i,:])) + 
			" \\\\")


def print_mean_unsat(names, results, constraints, nb_additional_results = 0):

	nb_runs = len(results)
	nb_datasets = len(names)
	unsats = np.zeros((len(names), nb_runs))

	for i in range(nb_datasets):

		for j in range(nb_runs):

			ml, cl = get_const_list(constraints[0][i])
			unsats[i][j] = get_usat_percent_list(ml, cl, results[j][i][:-(1 + nb_additional_results)]) if results[j][i][0] != "None" else 100

	print("Dataset & Mean & Std & Var \\\\")
	for i in range(nb_datasets):

		print(names[i].title() + " & %.3f" % (np.mean(unsats[i,:])) + 
			" & %.3f" % (np.std(unsats[i,:])) + 
			" & %.3f" % (np.var(unsats[i,:])) + 
			" \\\\")

def get_names_intersection(all_names):

	intersec = set(all_names[0])

	for i in range(1, len(all_names)):

		intersec = intersec & set(all_names[i])

	intersec = list(intersec)
	intersec.sort()

	return intersec

def get_paper_tables(all_names, all_labels, all_results, intersec_constraints, intersec_names, nb_additional_results):

	mean_ari = np.zeros((len(intersec_names), len(all_names)))
	mean_unsat =  np.zeros((len(intersec_names), len(all_names)))
	mean_time = np.zeros((len(intersec_names), len(all_names)))

	std_ari = np.zeros((len(intersec_names), len(all_names)))
	std_unsat =  np.zeros((len(intersec_names), len(all_names)))
	std_time = np.zeros((len(intersec_names), len(all_names)))

	for i in range(len(intersec_names)):

		ml, cl = get_const_list(intersec_constraints[0][i])

		for j in range(len(all_names)):

			nb_runs = len(all_results[j])

			name_index = all_names[j].index(intersec_names[i])

			results_list_ari = np.zeros(nb_runs)
			results_list_time = np.zeros(nb_runs)
			results_list_unsat = np.zeros(nb_runs)

			for k in range(nb_runs):

				results_list_ari[k] = adjusted_rand_score(all_labels[j][name_index], all_results[j][k][name_index][:-(1 + nb_additional_results[j])]) if all_results[j][k][name_index][0] != "None" else -1
				results_list_time[k] = float(all_results[j][k][name_index][-(1 + nb_additional_results[j])])
				results_list_unsat[k] = get_usat_percent_list(ml, cl, all_results[j][k][name_index][:-(1 + nb_additional_results[j])]) if all_results[j][k][name_index][0] != "None" else 100

			mean_ari[i,j] = np.mean(results_list_ari)
			mean_time[i,j] = np.mean(results_list_time)
			mean_unsat[i,j] = np.mean(results_list_unsat)

			std_ari[i,j] = np.std(results_list_ari)
			std_time[i,j] = np.std(results_list_time)
			std_unsat[i,j] = np.std(results_list_unsat)

	return intersec_names, mean_ari, std_ari, mean_unsat, std_unsat, mean_time, std_time





