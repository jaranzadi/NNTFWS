# THIS code follows a Weyl node throgh k space for different values of potential vector A
# (c) Juan Ignacio Aranzadi, 2024

# This sets the number of threads used by numpy to 1 assuming that the main part of the code is already paralellized.
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

from numba import njit
import time
from mpi4py import MPI
from weyl_node_functions import *


import argparse
import yaml
import math

import numpy as np 

@njit
def ceroifnegative(x):
	if x < 0:
		a = 0
	if x >= 0:
		a = 1
	return a


# Searches in a patch centered around wn in a box of size 2*delta
def search_in_patch(rank, size, wn, delta, lenght, fermi_energy, frq, A_field, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice):
	lenght = np.array(lenght, dtype=np.int32)
 

	if lenght[0] % 2 == 1:
		kx = 2*np.pi/a_lattice * np.linspace(wn[0]-delta[0], wn[0]+delta[0], lenght[0])
	if lenght[0] % 2 == 0:
		step0 = 2*delta[0]/lenght[0]
		kx = 2*np.pi/a_lattice * np.arange(wn[0]-delta[0], wn[0]+delta[0], step0) + step0 * ceroifnegative(delta[0])
  
	if lenght[1] % 2 == 1:
		ky = 2*np.pi/b_lattice * np.linspace(wn[1]-delta[1], wn[1]+delta[1], lenght[1])
	if lenght[1] % 2 == 0:
		step1 = 2*delta[1]/lenght[1]
		ky = 2*np.pi/b_lattice * np.arange(wn[1]-delta[1], wn[1]+delta[1], step1) + step1 * ceroifnegative(delta[1])
  
	if lenght[2] % 2 == 1:
		kz = 2*np.pi/c_lattice * np.linspace(wn[2]-delta[2], wn[2]+delta[2], lenght[2])
	if lenght[2] % 2 == 0:
		step2 = 2*delta[2]/lenght[2]
		kz = 2*np.pi/c_lattice * np.arange(wn[2]-delta[2], wn[2]+delta[2], step2) + step2 * ceroifnegative(delta[2])
		
	kx_search = int(lenght[0])
	ky_search = int(lenght[1])
	kz_search = int(lenght[2])
	indices = []

	for i in range(kx_search):
		for j in range(ky_search):
			for k in range(kz_search):
				indices.append([i, j, k])

	

	bands = {}
	k_bz = {}
 
	integrals_floquet = floquet_integrals(frq, A_field, n_replicas, n_bands, total_bands, data, a_lattice, b_lattice, c_lattice)
	t_comstruct = 0
	t_diagonalize = 0
	t_append = 0
	time5 = time.time()
	for m in range(rank, int(lenght[0]*lenght[1]*lenght[2]), size):
		index = indices[m]

		# CONSTRUCT MATRIX
		k = np.array([kx[index[0]],ky[index[1]],kz[index[2]]])
		time1 = time.time()

		if n_replicas <= 1:
			H1 = H_matrix_floquet_fast_v2(k, frq, A_field, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet)
		else:
			H1 = H_matrix_floquet_fast(k, frq, A_field, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet)
	
		t_comstruct += time.time()-time1

		# DIAGONALIZE MATRIX
		time2 = time.time()
		eigenvals = np.linalg.eigvalsh(H1)
		t_diagonalize += time.time()-time2

		# APPEND RESULTS
		time3 = time.time()
		bands_plot = np.transpose(eigenvals) - fermi_energy
		bands[m] = bands_plot
		k_bz[m] = k
		t_append += time.time()-time3
  
	# if rank == 0:
	# 	print('DEBUG, entire loop', (time.time()-time5))
	# 	print('DEBUG, entire loop p/matrix', (time.time()-time5)/(lenght[0]*lenght[1]*lenght[2]/size))
	# 	print('DEBUG, append time p/matrix', t_append/(lenght[0]*lenght[1]*lenght[2]/size))
	# 	print('DEBUG, diagnoalize p/matrix',t_diagonalize/(lenght[0]*lenght[1]*lenght[2]/size), 'DEBUG, construct p/matrix', t_comstruct/(lenght[0]*lenght[1]*lenght[2]/size))

	return bands, k_bz

# Recieves a list of weyl node positions and predict the movement using a taylor expansion of the movement
def prediction(wn_list, n_of_nodes, cutoff):
	movement = np.zeros(3)
	if n_of_nodes >= 2:
		if n_of_nodes <= cutoff:
			for n in range(1, n_of_nodes):
				for i in range(n+1):
					choose = math.comb(n, i)
					movement += (-1)**(i) * choose * np.array(wn_list[n_of_nodes - 1 - i, :]) / (math.factorial(n))

		if n_of_nodes > cutoff:
			for n in range(1, cutoff+1):
				for i in range(n+1):
					choose = math.comb(n, i)
					movement += (-1)**(i) * choose * np.array(wn_list[n_of_nodes - 1 - i, :]) / (math.factorial(n))
	return movement

# Function that loops over all A values and saves the results in a txt file.
def run_search(comm, rank, size, wn, delta, minsize, lenght, fermi_energy, frq, A_field_vect, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, output):
	if rank == 0:
		start = time.time()
		node = {}
		ll = 0

	wn_list = np.zeros((len(A_field_vect),3))
	cutoff = 8

	delta = np.array(minsize)
	for A_field_index in range(len(A_field_vect)): 
		if A_field_index >= 1:
			if A_field_vect[A_field_index] != A_field_vect[A_field_index-1]:
				move_nl = prediction(wn_list, A_field_index, cutoff)
				delta   = 1.25*(minsize + np.abs(move_nl))
				wn      = np.array(wn) + (move_nl)/6
	
	
		
		bands_mpi, k_mpi = search_in_patch( rank, size, wn, delta, lenght, fermi_energy, frq, A_field_vect[A_field_index], n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice)

		# Gather all local dictionaries at process 0
		global_bands_mpi = comm.gather(bands_mpi, root=0)
		global_k_mpi = comm.gather(k_mpi, root=0)

		# Process 0 aggregates all dictionaries into one
		if rank == 0:
			bands_final = {}
			for d in global_bands_mpi:
				bands_final.update(d)

			k_bz_plot  = {}
			for d in global_k_mpi:
				k_bz_plot.update(d)

			bands_final = {k: bands_final[k] for k in sorted(bands_final.keys())}
			k_bz_plot = {k: k_bz_plot[k] for k in sorted(k_bz_plot.keys())}

			kx = np.zeros(lenght[0]*lenght[1]*lenght[2])
			ky = np.zeros(lenght[0]*lenght[1]*lenght[2])
			kz = np.zeros(lenght[0]*lenght[1]*lenght[2])
			bands = np.zeros((lenght[0]*lenght[1]*lenght[2], n_bands*(2*n_replicas+1)))


			for key in k_bz_plot:
				kx[key] = k_bz_plot[key][0] 
				ky[key] = k_bz_plot[key][1]
				kz[key] = k_bz_plot[key][2]  

			for key in bands_final: 
				bands[key][:] = bands_final[key]


			bands_weyl = bands[:,nodes_at[1]]-bands[:,nodes_at[0]]
			if np.min(bands_weyl) < 1e-3:
				node[ll] = [nodes_at[1], nodes_at[0], np.min(bands_weyl), kx[np.argmin(bands_weyl)]*a_lattice/(2*np.pi),ky[np.argmin(bands_weyl)]*b_lattice/(2*np.pi), kz[np.argmin(bands_weyl)]*c_lattice/(2*np.pi), bands[np.argmin(bands_weyl),nodes_at[1]]]
				print("Gap: ", np.min(bands_weyl))

				wn = [kx[np.argmin(bands_weyl)]*a_lattice/(2*np.pi), ky[np.argmin(bands_weyl)]*b_lattice/(2*np.pi), kz[np.argmin(bands_weyl)]*c_lattice/(2*np.pi)]

				if len(node) > 1:
					wn_old = node[ll-1][3:6]
					with np.printoptions(precision=16):
						print(delta, wn, np.array(wn) - np.array(wn_old), A_field_vect[A_field_index], A_field_vect[A_field_index-1])
				wn_list[A_field_index, :] = np.array(wn)
				ll += 1
				A_field_index += 1
			else:
				print('NODE NOT FOUND', A_field_vect[A_field_index], np.min(bands_weyl))
				wn = [kx[np.argmin(bands_weyl)]*a_lattice/(2*np.pi), ky[np.argmin(bands_weyl)]*b_lattice/(2*np.pi), kz[np.argmin(bands_weyl)]*c_lattice/(2*np.pi)]
				print(wn)
	
		wn = comm.bcast(np.array(wn),root=0)
		wn_list = comm.bcast(wn_list,root=0)
  
		if rank == 0:
			if len(node) > 1:
				print(f'[{ll-1}]: ', node[ll-1])
			saveMatrix(node, file_path = output+'.pkl')

	if rank == 0:
		print('number of proceses = ', size)
		print(f'Time: {time.time() - start}')
	return

# opens tha config file and loads the parameters then run the search
def main():

	# Initialize MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	parser = argparse.ArgumentParser() 
	parser.add_argument('-c','--config_file',type=str, help='dictionary with all the parameters of the program',required=True)
	parser.add_argument('-o','--output',type=str, help=' Provide the output Directory',required=True)

	args = vars(parser.parse_args())
	output = "../results/" + args['output']

	if rank == 0:
		print(output)

	stream = open(args['config_file'], 'r')
	dictionary = yaml.load_all(stream, yaml.Loader)

	for doc in dictionary:
		for key, value in doc.items():
			info = key + " : " + str(value)
			globals()[key] = value


	stream2 = open(args['config_file'], 'r')
	dictionary = yaml.load_all(stream2, yaml.Loader)
	if rank == 0:
		with open(output+'.txt', 'w') as file:
			for doc in dictionary:
				for key, value in doc.items():
					info = key + " : " + str(value)
					print(info)
					file.write(info + '\n')


	data           = read_file(filename)
	degeneracy     = np.array(read_degeneracy(filename))


	A_field_vect   = np.linspace(A_field_vect_initial, A_field_vect_end, A_field_vect_points)
	atom_position  = np.array(atom_pos, dtype = float)
	bands_per_atoms = np.array(bands_per_atom, dtype = float)

	if node_type == 'V':
		wn = np.array(wn_V, dtype=np.float64)
	if node_type == 'W':
		wn = np.array(wn_W, dtype=np.float64)

	delta = np.array(minsize, dtype=np.float64)

	if rank == 0:
		print(wn)
	run_search(comm, rank, size, wn, delta, minsize, lenght, fermi_energy, frq, A_field_vect, n_replicas, n_bands, total_bands, atom_position, bands_per_atoms, degeneracy, data, a_lattice, b_lattice, c_lattice, output)
	return

if __name__ == "__main__":
	main()
