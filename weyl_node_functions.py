# This code contains several functions to construct a tightbanding Hamiltonian from Wannier90 output
# also computes the Floquet Hamiltonian in the reapeted scheme, see Eq. (18) of https://journals.aps.org/prx/pdf/10.1103/PhysRevX.3.031005
# (c) Juan Ignacio Aranzadi, 2024


# This sets the number of threads used by numpy to 1 assuming that the main part of the code is already paralellized.
from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

# Used libraries
import numpy as np 
from numba import njit
import pickle 

# Read the W90.dat file using numpy's genfromtxt function
def read_file(file_path):
	types = ['int32', 'int32' ,'int32' , 'int32', 'int32' ,np.float64,np.float64] 
	matrix = np.genfromtxt(file_path, skip_header = 13, dtype=types) 
	return matrix

# Read the degeneracy from W90.dat file
def read_degeneracy(file_path):
	with open(file_path, 'r') as f:
		# skip the header (1 line)
		f.readline()
		nwann = int(f.readline().strip())
		nrpt = int(f.readline().strip())
		# the degeneracy of R points
		nline = nrpt // 15 + 1
		tmp = []
		for i in range(nline):
			tmp.extend(f.readline().strip().split())
		tmp = [int(item) for item in tmp]
		deg_rpt = np.array(tmp, dtype='float')
	return deg_rpt

# Saves the Hamiltonian in matrix form
def saveMatrix(matrix, file_path = 'matrix_output.pkl', header = ""):
	# Save the matrix and header to a file_path.txt file
	with open(file_path, 'wb') as f:
		pickle.dump(dict(matrix), f)
	print(f"Matrix has been saved to {file_path}")
	return

@njit
def dagger(A):
	return np.conjugate(A.T)

# Each vector contains x - y - z - i - j - Re(t) - Im(t)
# Creates the complex k-dependent H matrix
@njit
def H_to_integrate_full(t, frq, position, A, m, n):
	field = A*np.array([np.sin(frq*t), np.cos(frq*t), 0])
	dot   = np.dot(field, position)
	return (np.e**(1j*(m-n)*frq*t) * np.e**(-1j*dot))


# Integrates using the rectangle rule
@njit
def integrate_rectangles(func, a, b, points, args):
	t = np.linspace(a, b, points)
	result = 0 + 0j
	for i in range(points-1):
		result += func(t[i+1], *args)*(t[i+1]-t[i])
	return result

# Creates a matrix with all the floquet integrals for a given Hamiltonian with m replicas. 
# This is independent of k so it is calculated only once to improve performance
@njit
def floquet_integrals(frq, A, n_replicas, n_bands, total_bands, data, a_lattice, b_lattice, c_lattice):
	tau = 2*np.pi/frq
 
	integrals_floquet = np.zeros((len(data[:]), (2*n_replicas+1),(2*n_replicas+1)), dtype=np.complex128)
	# Creates the integral matrix
	for m in range(0,2*n_replicas+1):
		for n in range(0,2*n_replicas+1):

			for i in range(0,len(data[:])):
				# Only computes the phase when the position vector changes
				# These if conditions are for just taking n_bands
				if i%total_bands < n_bands and i%(total_bands)**2 < n_bands*total_bands:
					if i%(total_bands)**2 == 0:
						rx = a_lattice * float(data[i][0])
						ry = b_lattice * float(data[i][1])
						rz = c_lattice * float(data[i][2])

						position = np.array([rx, ry, rz])
						result = 1/tau * integrate_rectangles(H_to_integrate_full, 0, tau, 30, args=(frq, position, A, m-n_replicas, n-n_replicas))
						
					integrals_floquet[i, m, n] += result

	return integrals_floquet

# Calculates the phases between different atoms
@njit
def atom_phases(k, n_bands, atom_pos, bands_per_atom):
	# Initialize matrix of phases
	atomic_phase = np.zeros((n_bands,n_bands),dtype=np.complex128)

	# Creates a list to compare the orbitals
	atom_check = np.array([1], dtype=np.float64)
	for h in range(0, len(bands_per_atom)):
		atom_check = np.append(atom_check, bands_per_atom[h]+atom_check[h])


	for orbital1 in range(1, n_bands+1):
		for orbital2 in range(1, n_bands+1):
			for idx in range(len(atom_check) - 1):
				if atom_check[idx] <= orbital1 < atom_check[idx + 1]:
					R1 = atom_pos[idx]
				if atom_check[idx] <= orbital2 < atom_check[idx + 1]:
					R2 = atom_pos[idx]
			atomic_phase[orbital1-1, orbital2-1] += np.e**(1j*np.dot(k, R2 - R1))

	return atomic_phase

# Calculates the floquet Hamiltonian in the reapeted zone scheme for an arbitrary number of replicas
@njit
def H_matrix_floquet_fast(k, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet):
	# Initializes position vector and H matrix
	H = np.zeros(((2*n_replicas+1)*n_bands,(2*n_replicas+1)*n_bands), dtype=np.complex128)
	atomic_phase = atom_phases(k, n_bands, atom_pos, bands_per_atom)

	# Fills the H matrix for one value of k
	
	for m in range(0,2*n_replicas+1):
		for n in range(m,2*n_replicas+1):
			rpoint = 0
			for i, vect in enumerate(data):
				# Only computes the phase when the position vector changes

				# These if conditions are for just taking n_bands
				if i%total_bands < n_bands and i%(total_bands)**2 < n_bands*total_bands:
					if i%(total_bands)**2 == 0:
						rx = a_lattice * float(vect[0])
						ry = b_lattice * float(vect[1])
						rz = c_lattice * float(vect[2])

						position = np.array([rx, ry, rz])
						phase = np.e**(1j*np.dot(position, k))/degeneracy[rpoint]
						rpoint += 1

					ii = vect[3] 
					jj = vect[4] 		# Indices of H matrix
					t_r = vect[5]      # Complex hopping coefficients
					t_i = vect[6]
					# For each m value I have to perform an integral and also from 0 to 2pi/frq and add the constant term
					complex_t = (t_r + t_i*1j) * phase * atomic_phase[int(ii)-1, int(jj)-1]

					H[int(n_bands*m+(ii-1)), int(n_bands*n+(jj-1))] += complex_t * integrals_floquet[i, int(m), int(n)]
			if m != n:
				H[int(n_bands*n):int(n_bands*(n+1)),int(n_bands*m):int(n_bands*(m+1))] += dagger(H[int(n_bands*m):int(n_bands*(m+1)), int(n_bands*n):int(n_bands*(n+1))])
	# adds the frequency term when m==n and i == j
	z = 0
	for m in range(2*n_replicas+1):
		for i in range(n_bands):
			H[m*n_bands+i, m*n_bands+i] +=  (z-n_replicas)*frq
		z += 1

	return H

# Calculates the floquet Hamiltonian in the reapeted zone scheme only for the subspaces with 0 ≤ |m| ≤ 1
# NOTE: in this m range this function is faster than the previous one
@njit
def H_matrix_floquet_fast_v2(k, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet):
	# Initializes position vector and H matrix
	if n_replicas != 1:
		print('WARNING: This function cant only handle 1 replica')
	H = np.zeros(((2*n_replicas+1)*n_bands,(2*n_replicas+1)*n_bands), dtype=np.complex128)
	atomic_phase = atom_phases(k, n_bands, atom_pos, bands_per_atom)	

	# Fills the H matrix for one value of k
	rpoint = 0
	i = 0
	for vect in data:
		# Only computes the phase when the position vector changes
		if i%(total_bands)**2 == 0:
			rx = a_lattice * float(data[i][0])
			ry = b_lattice * float(data[i][1])
			rz = c_lattice * float(data[i][2])
			position = np.array([rx, ry, rz])
			phase = np.e**(1j*np.dot(position, k))/degeneracy[rpoint]
			rpoint += 1

		ii  = vect[3] 
		jj  = vect[4] 		# Indices of H matrix
		t_r = vect[5]      # Complex hopping coefficients
		t_i = vect[6]
  
		# For each m value I have to perform an integral and also from 0 to 2pi/frq and add the constant term
		complex_t = (t_r + t_i*1j) * phase * atomic_phase[(ii)-1, (jj)-1]

		H[(n_bands*0+(ii-1)), (n_bands*0+(jj-1))] += complex_t * integrals_floquet[i, (0), (0)]
		H[(n_bands*1+(ii-1)), (n_bands*1+(jj-1))] += complex_t * integrals_floquet[i, (1), (1)]
		H[(n_bands*2+(ii-1)), (n_bands*2+(jj-1))] += complex_t * integrals_floquet[i, (2), (2)]
		H[(n_bands*0+(ii-1)), (n_bands*1+(jj-1))] += complex_t * integrals_floquet[i, (0), (1)]
		H[(n_bands*0+(ii-1)), (n_bands*2+(jj-1))] += complex_t * integrals_floquet[i, (0), (2)]
		H[(n_bands*1+(ii-1)), (n_bands*2+(jj-1))] += complex_t * integrals_floquet[i, (1), (2)]
		i += 1
   
	H[(n_bands*1):(n_bands*(1+1)),(n_bands*0):(n_bands*(0+1))] += dagger(H[(n_bands*0):(n_bands*(0+1)), (n_bands*1):(n_bands*(1+1))])
	H[(n_bands*2):(n_bands*(2+1)),(n_bands*0):(n_bands*(0+1))] += dagger(H[(n_bands*0):(n_bands*(0+1)), (n_bands*2):(n_bands*(2+1))])
	H[(n_bands*2):(n_bands*(2+1)),(n_bands*1):(n_bands*(1+1))] += dagger(H[(n_bands*1):(n_bands*(1+1)), (n_bands*2):(n_bands*(2+1))])

	# adds the frequency term when m==n and i == j
	z = 0
	for m in range(2*n_replicas+1):
		for i in range(n_bands):
			H[m*n_bands+i, m*n_bands+i] +=  (z-n_replicas)*frq
		z += 1
	return H




###########################	##############################################################################################################
#########################################################################################################################################
#########################################################################################################################################




