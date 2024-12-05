# Code to calculate Bulk band structure of MoTe2 from W90 data and plot it
# (c) Juan Ignacio Aranzadi, 2024

import numpy as np
import os
import time
from mpi4py import MPI
from floquet_functions import *
import matplotlib.pyplot as plt 

# Calculate the bands in a given cut
def k_path(comm, rank, size, kx, ky, kz, lenght, fermi_energy, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice):
	bands = {}
	k_bz = {}
	c = np.zeros((2*n_replicas+1)*n_bands)

	start = time.time()

	integrals_floquet = floquet_integrals(frq, A, n_replicas, n_bands, total_bands, data, a_lattice, b_lattice, c_lattice)

	for m in range(rank, lenght, size):
		if m%(np.round(lenght)/20) == 0:
			print(np.round(m/lenght, 2))

		k = np.array([kx[m],ky[m],kz[m]])
		#print(a_lattice*kx[m]/(2*np.pi),b_lattice*ky[m]/(2*np.pi),c_lattice*kz[m]/(2*np.pi))
		k_bz[m] = k

		if n_replicas == 1:
			H = H_matrix_floquet_fast_v2(k, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet)
		else:
			H = H_matrix_floquet_fast(k, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice, integrals_floquet)

		eigenvals, eigenvects = np.linalg.eigh(H)
		bands[m] = np.transpose(eigenvals) - fermi_energy

		for ll in range((2*n_replicas+1)*n_bands):
			for j in range(n_bands):
				v = eigenvects[:, ll]
				c[ll] += np.absolute(v[j+n_bands*n_replicas])**2/(lenght)



	# Gather all local dictionaries at process 0
	global_bands = comm.gather(bands, root=0)
	global_k_bz = comm.gather(k_bz, root=0)
	global_color = comm.reduce(c, MPI.SUM)

	if rank == 0:
		for i in range(len(global_color)):
			global_color[i] = np.round(1-global_color[i],2)

	# Process 0 aggregates all dictionaries into one
	if rank == 0:
		print(f'Time: {time.time() - start}')
		bands_plot = {}
		k_bz_plot  = {}
		for d in global_bands:
			bands_plot.update(d)

		for d in global_k_bz:
			k_bz_plot.update(d)

		bands_plot = {k: bands_plot[k] for k in sorted(bands_plot.keys())}
		k_bz_plot = {k: k_bz_plot[k] for k in sorted(k_bz_plot.keys())}

		saveMatrix(bands_plot, file_path = f'../results/KPATH_bands_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.pkl')
		saveMatrix(k_bz_plot, file_path = f'../results/KPATH_1BZ_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.pkl')
		np.savetxt(f'../results/KPATH_color_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.txt',X= global_color)
	return

def line_equation(point1, point2):
	# Calculate the differences between the coordinates of the two points
	dx = point2[0] - point1[0]
	dy = point2[1] - point1[1]
	dz = point2[2] - point1[2]

	# Calculate the slope of the line in each dimension
	if dx != 0:
		m_x = dy / dx


	if dz != 0:
		m_z = dz / dx

	# Calculate the y-intercept (origin) of the line in each dimension
	if m_x != float('inf'):
		b_x = point1[1] - m_x * point1[0]

	if m_z != float('inf'):
		b_z = point1[2] - m_z * point1[0]

	return m_x, b_x, m_z, b_z


def plot_cut(n_bands, n_replicas, lenght, A, frq, a_lattice, b_lattice, c_lattice):
    
    # Load band data
    with open(f'../results/KPATH_bands_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.pkl', 'rb') as f:
        bands = pickle.load(f)

    with open(f'../results/KPATH_1BZ_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.pkl', 'rb') as f:
        k_bz = pickle.load(f)

    # Load coloring data for plotting
    color_plot = np.genfromtxt(f'../results/KPATH_color_floquet_{n_bands}_{n_replicas}_{lenght}_{frq}_{A}.txt')

    # Prepare arrays for k-points and band plots
    kx = np.zeros(lenght)
    ky = np.zeros(lenght)
    kz = np.zeros(lenght)
    bands_plot = np.zeros((lenght, n_bands * (2 * n_replicas + 1)))

    for key in k_bz:
        kx[key] = k_bz[key][0]
        ky[key] = k_bz[key][1]
        kz[key] = k_bz[key][2]

    for key in bands:
        bands_plot[key][:] = bands[key]

    # Calculate k-path for plotting
    k_path = (c_lattice / (2 * np.pi)) * np.linspace(kz[0], kz[-1], lenght)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot bands with grayscale coloring
    for i in range(210, 220):
        c = np.arctan(color_plot[i]) / (np.pi / 2) ** 0.5
        ax.plot(k_path, 1000 * np.array(bands_plot[:, i]), color=(c, c, c), zorder=1)

    # Set plot limits and labels
    ax.set_ylim(-0.02 * 1000, 1000 * 0.1)
    ax.set_ylabel('Energy (meV)', size=15)
    ax.tick_params(axis="both", direction="in")
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.savefig(f"Nodes_{A}field.png", dpi=300)
    plt.show()

    return bands_plot, kx, ky, kz

def main():
	# Initialize MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# OPENS THE DATA FILE
	filename = "../data/wannier90_hr_fullbands.dat"


	data = read_file(filename)
	degeneracy = read_degeneracy(filename)



	# ATOMIC PARAMETERS
	bands_per_atom = np.array([12, 12, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8])

	atom_pos = np.array([
	[0.000000000, 	3.833941936,	 6.946220398],
	[0.000000000, 	0.190683499,	 0.204635426],
	[1.738499999, 	2.501058102,	 0.004720220],
	[1.738499999, 	6.144316673,	 7.146135330],
	[0.000000000,	5.464380741,	 9.103638649],
	[0.000000000,	4.057250977,	 1.561282277],
	[0.000000000,	1.836960077,	11.930217743],
	[0.000000000,	1.368423343,	 5.590961933],
	[1.738499999,	0.870619118,	 2.162138462],
	[1.738499999,	2.277749300,	 8.502782822],
	[1.738499999,	4.498039722,	 4.988717556],
	[1.738499999,	4.966577053,	12.532462120]], dtype='float')

	# SIMULATION PARAMETERS

	total_bands = 112
	n_bands = 112
	fermi_energy = 13.2737 # eV

	a_lattice = 3.477           # A
	b_lattice = 6.335           # A
	c_lattice = 13.883          # A

	# Floquet parameters


	##############################################################################################
	##############################################################################################
	# Calculate the bands in the cut which unites the two distinct weyl nodes
	lenght = 401
	frq = 0.135                   # eV  frq of the driving
	n_replicas = 1            # number of replicas considered

	A = 0.011111111111111112
 
	# Position of the weyl nodes at A = 0.011111111111111112 field strenght in arbitrary units.
	wn_V = np.array([0.10366505166212552, 0.05771258564752207, -0.020536292869989887]) 
	wn_W = np.array([0.10796102698028429, 0.021750727432900477, 0.04186452273924465])
	m_x, b_x, m_z, b_z = line_equation(wn_V, wn_W)

	kx_0 = np.linspace(wn_V[0]-0.004, wn_W[0]+0.004,lenght)
	ky = 2*np.pi*(m_x * kx_0 + b_x)/b_lattice
	kz = 2*np.pi*(m_z * kx_0 + b_z)/c_lattice
	kx = 2*np.pi*kx_0/a_lattice


	k_path(comm, rank, size, kx, ky, kz, lenght, fermi_energy, frq, A, n_replicas, n_bands, total_bands, atom_pos, bands_per_atom, degeneracy, data, a_lattice, b_lattice, c_lattice)

	# Plot the bands
	if rank == 0:
		plot_cut(n_bands, n_replicas, lenght, A, frq, a_lattice, b_lattice, c_lattice)
	return

if __name__ == "__main__":
	main()



#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################




