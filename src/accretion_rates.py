# This file contains the functions used for the computation of the accretion
# of ions/electrons free in the plasma.
import numpy as np
from globals import *



def J_accretion(Grain,Gas,species):
	"""
	Function that computes the ion/electron accretion rate of a given Grain
	in a given Gas.
	Input:
		Grain: DustGrain object
		Gas  : Gas object
		species: 1 for ions, -1 for electrons
	Output:
		Jspec: accretion rate for selected species
	"""
	global k_bolt
	Jacc = 0.0
	if species == 1:
		# Take into account multi-ionic mixture
		number_of_ions = len(Gas.dion)
		for numbi in range(0, number_of_ions):
			ni = Gas.dion[numbi]
			Ti = Gas.Tion[numbi]
			vi = np.sqrt(8*k_bolt*Ti/(np.pi*Gas.ion_mass[numbi]))
			Jhat = get_Jhat(Grain, Ti, species)
			sticking_coeff = get_sticking_coeff(species, Grain)
			Jacc += ni*sticking_coeff*vi*np.pi*np.power(Grain.rad,2)*Jhat
	elif species == -1:
		ni = Gas.delec
		vi = np.sqrt(8*k_bolt*Gas.Telec/(np.pi*Gas.elec_mass))
		Jhat = get_Jhat(Grain, Gas.Telec, species)
		sticking_coeff = get_sticking_coeff(species, Grain)
		Jacc += ni*sticking_coeff*vi*np.pi*np.power(Grain.rad,2)*Jhat
	else:
		raise ValueError("Species not available. Choose ions (1)" +
					"or electrons (-1)")
	
	return(Jacc)


def get_Jhat(Grain, Tspecies,species):
	"""
	Function that computes the scaling coefficient Jhat, following Draine & Sutin (1987).
	We consider that all species are singly charged
	Input:
		Grain: DustGrain object
		Gas: Gas object
	Output:
		Jhat: coefficient for computing the accretion rate
	"""
	global e_2
	rel_ch = Grain.Z/species # Ze/qi = Z/+-1
	red_temp = Grain.rad*k_bolt*Tspecies/e_2 # Reduced temperature akT/(qi^2)
	
	if rel_ch == 0:
		Jhat = 1+np.sqrt(np.pi/(2*red_temp))
	elif rel_ch < 0:
		Jhat = (1-rel_ch/red_temp)*(1+np.sqrt(2/(red_temp-2*rel_ch)))
	else:
		xi = 1+np.power(3*rel_ch,-0.5)
		th = rel_ch/xi - 1/(2*np.power(xi,2)*(np.power(xi,2)-1))
		Jhat = np.square(1+np.power(4*red_temp+3*rel_ch,-0.5))*np.exp(-th/red_temp)
	return Jhat


def get_sticking_coeff(species,Grain):
	"""
	Function that obtains the ion/electron sticking coefficient for a grain, following Weingartner & Draine (2001)
	We assume that ions, if they eventually collide, will always stick.
	Electron sticking will obey formulae (28-29-30)
	Input:
		species: index of species. -1 electron, 1 ion
		Grain: DustGrain object
	Output:
		sticking_prob: sticking probability. Floating between 0 and 1, near 0.5.
	"""
	global le
	if species == 1:
		sticking_prob = 1
	else:
		if Grain.Z <= 0:
			sticking_prob = 0.5*(1-np.exp(-Grain.rad/le))/(1+np.exp(20-Grain.Nc))
		else:
			sticking_prob = 0.5*(1-np.exp(-Grain.rad/le))
	return sticking_prob

