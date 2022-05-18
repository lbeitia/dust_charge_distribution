# This file contains the functions used for the computation of the accretion
# of electrons free in the plasma that arise from CRs.
import numpy as np
from globals import *
from scipy import integrate


def J_accretion_CRs_elec(Grain, model_data):
	"""
	Function that computes the CR electron accretion rate of a given Grain
	Input:
		Grain: DustGrain object
		model_data: dictionary with input parameters
	Output:
		Je_CR: accretion rate of CR electrons
	"""
	# This term is integrated from Emin = 1.5e-2 eV to infinity
	# Discrete integration has been preferred over continuous because
	# the results for continuous integration were wrong.
	Emin = 1.5e-2 # eV
	xE_long = np.logspace(-3, 10, 5000)
	xE = xE_long[xE_long >= Emin]
	Je_CR = lambda E: cr_spectrum(model_data, E) * (
					get_sticking_coef_CR(Grain, E) -
					get_second_yield_CR(model_data, E)) * 4 * np.pi
	yE = np.zeros(len(xE))
	for i in range(0,len(xE)):
		yE[i] = Je_CR(xE[i]) * np.pi * np.power(Grain.rad, 2)
	discrete = integrate.trapz(yE,xE)
	return(discrete)


def get_sticking_coef_CR(Grain, E_eV):
	"""
	This function computes the sticking efficiency of an electron (CR)
	as Ivlev+ (2015). 
	Input:
		Grain: DustGrain object
		E: energy (in keV)
	"""
	E_keV = E_eV*1e-3
	# Re_E is taken from Draine & Salpeter (1979)
	Re_E = 300*1e-8*np.power(Grain.solid_density, -0.85)*np.power(E_keV, 1.5) # cm
	max_val = 4*Grain.rad/3 # cm
	if Re_E < max_val:
		stick = 1.0
	else:
		stick = 0.0
	return(stick)


def get_second_yield_CR(model_data, E_eV):
	"""
	This function gives the secondary electron yield for CRs.
	"""
	E_keV = E_eV*1e-3
	constant = model_data["deltae_max"]*E_keV/model_data["Emax"]
	exp_fac = np.exp(2 - 2*np.sqrt(E_keV/model_data["Emax"]))
	return(constant * exp_fac)


def cr_spectrum(model_data, E_eV):
	"""
	This function provides the CR spectrum presented in Ivlev+ (2015)
	"""
	E0 = 500 * 1e6 # 500 MeV, in eV
	return(model_data["C_cr_elec"] * np.power(E_eV,
										   model_data["alpha_cr_elec"]) /
		np.power(E_eV + E0, model_data["beta_cr_elec"]))
