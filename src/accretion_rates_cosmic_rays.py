# This file contains the functions used for the computation of the accretion
# of ions/electrons free in the plasma.
import numpy as np
from globals import *
import matplotlib.pyplot as plt
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
	global k_bolt
	# This term is integrated from Emin = 1.5e-2 eV to infinity
	Emin = 1.5e-2 # eV
	Emax = 1e10 # "infinity"
	xE = np.logspace(0, 10, 100000)
	print("xE = ", xE)
	#xE = np.linspace(Emin, Emax, 10000)
	Je_CR = lambda E: cr_spectrum(model_data, E) * (
					get_sticking_coef_CR(Grain, E) -
					get_second_yield_CR(model_data, E)) * 4 * np.pi

	# Evaluate direct integration vs trapz integration
	continuous = integrate.quad(Je_CR, Emin, Emax)
		
	# 
	yE = np.zeros(len(xE))
	for i in range(0,len(xE)):
		yE[i] = Je_CR(xE[i])
	discrete = integrate.trapz(yE,xE)
	
	print("Continuous:", continuous[0])
	print("Discrete:", discrete)
	print(integrate.quadrature(Je_CR, Emin, Emax))
	
	
	return(0.0)

def get_sticking_coef_CR(Grain, E_eV):
	"""
	This function computes the sticking efficiency of an electron (CR)
	as Ivlev+ (2015). 
	Input:
		Grain: DustGrain object
		E: energy (in keV)
	"""
	E_keV = E_eV*1e-3
	Re_E = 300*1e-8*np.power(Grain.solid_density, -0.85)*np.power(E_keV, 1.5) # cm
	max_val = 4*Grain.rad/3 # cm
	if Re_E < max_val:
		stick = 1.0
	else:
		stick = 0.0
	#else:
	#	stick = np.zeros(len(E_eV))
	#	stick[Re_E < max_val] = 1.0
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
		np.power(E_eV + E0,model_data["beta_cr_elec"]))
