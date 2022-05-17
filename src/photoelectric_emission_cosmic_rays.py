# This file contains the functions used for the computation of
# the photoelectric emission of electrons induced by the cosmic rays
# spectrum

from globals import *
import numpy as np
from scipy import integrate
from dust_charge_distribution_classes import DustGrain
from photoelectric_emission_stellar import *



# # # # # # # # # # # # # # # # # #
#           Jpe CRs term          #
# # # # # # # # # # # # # # # # # #

def Jpe_CR(Grain, model_data, f_lin, f_spline, Qabs_fun, F_UV):
	"""
	Photoelectric rate corresponding to photoelectric effect of CRs
	Input:
		zeta_CR: impact rate of CRs
		dust_albedo: between 0 and 1
		NH2_Av: influences the UV induced radiation field
		Rv: measurement of the extinction properties
		F_UV = average UV radiation field produced by CRs
	Output: 
		Jpe_CR: rate of photoelectrons released
	"""
	global h_planck
	global erg_eV, speed_of_light
	freq_max_CR = 13.6/erg_eV/h_planck
	freq_min_CR = 11.2/erg_eV/h_planck
	freq_pet = get_freq_pet(Grain)
	freq_min_global = np.maximum(freq_min_CR, freq_pet)
	# Function to integrate
	f = lambda nu: Qabs_fun(speed_of_light*1e4/nu) * PhotYield(Grain,nu,freq_pet,f_lin,f_spline)
	# Perform integration
	x = np.linspace(freq_min_global,freq_max_CR,num=2000)
	y = np.zeros(len(x))
	for i in range(0,len(x)):
		y[i] = f(x[i])
	ynu_qabs_avg = np.mean(y)
	return np.pi*np.square(Grain.rad)*F_UV*ynu_qabs_avg


def get_F_UV_CR(model_data):
	"""
	This function returns the averaged flux in the UV range
	given by Ivlev+(2015) due to H2 photoluminiscence
	Output:
		F_UV: in photons cm-2 s-1
	"""
	dust_term = 1/(1 - model_data["dust_albedo"])
	CR_rate_term = model_data["zeta_CR"]*1e17
	coldens_term = model_data["NH2_Av"]*1e-21
	Rv_term = np.power(model_data["Rv"]/3.2,1.5)
	return(960 * dust_term * CR_rate_term * coldens_term * Rv_term)
