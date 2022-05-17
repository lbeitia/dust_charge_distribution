# This file contains the functions used for the computation of
# the photoelectric emission of electrons induced by the cosmic rays
# spectrum

from globals import *
import numpy as np
from scipy import integrate
from dust_charge_distribution_classes import DustGrain




# # # # # # # # # # # # # # # # # #
#           Jpe CRs term          #
# # # # # # # # # # # # # # # # # #

def Jpe_CR(Grain, model_data, f_lin, f_spline):
	"""
	Photoelectric rate corresponding to photoelectric effect of CRs
	Input:
		zeta_CR: impact rate of CRs
		dust_albedo: between 0 and 1
		NH2_Av: influences the UV induced radiation field
		Rv: measurement of the extinction properties
	Output: 
		Jpe_CR: rate of photoelectrons released
	"""
	global h_planck
	global erg_eV, speed_of_light
	freq_max = 13.6/erg_eV/h_planck
	freq_min = 11.2/erg_eV/h_planck
	#f = lambda nu: Qabs_fun(speed_of_light*1e4/nu) * 
	#				PhotYield(Grain,nu,freq_pet,f_lin,f_spline)
	#x = np.linspace(freq_pet,freq_max,num=2000)
	#y = np.zeros(len(x))
	#i = 0
	#while i < len(x):
	#	y[i] = f(x[i])
		#i += 1
	return 0.0
