# This file contains the functions that define the stellar spectra
# used as a source for the photoelectric effect.

import pandas as pd
import numpy as np
from globals import *


# # # # # # # # # # # # # # # # # #
#  Interstellar Radiation Field   #
#  Mathis, Mezger & Panagia 1983  #   
# # # # # # # # # # # # # # # # # #

def MMP83(freq):
	"""
	Interstellar radiation field by Mathis, Mezger & Panagia (1983) with weights corrected as in
	Draine (2011).
	Input: 
		freq: frequency desired (Hz)
	Output:
		freq*rad_dens: radiation density (erg cm-3)
	""" 
	global h_planck
	global erg_eV
	global speed_of_light
	energy_eV = freq*h_planck*erg_eV
	if  energy_eV >= 13.6:
		return 0
	elif energy_eV < 13.6 and energy_eV >= 11.2:
		return 3.328e-9*np.power(energy_eV,-4.4172)
	elif energy_eV < 11.2 and energy_eV >= 9.26:
		return 8.463e-13*np.power(energy_eV,-1)
	elif energy_eV < 9.26 and energy_eV >= 5.04:
		return 2.005e-14*np.power(energy_eV,0.6678)
	else:
		return 1e-14*bb_edens(freq,7500)+1.65e-13*bb_edens(freq,4000)+7e-13*bb_edens(freq,3000)


def bb_edens(freq,T):
	"""
	Formula of the Planck's BlackBody spectrum's energy density in frequencies.
	Input:
		freq: frequency (Hz)
		T: temperature (K)
	Output:
		rad_dens: radiation density (ergs cm-3)
	"""
	global h_planck
	global speed_of_light
	global k_bolt
	num = 8*np.pi*np.power(freq,4)*h_planck
	denom = np.power(speed_of_light,3)*(np.exp(h_planck*freq/(k_bolt*T))-1)
	return num/denom


# # # # # # # # # # # # # # # # # #
#     Solar Radiation Field       #
#    Based on STIS Sun spectrum   #   
# # # # # # # # # # # # # # # # # #


def SunRF(freq):
	"""
	Solar radiation field that reaches the Earth as observed by STIS.
	The original spectrum covers the wavelength range
	1195 - 26957.4 Angstroms. For intermediate
	values, an interpolation is performed.
	For wavelengths longer than 26957.4 A, a blackbody
	with T = 5780 K is considered (and scaled to the Earth's
	distance).
	For wavelengths shorter than 1195 A, we consider that the photons
	are effectively scattered so the radiation field is set to 0.
	"""
	global h_planck
	global erg_eV
	global speed_of_light
	energy_eV = freq*h_planck*erg_eV
	# Read the data
	sundat = pd.read_csv("src/Sun_STIS.csv")
	frequencies = speed_of_light/(sundat["wavelength"]*1e-8) # Wavelength is in Angstroms
	flux_freq = sundat["flux"]*3e18/np.power(frequencies,2)
	# Compute the energy density
	u_nu = flux_freq/speed_of_light
	nu_times_u_nu = u_nu*frequencies
	# And choose the functions to extrapolate in case the values are out of range
	minfreq = np.min(frequencies)
	maxfreq = np.max(frequencies)
	minpoint = np.where(frequencies == np.min(frequencies))[0][0]
	scale_factor = nu_times_u_nu[minpoint]/bb_edens(frequencies[minpoint],5750)
	if freq > maxfreq:
		return(0)
	elif freq < minfreq:
		return(bb_edens(freq,5750)*scale_factor)
	else:
		if freq in frequencies:
			return(nu_time_u_nu[freq == frequencies])
		# Linear interpolation
		else:
			# Frequencies are ordered in decreasing range
			up = np.where(freq > frequencies)[0][0]
			low = up - 1
			y2 = nu_times_u_nu[up]
			y1 = nu_times_u_nu[low]
			x2 = frequencies[up]
			x1 =frequencies[low]
			return(((y2-y1)/(x2-x1))*(freq - x1) + y1)
		
	

# ----------------------------------------------------------------------------# 
#        Determine radiation field 
# ----------------------------------------------------------------------------#
def determine_radiation_field(rf, intensity):
	"""
	This function constructs the radiation field function
	"""
	if int(rf) == 1:
		return(lambda nu: intensity*MMP83(nu))
	elif int(rf) == 2:
		return(lambda nu: intensity*SunRF(nu))
	else:
		raise ValueError("Only MMP83 (1) and Solar (2) radiation fields " + 
				   "have been implemented up to now.")
