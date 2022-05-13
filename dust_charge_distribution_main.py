
from scipy import integrate
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'src/')
from data_interpolation import *
from dust_charge_distribution_classes import DustGrain, Gas


k_bolt = 1.3807e-16 # Boltzmann's constant, erg/K
le = 10e-8 # Electron length, in cm.  Taken to be 10A.
e_2 = 23.0708e-20 # squared elementary charge, in esu
h_planck = 6.6260755e-27 # Planck constant, erg s
erg_eV = 6.2415e+11 # conversion, 1erg = 6.2415e+11 eV
speed_of_light = 2.99792458e+10 # Speed of light in vacuum, cm s-1

# # # # # # # # # # # # # # # # # #
#          Jion and Je            #   
# # # # # # # # # # # # # # # # # # 

def J_accretion(Grain,Gas,species):
	"""
	Function that computes the ion/electron accretion rate of a given Grain in a given Gas.
	Input:
		Grain: DustGrain object
		Gas  : Gas object
		species: 1 for ions, -1 for electrons
	Output:
		Jspec: accretion rate for selected species
	"""
	global k_bolt
	if species == 1:
		ni = Gas.dion
		vi = np.sqrt(8*k_bolt*Gas.Tion/(np.pi*Gas.ion_mass))
	elif species == -1:
		ni = Gas.delec
		vi = np.sqrt(8*k_bolt*Gas.Telec/(np.pi*Gas.elec_mass))
	else:
		raise ValueError("Species not available. Choose ions (1) or electrons (-1)")
	Jhat = get_Jhat(Grain,Gas,species)
	sticking_coeff = get_sticking_coeff(species,Grain)
	return ni*sticking_coeff*vi*np.pi*np.power(Grain.rad,2)*Jhat


def get_Jhat(Grain,Gas,species):
	"""
	Function that computes the scaling coefficient Jhat, following Draine & Sutin (1987).
	We consider that ions are singly charged, because my initial problem only considers gas of purely H.
	Input:
		Grain: DustGrain object
		Gas: Gas object
	Output:
		Jhat: coefficient for computing the accretion rate
	"""
	global e_2
	rel_ch = Grain.Z/species # Ze/qi = Z/+-1
	if species == 1:
		red_temp = Grain.rad*k_bolt*Gas.Tion/e_2 # Reduced temperature akT/(qi^2)
	elif species == -1:
		red_temp = Grain.rad*k_bolt*Gas.Telec/e_2 # Reduced temperature akT/(qi^2)
	else:
		raise ValueError("Species not available. Choose ions (1) or electrons (-1)")
	
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
	sundat = pd.read_csv("Sun_STIS.csv")
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
		
	



# # # # # # # # # # # # # # # # # #
#       Jpe conduction term       #
# # # # # # # # # # # # # # # # # #

def Jpe_cond(Grain,Gas,ISRF):
	"""
	Photoelectric rate corresponding to photodetachment of conduction electrons.
	It depends on the photodetachment cross section.
	Input:
		Grain: DustGrain object
		Gas:   Gas object
	Output:
		Jpe_cond: photodetachment rate of the conduction band
	"""
	global h_planck, speed_of_light
	global erg_eV 
	if Grain.Z >= 0:
		return 0
	else:
		freq_max = Gas.max_energy/erg_eV/h_planck
		freq_pdt = get_freq_pdt(Grain)
		f = lambda nu: speed_of_light*sigma_pdt(Grain,nu)*ISRF(nu)/(h_planck*np.square(nu))
		# Try to avoid integration problems
		nptaux = 1000
		xxx = np.linspace(freq_pdt,freq_max,nptaux)
		yyy = np.zeros(nptaux)
		for i in range(0,nptaux):
			yyy[i] = f(xxx[i])		
		#return integrate.quad(f,freq_pdt,freq_max,limit=100)[0]
		return np.trapz(yyy,xxx)

def sigma_pdt(Grain,freq):
	"""
	Function that returns the photodetachment cross section of a negatively charged grain.
	We use Eq. 20 from Weingartner & Draine (2001)
	Input: 
		freq: frequency of the photon (erg)
	Output:
		sigma_pdt: photodetachment cross section
	"""
	global h_planck
	freq_pdt = get_freq_pdt(Grain)
	x = h_planck*(freq-freq_pdt)/(3/erg_eV)
	return 1.2e-17*np.absolute(Grain.Z)*x/np.square(1+np.square(x)/3)

def get_freq_pdt(Grain):
	"""
	Function that obtains the minimum frequency required for photodetachment to occur.
	Eq. 18 from Weingartner & Draine (2001)
	Input:
		Grain: DustGrain object
	Output:
		freq_pdt: minimum frequency (Hz)
	"""
	global h_planck
	Grain_aux = DustGrain(Grain.rad*1e4,Grain.Z + 1, Grain.material)
	return (Grain_aux.EA + Grain.Emin)/h_planck


# # # # # # # # # # # # # # # # # #
#         Jpe valence term        #
# # # # # # # # # # # # # # # # # #

def Jpe_val(Grain,Gas,f_lin,f_spline,Qabs_fun,ISRF):
	"""
	Photoelectric rate corresponding to photoelectric effect of valence electrons.
	It depends on the absorption coefficiency, photoelectric yield and radiation field.
	Input:
		Grain: DustGrain object
		Gas: Gas object
	Output: 
		Jpe_val: rate of valence photoelectrons released
	"""
	global h_planck
	global erg_eV,speed_of_light
	freq_max = Gas.max_energy/erg_eV/h_planck
	freq_pet = get_freq_pet(Grain)
	f = lambda nu: speed_of_light*ISRF(nu)*Qabs_fun(speed_of_light*1e4/nu)*PhotYield(Grain,nu,freq_pet,f_lin,f_spline)/(h_planck*np.square(nu))
	x = np.linspace(freq_pet,freq_max,num=2000)
	y = np.zeros(len(x))
	i = 0
	while i < len(x):
		y[i] = f(x[i])
		i += 1
	return np.pi*np.square(Grain.rad)*integrate.trapz(y,x)




def get_freq_pet(Grain):
	"""
	This function returns the photoelectric threshold frequency for a given grain.
	Input:
		Grain: DustGrain object
	Output:
		freq_pet: photoelectric threshold (erg)
	"""
	global h_planck
	if Grain.Z >= -1:
		return Grain.IPv/h_planck
	else:
		return (Grain.IPv + Grain.Emin)/h_planck

def PhotYield(Grain,freq,freq_pet,f_lin,f_spline):
	"""
	This function computes the photoelectric yield. It is the probability that a photoelectron
	is emitted.
	Input:
		Grain: Dust Grain object
		freq_pet: photoelectric threshold
		freq:  frequency (erg)
	Output:
		phot_yield: photoelectric yield
	"""
	global h_planck, e_2
	# y0 depends on an additional parameter theta
	if Grain.Z >= 0:
		theta = h_planck*(freq-freq_pet) +  (Grain.Z+1)*e_2/Grain.rad
	else:
		theta = h_planck*(freq-freq_pet)
	return y2(Grain,freq,freq_pet)*min(y0(Grain,theta)*y1(Grain,freq,f_lin,f_spline),1)


def y0(Grain,theta):
	"""
	Parameter needed to compute the fraction of electrons that are converted into photoelectrons.
	It depends on the Grain charge and the material.
	Input:
		Grain: Dust Grain object
		theta: parameter (erg)
	Output:
		y0: charge factor of the photoelectric yield
	"""
	if Grain.material == "silicate":
		y0 = 0.5*(theta/Grain.W)/(1+5*theta/Grain.W)
	else: # Carbonaceous
		y0 = 9e-3*np.power(theta/Grain.W,5)/(1+3.7e-2*np.power(theta/Grain.W,5))
	return y0


def y2(Grain,freq,freq_pet):
	"""
	Parameter that accounts for the fraction of attempting electrons that finally escapes.
	Computed using Eq. 11 by Weingartner & Draine (2001).
	Input:
		Grain: DustGrain object
		freq: frequency (Hz)
		freq_pet: photoelectric threshold frequency (Hz)
	Output:
		y2: fraction of attempting electrons that escapes.
	"""
	global h_planck, e_2
	if Grain.Z >= 0:
		Ehigh = h_planck*(freq-freq_pet)
		Elow =  -(Grain.Z+1)*e_2/Grain.rad
		return np.square(Ehigh)*(Ehigh-3*Elow)/np.power(Ehigh-Elow,3)
	else:
		return 1

def y1(Grain,freq,f_lin,f_spline):
	"""
	Parameter needed to compute the fraction of electrons that are converted into photoelectrons.
	It accounts for a geometric factor. Depends on the grain radius and material.
	It is assumed that the escape length of electrons is 10A.
	It is also assumed that the photon attenuation length depends on a refractive index computed by Draine.
	Computed using Eq. 13 by Weingartner & Draine (2001).
	Input:
		Grain: DustGrain object
		freq: frequency (Hz)
	Output:
		y1: geometric factor of the photoelectric yield
	"""
	la = get_attenuation_length(Grain,freq,f_lin,f_spline) # Photon attenuation length, cm
	le = 10e-8 # Electron escape length, 10 A, in cm
	alpha = Grain.rad/la + Grain.rad/le
	beta = Grain.rad/la
	return np.square(beta/alpha)*(np.square(alpha)-2*alpha+2-2*np.exp(-alpha))/(np.square(beta)-2*beta+2-2*np.exp(-beta))

def get_attenuation_length(Grain,freq,f_lin,f_spline):
	"""
	This function return the attenuation length at a given frequency. We will use the refractive indexes given by Draine
	in https://www.astro.princeton.edu/~draine/dust/dust.diel.html
	Dust grains can be astrosilicates or carbonaceous, and the latter type is further subdivided into
	graphitic grains and PAHs.
	For the time being, I limit myself to astrosilicates. Graphite has been included under the tag 'carbonaceous' 
	(12/10/2020).
	Input:
		Grain: DustGrain object
		freq: frequency (Hz)
	Output:
		la: attenuation length (cm)
	"""
	global speed_of_light
	
	# We are interested in columns 0 and 4, named 'wave(um)' and 'Im(n)', respectively.
	# We are working with UV radiation, so I will consider only refractive index data with 
	# wavelengths between 0 and 1 microns.

	lam = (speed_of_light/freq)*1e4 # Desired frequency is converted to wavelength in microns
	if Grain.material == "carbonaceous":
		# 1/3 - 2/3 approximation
		Im_n = get_Im_n(lam,f_lin[0],f_spline[0])/3 + get_Im_n(lam,f_lin[1],f_spline[1])*2/3
	else:
		Im_n = get_Im_n(lam,f_lin,f_spline)
	return lam*1e-4/(4*np.pi*Im_n)
	
def get_Im_n(lam,f_lin,f_spline):
	"""
	Function that computes the refractive index at a given wavelength. 
	It will only work properly for wavelengths lower than 1 micron.
	This function has only been tested for astrosilicate's refractive index.
	Input:
		lam: wavelength (microns)
	Output:
		Im_n: imaginary part of the refractive index
	"""
	if lam <= 0.2:
		return  f_spline(lam)
	elif lam > 0.2: #and lam < 1:
		return f_lin(lam)
	else:
		raise ValueError("The frequency given corresponds to a wavelength greater than 1 micron")

# # # # # # # # # # # # # # # # # #
#         Read Input file         #
# # # # # # # # # # # # # # # # # #
def read_input_file():
	"""
	Function that reads the parameters needed to perform the simulation.
	We need dust properties, gas properties, and a radiation field.
	Output:
		param: dictionary containing the data.
	"""
	param = {}
	floats_are = ['rad','dens','frac_ion','rf_intensity']
	ints_are = ['Z0','Zmin','Zmax','T','rf']
	with open('input_file.txt') as input_file:
		for line in input_file:
			if "#" not in line and len(line)>1:
				new_line = line.rstrip().replace(" ","").split("=")
				if new_line[0] in floats_are:
					param[new_line[0]] = float(new_line[1])
				elif new_line[0] in ints_are:
					param[new_line[0]] = int(new_line[1])
				else:
					param[new_line[0]] = new_line[1]
	return(param)

# # # # # # # # # # # # # # # # # #
#      Charge distribution        #
# # # # # # # # # # # # # # # # # #
def dust_charge_distribution(Gas,rad,Z0,material,Zmin,Zmax,f_lin,f_spline,Qabs_fun,ISRF):
	"""
	Program that computes the charge distribution of a population of dust grains in the interval
	[Zmin,Zmax].
	It applies recursively equation 21 from Weingartner & Draine 2001.
	Input:
	        ISRF: interstellar radiation field
	Output:
		probabilities: array containing the probabilities for each value Z, ordered with increasing Z.
	"""
	prob_neg = np.array([1])
	prob_pos = np.array([1])
	Z = Z0 + 1
	while Z <= Zmax:
		print("Computing the terms for Z =",Z)
		J_pe = Jpe_val(DustGrain(rad,Z-1,material),Gas,f_lin,f_spline,Qabs_fun,ISRF) + Jpe_cond(DustGrain(rad,Z-1,material),Gas,ISRF)
		J_ion = J_accretion(DustGrain(rad,Z-1,material),Gas,1)
		J_electron = J_accretion(DustGrain(rad,Z,material),Gas,-1)
		prob_pos = np.append(prob_pos,(J_pe+J_ion)*prob_pos[Z-Z0-1]/J_electron)
		Z +=1
	Z = Z0 - 1
	while Z >= Zmin:
		print("Computing the terms for Z =",Z)
		J_pe = Jpe_val(DustGrain(rad,Z,material),Gas,f_lin,f_spline,Qabs_fun,ISRF) + Jpe_cond(DustGrain(rad,Z,material),Gas,ISRF)
		J_ion = J_accretion(DustGrain(rad,Z,material),Gas,1)
		J_electron = J_accretion(DustGrain(rad,Z+1,material),Gas,-1)
		prob_neg = np.append(prob_neg,prob_neg[Z0-Z-1]*J_electron/(J_pe+J_ion))
		Z-=1
	dust_distrib = np.delete(prob_neg,0)
	dust_distrib = dust_distrib[::-1]
	dust_distrib = np.append(dust_distrib,prob_pos)
	dust_distrib = dust_distrib/sum(dust_distrib)
	return dust_distrib
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


import matplotlib.pyplot as plt

def main():
	"""
	Main function that computes the charge probability distribution of a given population of dust grains.
	It reads the input from an input file, DD_input_file.txt. 
	Output: 
		dust_distrib.txt: file containing the values of Z with the corresponding probability
	"""
	# read input data
	model_data = read_input_file()
	# Refractive index functions. In this way, I only compute them once.
	# It is necessary that the wavelength cut is set to 1 micron if we want to use spline interpolation
	# at short wavelengths. If you change it, be careful and check that your interpolation is not noisy.
	# See get_Im_n for more information.
	[f_lin, f_spline] = interpolate_refractive_indices(model_data["material"])

	# Absorption coefficient function must be defined before the main loop because in that way, it will be computed
	# only once, being more efficient.
	Qabs_fun = Qabs(model_data['material'],model_data['rad'])
	#GAS
	gas = Gas(model_data["composition"],model_data["T"],model_data["dens"],model_data["frac_ion"])
	# ISRF
	if model_data["rf"] == 1:
		ISRF = lambda nu: model_data["rf_intensity"]*MMP83(nu)
	elif model_data["rf"] == 2:
		ISRF = lambda nu: model_data["rf_intensity"]*SunRF(nu)
	else:
		raise ValueError("Only MMP83 (1) and Solar (2) radiation fields have been implemented up to now.")

	# Main Body
	probabilities = dust_charge_distribution(gas,model_data["rad"],model_data["Z0"],model_data["material"],model_data["Zmin"],model_data["Zmax"],f_lin,f_spline,Qabs_fun,ISRF)
	Z_values = np.arange(model_data["Zmin"],model_data["Zmax"]+1)
	df = pd.DataFrame.from_dict({"Z":Z_values,"prob":probabilities})
	df.to_csv('DustCharge_Distribution.txt',sep="\t",index=False)
	fig = plt.figure()
	plt.plot(Z_values,probabilities,lw=2,color='b')
	plt.xlabel('Z')
	plt.ylabel('f(Z)')
	plt.title("Dust grain "+str(model_data["rad"])+" microns")
	plt.tight_layout()
	fig.savefig("Probabilities.eps")
	plt.close(fig)
	print("Chimpún")


if __name__ == "__main__":
	main()
