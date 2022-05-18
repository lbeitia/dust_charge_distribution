# This file contains the functions used for the computation of
# the photoelectric emission of electrons induced by the stellar
# spectrum

from globals import *
import numpy as np
from scipy import integrate
from dust_charge_distribution_classes import DustGrain

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
		nptaux = 5000
		xxx = np.linspace(freq_pdt,freq_max,nptaux)
		yyy = np.zeros(nptaux)
		for i in range(0,nptaux):
			yyy[i] = f(xxx[i])		
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
	Grain_aux = DustGrain(Grain.rad*1e4,Grain.Z + 1, Grain.material, Grain.solid_density)
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
	x = np.linspace(freq_pet,freq_max,num=5000)
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

