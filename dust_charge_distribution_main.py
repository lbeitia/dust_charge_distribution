
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'src/')
from data_interpolation import *
from dust_charge_distribution_classes import DustGrain, Gas
from accretion_rates import *
from stellar_spectra import *
from photoelectric_emission_stellar import *


# # # # # # # # # # # # # # # # # #
#         Read Input file         #
# # # # # # # # # # # # # # # # # #
def read_input_file():
	"""
	Function that reads the parameters needed to perform the simulation.
	We need dust properties, gas properties, and a radiation field.
	Also Cosmic Rays.
	Output:
		param: dictionary containing the data.
	"""
	param = {}
	with open('input_file.txt') as input_file:
		for line in input_file:
			if "#" not in line and len(line)>1:
				[key, value_stn] = line.rstrip().replace(" ","").split("=")
				value = determine_value_format(value_stn)
				param[key] = value
	return(param)


def determine_value_format(stn):
	"""
	This function tries to convert the value stored in stn into a float.
	Otherwise returns the original string removing trailing spaces.
	"""
	# Distinguish between string and float
	try:
		outval = float(stn)
	except ValueError:
		outval = stn.strip()
	# Also consider numpy arrays
	if '[' in stn:
		newstn = stn.replace("[","")
		newstn = newstn.replace("]","")
		newstn = newstn.split(",")
		outval = np.zeros(len(newstn), dtype = "int")
		for i in range(0, len(newstn)):
			outval[i] = int(newstn[i])
	return(outval)


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
	print(model_data)
	# Refractive index functions. In this way, I only compute them once.
	# It is necessary that the wavelength cut is set to 1 micron if we want to use spline interpolation
	# at short wavelengths. If you change it, be careful and check that your interpolation is not noisy.
	# See get_Im_n for more information.
	#[f_lin, f_spline] = interpolate_refractive_indices(model_data["material"])

	# Absorption coefficient function must be defined before the main loop because in that way, it will be computed
	# only once, being more efficient.
	#Qabs_fun = Qabs(model_data['material'],model_data['rad'])
	#GAS
	#gas = Gas(model_data["composition"],model_data["T"],model_data["dens"],model_data["frac_ion"])
	# ISRF
	#if model_data["rf"] == 1:
	#	ISRF = lambda nu: model_data["rf_intensity"]*MMP83(nu)
	#elif model_data["rf"] == 2:
		#ISRF = lambda nu: model_data["rf_intensity"]*SunRF(nu)
	#else:
		#raise ValueError("Only MMP83 (1) and Solar (2) radiation fields have been implemented up to now.")

	# Main Body
	#probabilities = dust_charge_distribution(gas,model_data["rad"],model_data["Z0"],
	#		model_data["material"],model_data["Zmin"],model_data["Zmax"],
	#		f_lin,f_spline,Qabs_fun,ISRF)
	#Z_values = np.arange(model_data["Zmin"],model_data["Zmax"]+1)
	#df = pd.DataFrame.from_dict({"Z":Z_values,"prob":probabilities})
	#df.to_csv('DustCharge_Distribution.txt',sep="\t",index=False)
	#fig = plt.figure()
	#plt.plot(Z_values,probabilities,lw=2,color='b')
	#plt.xlabel('Z')
	#plt.ylabel('f(Z)')
	#plt.title("Dust grain "+str(model_data["rad"])+" microns")
	#plt.tight_layout()
	#fig.savefig("Probabilities.eps")
	#plt.close(fig)
	#print("Chimpún")


if __name__ == "__main__":
	main()
