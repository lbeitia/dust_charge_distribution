import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'src/')
from data_interpolation import *
from dust_charge_distribution_classes import DustGrain, Gas
from accretion_rates import *
from stellar_spectra import *
from photoelectric_emission_stellar import *
from photoelectric_emission_cosmic_rays import *
from accretion_rates_cosmic_rays import *


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
			if "#" not in line and "=" in line:
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
def dust_charge_distribution(Gas, model_data,
							 f_lin, f_spline, Qabs_fun, ISRF):
	"""
	Program that computes the charge distribution of a population of dust 
	grains in the interval	[Zmin,Zmax].
	It applies recursively equation 21 from Weingartner & Draine 2001.
	Input:
	        ISRF: interstellar radiation field
	Output:
		probabilities: array containing the probabilities for each value Z,
					ordered with increasing Z.
	"""
	Z0 = int(model_data["Z0"])
	Zmin = int(model_data["Zmin"])
	Zmax = int(model_data["Zmax"])
	F_UV = get_F_UV_CR(model_data)
	
	prob_neg = np.array([1])
	prob_pos = np.array([1])
	Z = Z0 + 1
	J_pe_CR = 0.0
	while Z <= Zmax:
		print("Computing the terms for Z =",Z)
		# Dust grains involved
		dust_grain_Zm1 = DustGrain(model_data["rad"],Z-1,model_data["material"], model_data["solid_density"])
		dust_grain_Z = DustGrain(model_data["rad"],Z,model_data["material"], model_data["solid_density"])
		# 
		J_pe = Jpe_val(dust_grain_Zm1, Gas, f_lin, f_spline, Qabs_fun,ISRF)
		J_pe += Jpe_cond(dust_grain_Zm1,Gas,ISRF)
		J_ion = J_accretion(dust_grain_Zm1,Gas,1)
		J_electron = J_accretion(dust_grain_Z,Gas,-1)
		if model_data["include_CR"] == 1.0:
			J_pe_CR = Jpe_CR(dust_grain_Zm1, model_data, f_lin, f_spline, Qabs_fun, F_UV)
			Je_CR = J_accretion_CRs_elec(dust_grain_Z, model_data)
		# Add probability
		prob_pos = np.append(prob_pos, 
					   (J_pe + J_ion + J_pe_CR)*prob_pos[Z-Z0-1]/J_electron)
		Z +=1
	Z = Z0 - 1
	while Z >= Zmin:
		print("Computing the terms for Z =",Z)
		# Dust grains involved
		dust_grain_Zp1 = DustGrain(model_data["rad"],Z+1,model_data["material"], model_data["solid_density"])
		dust_grain_Z = DustGrain(model_data["rad"],Z,model_data["material"], model_data["solid_density"])
		#
		J_pe = Jpe_val(dust_grain_Z,Gas,f_lin,f_spline,Qabs_fun,ISRF)
		J_pe += Jpe_cond(dust_grain_Z,Gas,ISRF)
		if model_data["include_CR"] == 1.0:
			J_pe_CR = Jpe_CR(dust_grain_Z, model_data, f_lin, f_spline, Qabs_fun, F_UV)
			Je_CR = J_accretion_CRs_elec(dust_grain_Zp1, model_data)
		J_ion = J_accretion(dust_grain_Z,Gas,1)
		J_electron = J_accretion(dust_grain_Zp1,Gas,-1)
		# Add probability
		prob_neg = np.append(prob_neg, 
					   prob_neg[Z0-Z-1]*J_electron/(J_pe + J_ion + J_pe_CR))
		Z-=1
	dust_distrib = np.delete(prob_neg,0)
	dust_distrib = dust_distrib[::-1]
	dust_distrib = np.append(dust_distrib,prob_pos)
	dust_distrib = dust_distrib/sum(dust_distrib)
	return dust_distrib


def compute_centroid_and_write_file(model_data, probabilities):
	"""
	This function finds the centroid of the distribution and writes it
	in a file together with the full charge distribution,
	Input: 
		model_data
		probabilities
	Output:
		DustCharge_Distribution.txt
	"""
	Z_values = np.arange(model_data["Zmin"],model_data["Zmax"]+1)
	df = pd.DataFrame.from_dict({"Z":Z_values,"prob":probabilities})
	centroid = np.round(np.sum(Z_values * probabilities), 2)
	dispersion2 = np.sum(probabilities * np.power(Z_values - centroid,2))
	dispersion = np.round(np.sqrt(dispersion2), 2)
	
	with open(model_data['out_fname'], 'w') as f:
		f.write('# <Z> = ' + str(centroid) + '\n')
		f.write('# sigma_Z = ' + str(dispersion) + '\n')

	df.to_csv(model_data['out_fname'],sep="\t",index=False, mode = 'a')
	
	if model_data['plot_distribution'] == 1.0:
		fig = plt.figure()
		plt.plot(Z_values,probabilities,lw=2,color='b')
		plt.xlabel('Z')
		plt.ylabel('f(Z)')
		plt.title("Dust grain "+str(model_data["rad"])+" microns")
		plt.tight_layout()
		fig.savefig("Probabilities.eps")
		plt.close(fig)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def main():
	"""
	Main function that computes the charge probability distribution of a
	given population of dust grains.
	Output: 
		dust_distrib.txt: file containing the values of Z with the 
							corresponding probability
	"""
	# read input data
	model_data = read_input_file()
	# Interpolate refractive indices for later usage
	[f_lin, f_spline] = interpolate_refractive_indices(model_data["material"])
	# Absorption coefficient function must be defined before the main loop
	# because in that way, it will be computed only once, being more efficient.
	Qabs_fun = Qabs(model_data['material'],model_data['rad'])
	# GAS
	gas = Gas(model_data["composition"], model_data["T"], 
			model_data["dens"], model_data["frac_ion"])
	# Radiation Field
	radfield = determine_radiation_field(model_data["rf"],
									model_data["rf_intensity"])
	# Main Body
	probabilities = dust_charge_distribution(gas,model_data,
			f_lin,f_spline,Qabs_fun,radfield)
	
	# Post processing of results
	compute_centroid_and_write_file(model_data, probabilities)



if __name__ == "__main__":
	main()
