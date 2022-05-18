import numpy as np

# Global variables
e_1 = 4.8032e-10 # elementary charge
e_2 = 23.0708e-20 # squared elementary charge, in esu
h_planck = 6.6260755e-27 # Planck constant, erg s
erg_eV = 6.2415e+11 # conversion, 1erg = 6.2415e+11 eV
#

# # # # # # # # # # #
#    Dust Grain     #
# # # # # # # # # # #
class DustGrain:
	"""
	Class that defines an interstellar dust grain. It will be determined by the radius, charge (Z) and
	composition (material). Given them, we can also set the Work Function, ionization potential, electron 
	affinity, minimum energy to produce a spontaneous detachment of an electron, and the photoelectric threshold frequency.
	Electron Affinity, Ionization Potential, and Emin are taken from Weingartner & Draine (2001)
	All units are given in CGS.
	Attributes:
		rad : radius (cm)
		Nc  : number of Carbon atoms (Eq. 1 of Weingartner et al. (2001)
		Z   : charge (dimensionless)
		material: either silicate or carbonnaceous
		W   : work function (erg)
		EA  : electron affinity (erg)
		IPv : ionization potential of the valence band (erg)
		Emin: minimum energy required for tunneling (erg)
		freq_phot: minimum frequency required to produce a photoelectron (Hz)
	"""
	def __init__(self,rad,Z,material, soliddens):
		"""
		rad: float value, in microns. Has to be greater than 0.03 microns
		Z:   integer. Grain charge.
		material: string. Grain's composition. Must be silicate or carbonaceous.
		"""
		global e_2, h_planck,erg_eV
		if rad >= 0.001:#0.03: # I relax this condition
			self.rad = rad*1e-4 # We work in cgs
			self.Nc = 468*np.power(self.rad/1e-7,3)
		else:
			raise ValueError("Only grains with radius greater than 0.03 microns (Classical grains) are allowed")
		try:
			self.Z = int(Z)
		except ValueError:
			raise AttributeError("Dust Charge must be integer")
		self.material = material
		if self.material == "silicate":
			self.W = 8*1.6021772e-12 # erg
			self.EA = 3*1.6021772e-12+(self.Z-0.5)*e_2/self.rad # erg
		elif self.material == "carbonaceous":
			self.W = 4.4*1.6021772e-12 # erg
			self.EA = self.W + (self.Z-0.5)*e_2/self.rad - e_2*4e-8/(self.rad*(self.rad+7e-8)) # erg
		else:
			raise AttributeError("This material is not implemented")
		self.IPv = self.W + (self.Z + 0.5)*e_2/self.rad + (self.Z+2)*e_2*0.3e-8/(self.rad**2)
		# Minimum energy required for tunneling Emin

		if self.Z >= -1:
			self.Emin = 0
		else:
			Emin = -(Z+1)*(e_2/rad)/(np.power(1+27e-8/rad,0.75))  # eV
			self.Emin = Emin/erg_eV

		# Minimum frequency for photoelectric effect freq_phot		
		self.freq_phot = (self.IPv + self.Emin)/h_planck # Hz
		
		# Internal solid density
		self.solid_density = soliddens


# # # # # # # # # # #
#        Gas        #
# # # # # # # # # # #
class Gas:
	"""
	Class that defines the gas. It has to account for the composition of the gas (mainly hydrogen), as well as its density (cm-3), temperature, and ionization fraction.
	Attributes:
		comp: composition
		T   : temperature (K)
		dens: density (cm-3)
		ioniz_frac: ionization fraction
		max_freq: maximum frequency that can propagate through the gas (Hz)
		dneutral: density of neutral species (cm-3)
		dion   : ion density (cm-3)
		delec  : electron density (cm-3)
		ion_mass: mass of the ionic species (g)
		electron_mass: mass of an electron (g)
	"""
	def __init__(self,max_energy, n_e, T_e, n_ion_list, m_ion_list, T_ion_list):
		self.max_energy = max_energy
		self.delec = n_e
		self.Telec = T_e
		self.dion = n_ion_list
		self.ion_mass = m_ion_list
		self.Tion = T_ion_list
		self.elec_mass = 9.1094e-28 # mass of an electron, g
		
# -------------------------------------------------------


def main():
	gr1 = DustGrain(0.5,0,"silicate")
	gr2 = DustGrain(0.5,-2,"silicate")
	gas = Gas("H",6000,0.5,0.1)
	print("Neutral grain",gr1.__dict__,"\n")
	print("Grain with Z = -2",gr2.__dict__,"\n")
	print("Gas properties:\n",gas.__dict__,"\n")
if __name__ == "__main__":
	main()
