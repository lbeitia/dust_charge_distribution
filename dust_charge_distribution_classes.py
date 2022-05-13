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
	def __init__(self,rad,Z,material):
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
		if isinstance(Z,int):
			self.Z = Z
		else:
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
	def __init__(self,comp,T,dens,ioniz_frac):
		if T < 0 or dens < 0 or ioniz_frac < 0:
			raise ValueError("Check your parameters. T/dens/ioniz_frac cannot be negative!")
		else:
			if comp == "H":
				self.comp = comp
				self.T = T
				self.dens = dens
				self.ioniz_frac = ioniz_frac
				self.dion = self.dens*self.ioniz_frac
				self.dneutral = self.dens-self.dion
				self.delec = self.dion # Atomic hydrogen only has one electron
				self.ion_mass = 1.6726e-24 # mass of HII, g
				self.elec_mass = 9.1094e-28 # mass of an electron, g
				self.max_energy = 13.6 # Maximum energy of an electron that can propagate inside the gas, eV
				self.Tion = T
				self.Telec = T
			elif comp == "H2+HCO+":
				self.comp = comp
				self.T = T
				self.dens = dens 
				self.ioniz_frac = ioniz_frac
				self.dion = self.dens*self.ioniz_frac # cm3
				self.dneutral = self.dens-self.dion
				self.delec = self.dion # as many electrons as HCO+ ions
				self.ion_mass = 29*1.67e-24 # mass of HCO+, g
				self.elec_mass = 9.1094e-28 # mass of an electron, g
				self.max_energy = 13.6 # Maximum energy of an electron that can propagate inside the gas, eV
				self.Tion = T
				self.Telec = T
			elif comp == "O+O+_northpole":
				self.comp = comp
				self.T = T # not used
				self.dens = dens # not used
				self.ioniz_frac = ioniz_frac # not used
				self.dion = 3.452e4 # cm-3, IRI2016 model
				self.dneutral = 1.545e7 # cm-3, NRLMSISE-00 model
				self.delec = 3.63e4 # cm-3, IRI2016 model
				self.ion_mass= 15.999*1.67e-24 # oxygen mass, g
				self.elec_mass = 9.109e-28 # electron mass, g
				self.max_energy = 10.375 # imposed by the spectrum
				self.Tion = 1378.6 # K
				self.Telec = 2562.6 # K
			elif comp == "O+O+_southpole":
				self.comp = comp
				self.T = T # not used
				self.dens = dens # not used
				self.ioniz_frac = ioniz_frac # not used
				self.dion = 8.928e4 # cm-3, IRI2016 model
				self.dneutral = 2.768e7 # cm-3, NRLMSISE-00 model
				self.delec = 9.29e4 # cm-3, IRI2016 model
				self.ion_mass= 15.999*1.67e-24 # oxygen mass, g
				self.elec_mass = 9.109e-28 # electron mass, g
				self.max_energy = 10.375 # eV, imposed by the spectrum
				self.Tion = 1403.1 # K
				self.Telec = 2150.3 # K
			else:
				raise Warning("Gas composition not available")
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
