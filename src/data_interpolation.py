# This file contains the functions used for interpolation of the
# refractive indices and absorption coefficients
from scipy.interpolate import interp1d
import pandas as pd


def interpolate_refractive_indices(material):
	"""
	This function interpolates the refractive indices
	of the desired material.
	After some tests, I found that a spline interpolation
	behaves well for most part of the refractive
	indices for astrosilicate and graphite, but at some
	points a linear interpolation is needed.
	Input:
		material: string.
	Output:
		flin: interp1d object with linear interpolation.
		fspline: interp1d object with spline interpolation.
	"""
	if material == "silicate":
		data = pd.read_csv('src/refractive_indices/silicate_refractive_indices.csv',comment="#",sep="\t")
		uv_cols = data['wave(um)']< 1 # Wavelengths lower than 1 micron
		x = data['wave(um)'][uv_cols]
		y = data['Im(n)'][uv_cols]
		f_lin = interp1d(x,y,kind="linear",fill_value="extrapolate")
		f_spline = interp1d(x,y,kind="cubic",fill_value = "extrapolate")
	elif material == "graphite":
		# Parallel
		data_para = pd.read_csv('src/refractive_indices/graphite_refractive_indices_para.csv',comment="#",sep="\t")
		uv_cols = data_para['wave(um)']< 1 # Wavelengths lower than 1 micron
		x_para = data_para['wave(um)'][uv_cols]
		y_para = data_para['Im(n)'][uv_cols]
		# Perpendicular
		data_perp = pd.read_csv('src/refractive_indices/graphite_refractive_indices_perp.csv',comment="#",sep="\t")
		uv_cols = data_perp['wave(um)']< 1 # Wavelengths lower than 1 micron
		x_perp = data_perp['wave(um)'][uv_cols]
		y_perp = data_perp['Im(n)'][uv_cols]
		# 1/3 - 2/3 approximation -> list
		f_lin = [interp1d(x_para,y_para,kind="linear",fill_value="extrapolate"),interp1d(x_perp,y_perp,kind="linear",fill_value="extrapolate")]
		f_spline = [interp1d(x_para,y_para,kind="cubic",fill_value="extrapolate"),interp1d(x_perp,y_perp,kind="cubic",fill_value="extrapolate")]
	else:
		raise ValueError("Currently only silicate and graphite are available")
	return([f_lin, f_spline])


def Qabs(material,rad):
	"""
	Absorption coefficient of a dust grain at a given frequency.
	Absorption properties vary with material and wavelength, and it is necessary to have
	precise measures.
	Here I use  the tables provided by Draine in his webpage to compute an function that will return the refractive index.
	Input:
		material: grain material
		radius: dust grain radius, in microns
	Output:
		Qabs_fun: absorption coefficient function, dependent on the wavelength in microns
	"""
	if material == "silicate":
		data = pd.read_csv("src/Qabs_tables/Qabs_Sil_81")
	else: # material == "carbonaceous"
		data = pd.read_csv("src/Qabs_tables/Qabs_Gra_81")

	if rad >= 1e-3 and rad <=10: # Values taken from Qabs_Sil_81 and Qabs_Gra_81
		kw = list(data.columns.values)
		i = 0
		available_rads = [] # Los índices están corridos una posición a la izda respecto a los de data.
		while i < len(kw):
			if "=" in kw[i]:
				line = kw[i].split("=")
				available_rads.append(float(line[1]))
			i+=1
		ldo = pd.to_numeric(data['wav_microns'])
		if rad in available_rads:
			ind = available_rads.index(rad)
			Qabs = data[kw[ind+1]]
		else:
			first_elem = next(x[0] for x in enumerate(available_rads) if x[1]>rad)
			rad_low = available_rads[first_elem-1]
			Q_low = data[kw[first_elem]]
			rad_up = available_rads[first_elem]
			Q_up = data[kw[first_elem+1]]
			# Construyo Qabs mediante interpolación lineal de los dos valores superior e inferior
			slp = (Q_up-Q_low)/(rad_up-rad_low)
			Qabs = (rad-rad_low)*slp + Q_low
	else:
		raise ValueError("Absorption coefficient Qabs cannot be computed at that radius")
	
	return interp1d(ldo,Qabs,kind='linear',fill_value="extrapolate")


