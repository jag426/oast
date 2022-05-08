import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import transformations


# Used for initial continuum when computing dynamic tie points
INITIAL_TIE_POINTS = [700, 1489, 2616]

# Ranges within which to look for the dynamic tie points
TIE_POINT_RANGES = [(600, 1000), (1000, 1700), (2000, 2600)]

# Static tie points
DEFAULT_TIE_POINTS = [750, 1489, 2896]

# Range around the band minimum to use for a polynomial fit to compute the
# band center. This is the one-sided range, so the polynomial will be fit from
# (minimum - range) to (minimum + range).
BAND_CENTER_RANGE = 200

GLASS_BAND_POINTS = [1150, 1170, 1190]
BAND_DEPTH_POINTS = [950, 1050, 1249, 1898, 2417]


def nearest_wavelength(x, wavelengths):
    return wavelengths[np.abs(wavelengths - x).argmin()]


def main():
    parser = argparse.ArgumentParser(description='Analyze a single spectrum')
    parser.add_argument('--printout', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--polynomials', action='store_true')
    parser.add_argument('--dynamic_tie_points', '-d', action='store_true')
    parser.add_argument('csvfile')
    args = parser.parse_args()

    csvfile = args.csvfile
    plot = not args.noplot
    polynomials = args.polynomials
    printout = args.printout
    dynamic = args.dynamic_tie_points

    band_info_file = csvfile[:-4] + '_parameter.csv'
    continuum_removed_file = csvfile[:-4] + '_continuum-removed.csv'
    if os.path.exists(band_info_file):
        print(f'{band_info_file} already exists. Quitting...')
        return
    if os.path.exists(continuum_removed_file):
        print(f'{continuum_removed_file} already exists. Quitting...')
        return

    data = pd.read_csv(csvfile, header=0,
                       skiprows=range(1, 4), index_col=1, dtype='f8')
    glass = [nearest_wavelength(x, data.index) for x in GLASS_BAND_POINTS]
    continuum_removed = pd.DataFrame(index=data.index)
    band_info_rows = []
    for i, (colname, spectrum) in enumerate(list(data.items())[1:]):
        if plot:
            plt.figure(colname)
            plt.title(colname)

        spectrum = transformations.smooth(spectrum)

        if dynamic:
            ties = [
                nearest_wavelength(x, data.index)
                for x in INITIAL_TIE_POINTS
            ]
            initial_cont = transformations.continuum(spectrum, ties)
            initial_removed = spectrum / initial_cont
            ties = [
                initial_removed[start:end].idxmax()
                for start, end in TIE_POINT_RANGES
            ]
        else:
            ties = [
                nearest_wavelength(x, data.index)
                for x in DEFAULT_TIE_POINTS
            ]

        # continuum removal
        continuum = transformations.continuum(spectrum, ties)
        removed = spectrum / continuum
        continuum_removed[colname] = removed
        if plot:
            plt.plot(removed)

        # 1um band
        left, right = ties[0], ties[1]
        band_1 = removed[left:right]
        minimum_1 = band_1.idxmin()
        poly_band_1 = band_1[minimum_1 - BAND_CENTER_RANGE:
                             minimum_1 + BAND_CENTER_RANGE]
        ctr_1 = transformations.center(poly_band_1)
        if ctr_1 is not None:
            x_1, y_1 = ctr_1
            depth_1 = 1 - y_1
            ibd_1 = transformations.integrated_depth(band_1)
            asym_1 = transformations.asymmetry(band_1, x_1)
        else:
            x_1 = np.nan
            y_1 = np.nan
            depth_1 = np.nan
            ibd_1 = np.nan
            asym_1 = np.nan

        # 2um band
        left, right = ties[1], ties[2]
        band_2 = removed[left:right]
        minimum_2 = band_2.idxmin()
        poly_band_2 = band_2[minimum_2 - BAND_CENTER_RANGE:
                             minimum_2 + BAND_CENTER_RANGE]
        ctr_2 = transformations.center(poly_band_2)
        if ctr_2 is not None:
            x_2, y_2 = ctr_2
            depth_2 = 1 - y_2
            ibd_2 = transformations.integrated_depth(band_2)
            asym_2 = transformations.asymmetry(band_2, x_2)
        else:
            x_2 = np.nan
            y_2 = np.nan
            depth_2 = np.nan
            ibd_2 = np.nan
            asym_2 = np.nan

        # interband distance, glass
        interband_distance = x_2 - x_1
        glass_depth = 1 - removed[glass].mean()

        band_info_rows.append(pd.DataFrame([{
            'name': colname,
            '1st tie point': ties[0],
            '2nd tie point': ties[1],
            '3rd tie point': ties[2],
            'reflectance at 1st tie point': spectrum[ties[0]],
            'reflectance at 2nd tie point': spectrum[ties[1]],
            'reflectance at 3rd tie point': spectrum[ties[2]],
            '1um band minimum': minimum_1,
            '1um band center': x_1,
            '1um band depth': depth_1,
            '1um integrated band depth': ibd_1,
            '1um band asymmetry': asym_1,
            '2um band minimum': minimum_2,
            '2um band center': x_2,
            '2um band depth': depth_2,
            '2um band integrated band depth': ibd_2,
            '2um band asymmetry': asym_2,
            'interband distance': interband_distance,
            'glass band depth': glass_depth,
        }]))
        if plot and polynomials:
            poly_1 = transformations.polynomial_approximation(poly_band_1)
            plt.plot(poly_band_1.index, np.vectorize(
                poly_1)(poly_band_1.index))
            plt.plot([x_1], [y_1], marker='x', color='grey')
            poly_2 = transformations.polynomial_approximation(poly_band_2)
            plt.plot(poly_band_2.index, np.vectorize(
                poly_2)(poly_band_2.index))
            plt.plot([x_2], [y_2], marker='x', color='grey')

    band_info = pd.concat(band_info_rows, ignore_index=True)
    if printout:
        print(band_info.to_csv(sep='\t'))
        print(continuum_removed.to_csv(sep='\t'))
    band_info.to_csv(band_info_file)
    continuum_removed.to_csv(continuum_removed_file)
    if plot:
        plt.show()


if __name__ == '__main__':
    main()
