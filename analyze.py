import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import transformations


def nearest_wavelength(x, wavelengths):
    return wavelengths[np.abs(wavelengths - x).argmin()]


def main():
    ties = [750, 1489, 2896]
    glass = [1150, 1170, 1190]
    deg = 4
    smooth = True
    plot = True
    printout = True

    parser = argparse.ArgumentParser(description='Analyze a single spectrum')
    parser.add_argument('csvfile')
    args = parser.parse_args()
    csvfile = args.csvfile
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
    ties = [nearest_wavelength(x, data.index) for x in ties]
    glass = [nearest_wavelength(x, data.index) for x in glass]
    band_info = pd.DataFrame(columns=[
        'name',
        'reflectance at 750nm',
        'reflectance at 1489nm',
        'reflectance at 2896nm',
        '1um band minimum',
        '1um band center',
        '1um band depth',
        '1um integrated band depth',
        '1um band asymmetry',
        '2um band minimum',
        '2um band center',
        '2um band depth',
        '2um band integrated band depth',
        '2um band asymmetry',
        'interband distance',
        'glass band depth',
    ])
    continuum_removed = pd.DataFrame(index=data.index)
    for i, (colname, spectrum) in enumerate(list(data.items())[1:]):
        if plot:
            plt.figure(colname)
            plt.title(colname)
        if smooth:
            spectrum = transformations.smooth(spectrum)

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
        ctr_1 = transformations.center(band_1)
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
        ctr_2 = transformations.center(band_2)
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

        band_info = band_info.append({
            'name': colname,
            'reflectance at 750nm': spectrum[ties[0]],
            'reflectance at 1489nm': spectrum[ties[1]],
            'reflectance at 2896nm': spectrum[ties[2]],
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
        }, ignore_index=True)
        if plot:
            poly_1 = transformations.polynomial_approximation(band_1, deg)
            plt.plot(band_1.index, np.vectorize(poly_1)(band_1.index))
            plt.plot([x_1], [y_1], marker='x', color='grey')
            poly_2 = transformations.polynomial_approximation(band_2, deg)
            plt.plot(band_2.index, np.vectorize(poly_2)(band_2.index))
            plt.plot([x_2], [y_2], marker='x', color='grey')

    if printout:
        print(band_info.to_csv(sep='\t'))
        print(continuum_removed.to_csv(sep='\t'))
    band_info.to_csv(band_info_file)
    continuum_removed.to_csv(continuum_removed_file)
    if plot:
        plt.show()


if __name__ == '__main__':
    main()
