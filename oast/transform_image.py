import argparse
import os

import joblib
import numpy as np
import pandas as pd
import spectral.io.envi as envi

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

DATA_IGNORE_VALUE_DEFAULT = -999.0
DATA_IGNORE_VALUE_NAME = 'data ignore value'


def metadata(old_md, hdrfile, depth_wavelengths):
    md = dict()
    md['description'] = 'Spectral parameters computed from ' + hdrfile
    copy_keys = [
        'coordinate system string',
        'map info',
        'y start',
    ]
    for key in copy_keys:
        if key in old_md:
            md[key] = old_md[key]
    md['band names'] = transformations.output_band_names(depth_wavelengths)
    if DATA_IGNORE_VALUE_NAME in old_md:
        md[DATA_IGNORE_VALUE_NAME] = old_md[DATA_IGNORE_VALUE_NAME]
    else:
        md[DATA_IGNORE_VALUE_NAME] = DATA_IGNORE_VALUE_DEFAULT
    return md


def nearest_wavelength(x, wavelengths):
    return wavelengths[np.abs(wavelengths - x).argmin()]


def transform_image(img, wavelengths, ties, glass, depth_wavelengths,
                    dynamic_tie_ranges, center_range, ignore_value):
    ties = [nearest_wavelength(x, wavelengths) for x in ties]
    glass = [nearest_wavelength(x, wavelengths) for x in glass]
    depth_wavelengths = [nearest_wavelength(
        x, wavelengths) for x in depth_wavelengths]
    if ignore_value is None:
        # treat negative reflectances as ignored
        print('ignoring negative values.')
        img[img <= 0] = np.nan
    else:
        img[img == ignore_value] = np.nan
    transformv = np.vectorize(transformations.transform_pixel,
                              excluded={'wavelengths',
                                        'ties',
                                        'glass',
                                        'depth_wavelengths',
                                        'dynamic_tie_ranges',
                                        'center_range'},
                              signature='(n)->(k)')
    out = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(transformv)(img[i],
                                   wavelengths=wavelengths,
                                   ties=ties,
                                   glass=glass,
                                   depth_wavelengths=depth_wavelengths,
                                   dynamic_tie_ranges=dynamic_tie_ranges,
                                   center_range=center_range)
        for i in range(img.shape[0]))
    out = np.squeeze(np.array(out))
    if ignore_value is None:
        print(
            f'writing output with {DATA_IGNORE_VALUE_DEFAULT} as {DATA_IGNORE_VALUE_NAME}')
        out[np.isnan(out)] = DATA_IGNORE_VALUE_DEFAULT
    else:
        print(
            f'writing output with {ignore_value} as {DATA_IGNORE_VALUE_NAME}')
        out[np.isnan(out)] = ignore_value
    return out


def main():
    parser = argparse.ArgumentParser(description='Transform an ENVI image')
    parser.add_argument('hdrfile')
    parser.add_argument('--dynamic_tie_points', '-d', action='store_true')
    args = parser.parse_args()
    hdrfile = args.hdrfile
    out_filename = hdrfile[:-4] + '_parameter.hdr'
    if os.path.exists(out_filename):
        print(f'{out_filename} already exists. Quitting...')
        return
    img = envi.open(hdrfile).load()
    lines, samples, bands = img.shape
    print(f'{lines} lines, {samples} samples, {bands} bands')
    dt = img.dtype
    ties = DEFAULT_TIE_POINTS
    glass = GLASS_BAND_POINTS
    depth_wavelengths = BAND_DEPTH_POINTS
    dynamic_tie_ranges = None
    if args.dynamic_tie_points:
        dynamic_tie_ranges = TIE_POINT_RANGES
    center_range = BAND_CENTER_RANGE
    wavelengths = np.array(img.metadata['wavelength'], dtype=dt)
    if DATA_IGNORE_VALUE_NAME in img.metadata:
        ignore_value = dt.type(img.metadata[DATA_IGNORE_VALUE_NAME])
        print(f'{DATA_IGNORE_VALUE_NAME} = {ignore_value}')
    else:
        ignore_value = None
        print(f'{DATA_IGNORE_VALUE_NAME} not set.')
    md = metadata(img.metadata, hdrfile, depth_wavelengths)
    interleave = img.metadata['interleave']
    out = transform_image(img,
                          wavelengths=wavelengths,
                          ties=ties,
                          glass=glass,
                          depth_wavelengths=depth_wavelengths,
                          dynamic_tie_ranges=dynamic_tie_ranges,
                          center_range=center_range,
                          ignore_value=ignore_value)
    envi.save_image(out_filename, out,
                    metadata=md, interleave=interleave)


if __name__ == '__main__':
    main()
