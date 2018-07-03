import argparse
import os

import joblib
import numpy as np
import pandas as pd
import spectral.io.envi as envi

import transformations


DATA_IGNORE_VALUE_DEFAULT = -999.0
DATA_IGNORE_VALUE_NAME = 'data ignore value'


def metadata(old_md, hdrfile):
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
    md['band names'] = [
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
    ]
    if DATA_IGNORE_VALUE_NAME in old_md:
        md[DATA_IGNORE_VALUE_NAME] = old_md[DATA_IGNORE_VALUE_NAME]
    else:
        md[DATA_IGNORE_VALUE_NAME] = DATA_IGNORE_VALUE_DEFAULT
    return md


def nearest_wavelength(x, wavelengths):
    return wavelengths[np.abs(wavelengths - x).argmin()]


def transform_image(img, wavelengths, ties, glass, ignore_value):
    ties = [nearest_wavelength(x, wavelengths) for x in ties]
    glass = [nearest_wavelength(x, wavelengths) for x in glass]
    if ignore_value is None:
        # treat negative reflectances as ignored
        print('ignoring negative values.')
        img[img <= 0] = np.nan
    else:
        img[img == ignore_value] = np.nan
    transformv = np.vectorize(transformations.transform_pixel,
                              excluded={'wavelengths', 'ties', 'glass'},
                              signature='(n)->(k)')
    out = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(transformv)(
            img[i], wavelengths=wavelengths, ties=ties, glass=glass)
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
    ties = [750, 1489, 2896]
    glass = [1150, 1170, 1190]
    wavelengths = np.array(img.metadata['wavelength'], dtype=dt)
    if DATA_IGNORE_VALUE_NAME in img.metadata:
        ignore_value = dt.type(img.metadata[DATA_IGNORE_VALUE_NAME])
        print(f'{DATA_IGNORE_VALUE_NAME} = {ignore_value}')
    else:
        ignore_value = None
        print(f'{DATA_IGNORE_VALUE_NAME} not set.')
    md = metadata(img.metadata, hdrfile)
    interleave = img.metadata['interleave']
    out = transform_image(img, wavelengths, ties, glass, ignore_value)
    envi.save_image(out_filename, out,
                    metadata=md, interleave=interleave)


if __name__ == '__main__':
    main()
