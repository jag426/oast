import argparse

import joblib
import numpy as np
import pandas as pd
import spectral.io.envi as envi

import transformations


def metadata(old_md, hdrfile):
    md = dict()
    md['description'] = 'Spectral parameters computed from ' + hdrfile
    md['map info'] = old_md['map info']
    md['coordinate system string'] = old_md['coordinate system string']
    md['y start'] = old_md['y start']
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
    md['data ignore value'] = old_md['data ignore value']
    return md


def nearest_wavelength(x, wavelengths):
    return wavelengths[np.abs(wavelengths - x).argmin()]


def transform_image(img, wavelengths, ties, glass, ignore_value):
    ties = [nearest_wavelength(x, wavelengths) for x in ties]
    glass = [nearest_wavelength(x, wavelengths) for x in glass]
    img[img == ignore_value] = np.nan
    transformv = np.vectorize(transformations.transform_pixel,
                              excluded={'wavelengths', 'ties', 'glass'},
                              signature='(n)->(k)')
    out = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(transformv)(
            img[i], wavelengths=wavelengths, ties=ties, glass=glass)
        for i in range(img.shape[0]))
    out = np.squeeze(np.array(out))
    out[np.isnan(out)] = ignore_value
    return out


def main():
    parser = argparse.ArgumentParser(description='Transform an ENVI image')
    parser.add_argument('hdrfile')
    args = parser.parse_args()
    hdrfile = args.hdrfile
    img = envi.open(hdrfile).load()
    lines, samples, bands = img.shape
    print(f'{lines} lines, {samples} samples, {bands} bands')
    dt = img.dtype
    ties = [750, 1489, 2896]
    glass = [1150, 1170, 1190]
    wavelengths = np.array(img.metadata['wavelength'], dtype=dt)
    ignore_value = dt.type(img.metadata['data ignore value'])
    out = transform_image(img, wavelengths, ties, glass, ignore_value)
    md = metadata(img.metadata, hdrfile)
    interleave = img.metadata['interleave']
    envi.save_image('out.hdr', out,
                    metadata=md, interleave=interleave, force=True)


if __name__ == '__main__':
    main()
