import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def continuum(data, ties):
    cont = pd.Series(index=data.index, dtype='float64')
    # line through left and middle tie points
    slope = (data[ties[1]] - data[ties[0]]) / (ties[1] - ties[0])
    cont[:ties[1]] = data[ties[0]] + slope * (cont[:ties[1]].index - ties[0])
    # line through middle and right tie points
    slope = (data[ties[2]] - data[ties[1]]) / (ties[2] - ties[1])
    cont[ties[1]:] = data[ties[1]] + slope * (cont[ties[1]:].index - ties[1])
    return cont


def smooth(band):
    return band.rolling(window=3, center=True).mean()


def polynomial_approximation(band, deg=4):
    return np.poly1d(np.polyfit(band.index, band, deg))


def center(band):
    left, right = band.index[0], band.index[-1]
    poly = polynomial_approximation(band)

    crit_x = poly.deriv().roots
    crit_x = np.real_if_close(crit_x[np.isreal(crit_x)])
    crit_x = crit_x[left <= crit_x]
    crit_x = crit_x[crit_x <= right]
    if len(crit_x) == 0:
        return None

    crit_y = np.vectorize(poly)(crit_x)
    index = np.argmin(crit_y)
    x, y = crit_x[index], crit_y[index]
    is_min = poly.deriv().deriv()(x) > 0
    if not is_min:
        return None

    return x, y


def integrated_depth(band):
    return np.trapz(1 - band, band.index)


def asymmetry(band, ctr_x):
    ctr_left = band[:ctr_x].index[-1]
    ctr_right = band[ctr_x:].index[0]
    ctr_y = (band[ctr_left] + (ctr_x - ctr_left) *
             (band[ctr_right] - band[ctr_left]) / (ctr_right - ctr_left))
    band = pd.concat([
        band[:ctr_left],
        pd.Series([ctr_y], index=[ctr_x]),
        band[ctr_right:]
    ])
    left = band[:ctr_x]
    asym = integrated_depth(left) / integrated_depth(band)

    return asym


def transform_pixel(pin, wavelengths, ties, glass, depth_wavelengths,
                    dynamic_tie_ranges=None, center_range=200):
    """There are 20 output parameters for each pixel.
    1-3. The 3 tie points.
    4-6. reflectance at 3 tie points.
    7. 1um band minimum. The wavelength with the minimum (continuum-removed) 
       reflectance.
    8. 1um band center. The wavelength of the minimum of the 4th-degree
       polynomial fit to the (continuum-removed) band within center_range of
       the band minimum.
    9. 1um band depth. 1 - y, where y is the (continuum-removed) 
       reflectance at the band center.
    10. 1um band integrated band depth. The integral of 1 - y over the band.
    11. 1um band asymmetry. The "area of a band" is the area between the band
        and the line y=1 in the continuum-removed reflectance graph; i.e.
        integral across the band of (1 - y) where y is (continuum-removed)
        reflectance.
        The band asymmetry is defined as the portion of the band that's left of
        the band center.
    12-16. Same as 7-11, but for the 2um band.
    17. Interband distance. (2um band center) - (1um band center).
    18. Glass band depth.
    19-. Band depth at the specified wavelengths.
    """

    num_parameters = 18 + len(depth_wavelengths or [])

    data = pd.Series(data=pin, index=wavelengths)
    pout = np.full(num_parameters, np.nan, dtype=pin.dtype)

    # Boxcar smoothing
    data = smooth(data)

    if dynamic_tie_ranges:
        initial_cont = continuum(data, ties)
        initial_removed = data / initial_cont
        ties = [
            initial_removed[start:end].idxmax()
            for start, end in dynamic_tie_ranges
        ]

    # 3 tie point wavelengths
    pout[0:3] = ties

    # reflectance values at the 3 tie points
    pout[3:6] = data[ties]

    # continuum removal
    cont = continuum(data, ties)
    removed = data / cont

    # 1um band
    left1, right1 = ties[0], ties[1]
    band1 = removed[left1:right1]
    minimum1 = band1.idxmin()
    ctr1 = center(band1[minimum1 - center_range:minimum1 + center_range])
    if ctr1 is not None:
        x1, y1 = ctr1
        depth1 = 1 - y1
        ibd1 = integrated_depth(band1)
        asym1 = asymmetry(band1, x1)
        pout[6:11] = [minimum1, x1, depth1, ibd1, asym1]

    # 2um band
    left2, right2 = ties[1], ties[2]
    band2 = removed[left2:right2]
    minimum2 = band2.idxmin()
    ctr2 = center(band2[minimum2 - center_range:minimum2 + center_range])
    if ctr2 is not None:
        x2, y2 = ctr2
        depth2 = 1 - y2
        ibd2 = integrated_depth(band2)
        asym2 = asymmetry(band2, x2)
        pout[11:16] = [minimum2, x2, depth2, ibd2, asym2]

    # interband distance
    if ctr1 is not None and ctr2 is not None:
        pout[16] = x2 - x1

    # glass band depth
    pout[17] = 1 - removed[glass].mean()

    # other band depths
    pout[18:] = 1 - removed[depth_wavelengths]

    return pout


def output_band_names(depth_wavelengths):
    return [
        '1st tie point',
        '2nd tie point',
        '3rd tie point',
        'reflectance at 1st tie point',
        'reflectance at 2nd tie point',
        'reflectance at 3rd tie point',
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
    ] + [
        f'band depth at {x}nm'
        for x in depth_wavelengths
    ]
