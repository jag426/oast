import numpy as np
import pandas as pd


def continuum(data, ties):
    cont = pd.Series(index=data.index)
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


def transform_pixel(pin, wavelengths, ties, glass, depth_wavelengths):
    """There are 20 output parameters for each pixel.
    1-3. reflectance at 3 tie points.
    4. 1um band minimum. The wavelength with the minimum (continuum-removed) 
       reflectance.
    5. 1um band center. The wavelength of the minimum of the 4th-degree
       polynomial fit to the (continuum-removed) band.
    6. 1um band depth. 1 - y, where y is the (continuum-removed) 
       reflectance at the band center.
    7. 1um band integrated band depth. The integral of 1 - y over the band.
    8. 1um band asymmetry. The "area of a band" is the area between the band and
       the line y=1 in the continuum-removed reflectance graph; i.e. integral 
       across the band of (1 - y) where y is (continuum-removed) reflectance.
       The band asymmetry is defined as the portion of the band that's left of
       the band center.
    9-13. Same as 4-8, but for the 2um band.
    14. Interband distance. (2um band center) - (1um band center).
    15. Glass band depth.
    16-20. Band depth at the 5 specified wavelengths.
    """

    data = pd.Series(data=pin, index=wavelengths)
    pout = np.full(20, np.nan, dtype=pin.dtype)

    # reflectance values at the 3 tie points
    pout[:3] = data[ties]

    # continuum removal
    cont = continuum(data, ties)
    removed = data / cont

    # 1um band
    left, right = ties[0], ties[1]
    band = removed[left:right]
    minimum = band.idxmin()
    ctr = center(band)
    if ctr is not None:
        x, y = ctr
        depth = 1 - y
        ibd = integrated_depth(band)
        asym = asymmetry(band, x)
        pout[3:8] = [minimum, x, depth, ibd, asym]

    # 2um band
    left, right = ties[1], ties[2]
    band = removed[left:right]
    minimum = band.idxmin()
    ctr = center(band)
    if ctr is not None:
        x, y = ctr
        depth = 1 - y
        ibd = integrated_depth(band)
        asym = asymmetry(band, x)
        pout[8:13] = [minimum, x, depth, ibd, asym]

    # interband distance
    pout[13] = pout[9] - pout[4]

    # glass band depth
    pout[14] = 1 - removed[glass].mean()

    # other band depths
    pout[15:20] = 1 - removed[depth_wavelengths]

    return pout
