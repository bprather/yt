
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt

from unyt import unyt_array

class CoordinateTransform(object):
    """Base class for coordinate transforms.
    Define some basic functions like transformation matrix applications & inverses
    """
    def transform_vector(self, x, vec):
        """Transform a three-vector vec at location x from """
        return np.einsum("ij...,j...->i...", self._dxdX(x), vec)

    def transform_cellshape(self, x, vec):
        """Transform a three-vector vec at location x from """
        return np.einsum("ij...,j...->i...", self._dxdX(x, visual=True), vec)

    # This is currently unused but might be useful for transforming vectors/locations *back*
    def _dXdx(self, x):
        return np.einsum("...ij->ij...", la.inv(np.einsum("ij...->...ij", self.dxdX(x))))
    
    # TODO(BSP) transforms back to native coords?

class SphericalTransform(CoordinateTransform):
    """Base class for spherical coordinate system transformations.
    Includes methods to Cartesianize, used in e.g. plotting 
    """
    # TODO(BSP) don't strip units here, integrate them more formally
    def transform(self, x):
        """Transform a three-component coordinate location x from the code's "native"
        coordinates to spherical coordinates r, th, phi
        """
        if isinstance(x, unyt_array):
            x = x.v
        return np.array((self.r(x), self.th(x), self.phi(x)))

    def r(self, x):
        return x[0]

    def th(self, x):
        return x[1]

    def phi(self, x):
        return x[2]

    def cart_x(self, x):
        return self.r(x)*np.sin(self.th(x))*np.cos(self.phi(x))

    def cart_y(self, x):
        return self.r(x)*np.sin(self.th(x))*np.sin(self.phi(x))

    def cart_z(self, x):
        return self.r(x)*np.cos(self.th(x))

class SphericalExponential(SphericalTransform):
    """Exponentiated radial coordinate in a spherical coordinate system.
    """
    def __init__(self, met_params):
        super(SphericalExponential, self).__init__(met_params)

    def r(self, x):
        return np.exp(x[0])

    def _dxdX(self, x):
        dxdX = np.zeros([3, 3, *x.shape[1:]])
        dxdX[0, 0] = np.exp(x[0])
        dxdX[1, 1] = 1
        dxdX[2, 2] = 1
        return dxdX

class SphericalHyperExponential(SphericalTransform):
    """Hyper-exponential coordinate transformation in radius, see e.g. Tchekhovskoy et al. 2011
    """
    def __init__(self, met_params):
        self.xe1br = met_params['r_br']
        self.xn1br = np.log(self.xe1br)
        self.npow2 = met_params['npow']
        self.cpow2 = met_params['cpow']

    def r(self, x):
        super_dist = np.where(x[0] > self.xn1br, x[0] - self.xn1br, 0.0)
        return np.exp(x[0] + self.cpow2 * np.power(super_dist, self.npow2))

    def _dxdX(self, x):
        super_dist = np.where(x[0] > self.xn1br, x[0] - self.xn1br, 0.0)
        dxdX = np.zeros([3, 3, *x.shape[1:]])
        dxdX[0, 0] = np.exp(x[0] + self.cpow2 * np.power(super_dist, self.npow2)) \
                            * (1 + self.cpow2 * self.npow2 * np.power(super_dist, self.npow2-1))
        dxdX[1, 1] = 1
        dxdX[2, 2] = 1
        return dxdX

class SphericalModified(SphericalExponential):
    """Modified Kerr-Schild coordinates, see e.g. Gammie et al. 2003 "HARM"
    Note hslope is defined as in that work and not elsewhere:
    `hslope = 1` corresponds to even spacing/no transformation.
    """
    def __init__(self, met_params):
        self.hslope = float(met_params['hslope'])

    def th(self, x):
        return np.pi*x[1] + ((1. - self.hslope)/2.)*np.sin(2.*np.pi*x[1])

    def _dxdX(self, x):
        dxdX = np.zeros([3, 3, *x.shape[1:]])
        dxdX[0, 0] = np.exp(x[0])
        dxdX[1, 1] = np.pi - (self.hslope - 1.) * np.pi * np.cos(2. * np.pi * x[1])
        dxdX[2, 2] = 1
        return dxdX


class SphericalFunkyModified(SphericalExponential):
    """Funky Modified Kerr-Schild coordinate transformation, see e.g. Wong et al. 2022 "PATOKA"
    """
    def __init__(self, met_params):
        self.hslope = float(met_params['hslope'])
        self.poly_xt =float(met_params['poly_xt'])
        self.poly_alpha = float(met_params['poly_alpha'])
        self.mks_smooth = float(met_params['mks_smooth'])
        try:
            self.startx1 = float(met_params['fmks_zero_point'])
        except KeyError:
            self.startx1 = 1. + np.sqrt(1. - float(met_params['a'])**2) - 0.1
        self.poly_norm = 0.5 * np.pi * 1. / (1. + 1. / (self.poly_alpha + 1.) *
                                             1. / np.power(self.poly_xt, self.poly_alpha))

    def th(self, x):
        th_g = np.pi * x[1] + ((1. - self.hslope) / 2.) * np.sin(2. * np.pi * x[1])
        y = 2 * x[1] - 1.
        th_j = self.poly_norm * y * (
                    1. + np.power(y / self.poly_xt, self.poly_alpha) / (self.poly_alpha + 1.)) + 0.5 * np.pi
        return th_g + np.exp(self.mks_smooth * (self.startx1 - x[0])) * (th_j - th_g)

    def _dxdX(self, x, visual=False):
        if visual:
            # Reflect across x=0.5 and calculate widths at zone ends, for symmetry
            x[1] = np.where(x[1] < 0.5, 1. - x[1], x[1])
        # Otherwise standard
        dxdX = np.zeros([3, 3, *x.shape[1:]])
        dxdX[0, 0] = np.exp(x[0])
        dxdX[1, 0] = -np.exp(self.mks_smooth * (self.startx1 - x[0])) * self.mks_smooth *\
                     (np.pi / 2. - np.pi * x[1] + self.poly_norm * (2. * x[1] - 1.) *
                      (1 + (np.power((-1. + 2 *x[1]) / self.poly_xt, self.poly_alpha)) /
                       (1 + self.poly_alpha)) -
                      1. / 2. * (1. - self.hslope) * np.sin(2. * np.pi * x[1]))
        dxdX[1, 1] = np.pi + (1. - self.hslope) * np.pi * np.cos(2. * np.pi * x[1]) + \
                     np.exp(self.mks_smooth * (self.startx1 - x[0])) * \
                     (-np.pi + 2. * self.poly_norm * (1. + np.power((2. * x[1] - 1.) / self.poly_xt, self.poly_alpha) /
                                                      (self.poly_alpha + 1.)) +
                                                     (2. * self.poly_alpha * self.poly_norm * (2. * x[1] - 1.) *
                                                      np.power((2. * x[1] - 1.) / self.poly_xt, self.poly_alpha - 1.)) /
                                                     ((1. + self.poly_alpha) * self.poly_xt) -
                                                    (1. - self.hslope) * np.pi * np.cos(2. * np.pi * x[1]))
        dxdX[2, 2] = 1
        return dxdX


