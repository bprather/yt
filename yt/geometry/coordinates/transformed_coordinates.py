from functools import cached_property

import numpy as np

from yt.utilities.lib.pixelization_routines import pixelize_aitoff, pixelize_cylinder

from .coordinate_handler import (
    CoordinateHandler,
    _get_coord_fields,
    _get_polar_bounds,
    _setup_dummy_cartesian_coords_and_widths,
    _setup_polar_coordinates,
)

class TransformedSphericalCoordinateHandler(CoordinateHandler):
    name = "Transformed"
    _default_axis_order = ("r", "theta", "phi")

    def __init__(self, ds, ordering=None):
        super().__init__(ds, ordering)
        # Generate
        self.image_units = {}
        self.image_units[self.axis_id["r"]] = (1, 1)
        self.image_units[self.axis_id["theta"]] = (None, None)
        self.image_units[self.axis_id["phi"]] = (None, None)
        self._transform = ds.transform

    def setup_fields(self, registry):
        # Missing implementation for x, y and z coordinates.
        _setup_dummy_cartesian_coords_and_widths(registry, axes=("x", "y", "z"))
        _setup_polar_coordinates(registry, self.axis_id)

        f1, f2 = _get_coord_fields(self.axis_id["phi"], "dimensionless")
        registry.add_field(
            ("index", "dphi"),
            sampling_type="cell",
            function=f1,
            display_field=False,
            units="dimensionless",
        )

        registry.add_field(
            ("index", "phi"),
            sampling_type="cell",
            function=f2,
            display_field=False,
            units="dimensionless",
        )

        def _DiffGeoVolume(field, data):
            raise NotImplementedError

        registry.add_field(
            ("index", "cell_volume"),
            sampling_type="cell",
            function=_DiffGeoVolume,
            units="code_length**3",
        )
        registry.alias(("index", "volume"), ("index", "cell_volume"))

        def _path_phi(field, data):
            # Note: this already assumes cell-centered
            return (
                data["index", "r"]
                * data["index", "dphi"]
                * np.sin(data["index", "theta"])
            )

        registry.add_field(
            ("index", "path_element_phi"),
            sampling_type="cell",
            function=_path_phi,
            units="code_length",
        )

    def pixelize(
        self,
        dimension,
        data_source,
        field,
        bounds,
        size,
        antialias=True,
        periodic=True,
        *,
        return_mask=False,
    ):  
        name = self.axis_name[dimension]
        if name == "r":
            buff, mask = self._ortho_pixelize(
                data_source, field, bounds, size, antialias, dimension, periodic
            )
        elif name in ("theta", "phi"):
            if name == "theta":
                # This is admittedly a very hacky way to resolve a bug
                # it's very likely that the *right* fix would have to
                # be applied upstream of this function, *but* this case
                # has never worked properly so maybe it's still preferable to
                # not having a solution ?
                # see https://github.com/yt-project/yt/pull/3533
                bounds = (*bounds[2:4], *bounds[:2])
            buff, mask = self._cyl_pixelize(
                data_source, field, bounds, size, antialias, dimension
            )
        else:
            raise NotImplementedError

        if return_mask:
            assert mask is None or mask.dtype == bool
            return buff, mask
        else:
            return buff

    def pixelize_line(self, field, start_point, end_point, npoints):
        raise NotImplementedError

    def _ortho_pixelize(
        self, data_source, field, bounds, size, antialias, dim, periodic
    ):
        # use Aitoff projection
        # http://paulbourke.net/geometry/transformationprojection/
        bounds = tuple(_.ndview for _ in self._aitoff_bounds)
        buff, mask = pixelize_aitoff(
            azimuth=data_source["py"],
            dazimuth=data_source["pdy"],
            colatitude=data_source["px"],
            dcolatitude=data_source["pdx"],
            buff_size=size,
            field=data_source[field],
            bounds=bounds,
            input_img=None,
            azimuth_offset=0,
            colatitude_offset=0,
            return_mask=True,
        )
        return buff.T, mask.T

    def _cyl_pixelize(self, data_source, field, bounds, size, antialias, dimension):
        name = self.axis_name[dimension]
        buff = np.full((size[1], size[0]), np.nan, dtype="f8")
        if name == "theta":
            # TODO incorporate where we are in theta rather than 0
            x_native = np.array((data_source["px"], np.zeros_like(data_source["px"]), data_source["py"]))
            dx_native = np.array((data_source["pdx"], np.zeros_like(data_source["pdx"]), data_source["pdy"]))
            r, _, phi = self._transform.transform(x_native)
            dr, _, dphi = self._transform.transform_cellshape(x_native, dx_native)
            mask = pixelize_cylinder(
                buff,
                r,
                dr,
                phi,
                dphi,
                data_source[field],
                bounds,
                return_mask=True,
            )
        elif name == "phi":
            x_native = np.array((data_source["px"], data_source["py"], np.zeros_like(data_source["px"])))
            dx_native = np.array((data_source["pdx"], data_source["pdy"], np.zeros_like(data_source["pdx"])))
            r, th, _ = self._transform.transform(x_native)
            dr, dth, _ = self._transform.transform_cellshape(x_native, dx_native)
            # Note that we feed in buff.T here
            mask = pixelize_cylinder(
                buff.T,
                r,
                dr,
                th,
                dth,
                data_source[field],
                bounds,
                return_mask=True,
            ).T
        else:
            raise RuntimeError
        return buff, mask

    def convert_from_cartesian(self, coord):
        raise NotImplementedError

    def convert_to_cartesian(self, coord):
        if isinstance(coord, np.ndarray) and len(coord.shape) > 1:
            ri = self.axis_id["r"]
            thetai = self.axis_id["theta"]
            phii = self.axis_id["phi"]
            r, theta, phi = self._transform.transform((coord[:, ri], coord[:, thetai], coord[:, phii]))
            nc = np.zeros_like(coord)
            # r, theta, phi
            nc[:, ri] = np.cos(phi) * np.sin(theta) * r
            nc[:, thetai] = np.sin(phi) * np.sin(theta) * r
            nc[:, phii] = np.cos(theta) * r
        else:
            r, theta, phi = self._transform.transform(coord)
            nc = (
                np.cos(phi) * np.sin(theta) * r,
                np.sin(phi) * np.sin(theta) * r,
                np.cos(theta) * r,
            )
        return nc

    def convert_to_cylindrical(self, coord):
        raise NotImplementedError

    def convert_from_cylindrical(self, coord):
        raise NotImplementedError

    def convert_to_spherical(self, coord):
        raise NotImplementedError

    def convert_from_spherical(self, coord):
        raise NotImplementedError

    _image_axis_name = None

    @property
    def image_axis_name(self):
        if self._image_axis_name is not None:
            return self._image_axis_name
        # This is the x and y axes labels that get displayed.  For
        # non-Cartesian coordinates, we usually want to override these for
        # Cartesian coordinates, since we transform them.
        rv = {
            self.axis_id["r"]: (
                # these are the Hammer-Aitoff normalized coordinates
                # conventions:
                #   - theta is the colatitude, from 0 to PI
                #   - bartheta is the latitude, from -PI/2 to +PI/2 (bartheta = PI/2 - theta)
                #   - phi is the azimuth, from 0 to 2PI
                #   - lambda is the longitude, from -PI to PI (lambda = phi - PI)
                r"\frac{2\cos(\mathrm{\bar{\theta}})\sin(\lambda/2)}{\sqrt{1 + \cos(\bar{\theta}) \cos(\lambda/2)}}",
                r"\frac{sin(\bar{\theta})}{\sqrt{1 + \cos(\bar{\theta}) \cos(\lambda/2)}}",
                "\\theta",
            ),
            self.axis_id["theta"]: ("x / \\sin(\\theta)", "y / \\sin(\\theta)"),
            self.axis_id["phi"]: ("R", "z"),
        }
        for i in list(rv.keys()):
            rv[self.axis_name[i]] = rv[i]
            rv[self.axis_name[i].capitalize()] = rv[i]
        self._image_axis_name = rv
        return rv

    _x_pairs = (("r", "theta"), ("theta", "r"), ("phi", "r"))
    _y_pairs = (("r", "phi"), ("theta", "phi"), ("phi", "theta"))

    @property
    def period(self):
        embed_left_edge = self._transform.transform(self.ds.domain_left_edge)
        embed_right_edge = self._transform.transform(self.ds.domain_right_edge)
        return embed_right_edge - embed_left_edge

    @cached_property
    def _poloidal_bounds(self):
        ri = self.axis_id["r"]
        ti = self.axis_id["theta"]
        # Dataset edges are in native coordinates, transform them
        embed_left_edge = self._transform.transform(self.ds.domain_left_edge)
        embed_right_edge = self._transform.transform(self.ds.domain_right_edge)
        rmin = embed_left_edge[ri] * self.ds.length_unit
        rmax = embed_right_edge[ri] * self.ds.length_unit
        thetamin = embed_left_edge[ti]
        thetamax = embed_right_edge[ti]
        corners = [
            (rmin, thetamin),
            (rmin, thetamax),
            (rmax, thetamin),
            (rmax, thetamax),
        ]

        def to_poloidal_plane(r, theta):
            # take a r, theta position and return
            # cylindrical R, z coordinates
            R = r * np.sin(theta)
            z = r * np.cos(theta)
            return R, z

        cylindrical_corner_coords = [to_poloidal_plane(*corner) for corner in corners]

        #thetamin = thetamin.d
        #thetamax = thetamax.d

        Rmin = min(R for R, z in cylindrical_corner_coords)

        if thetamin <= np.pi / 2 <= thetamax:
            Rmax = rmax
        else:
            Rmax = max(R for R, z in cylindrical_corner_coords)

        zmin = min(z for R, z in cylindrical_corner_coords)
        zmax = max(z for R, z in cylindrical_corner_coords)

        return Rmin, Rmax, zmin, zmax

    @cached_property
    def _conic_bounds(self):
        return _get_polar_bounds(self, axes=("r", "phi"))

    @cached_property
    def _aitoff_bounds(self):
        # at the time of writing this function, yt's support for curvilinear
        # coordinates is a bit hacky, as many components of the system still
        # expect to receive coordinates with a length dimension. Ultimately
        # this is not needed but calls for a large refactor.
        ONE = self.ds.quan(1, "code_length")

        # colatitude
        ti = self.axis_id["theta"]
        thetamin = self.ds.domain_left_edge[ti]
        thetamax = self.ds.domain_right_edge[ti]
        
        # latitude
        latmin = ONE * np.pi / 2 - thetamax
        latmax = ONE * np.pi / 2 - thetamin

        # azimuth
        pi = self.axis_id["phi"]
        phimin = self.ds.domain_left_edge[pi]
        phimax = self.ds.domain_right_edge[pi]
        # longitude
        lonmin = phimin - ONE * np.pi
        lonmax = phimax - ONE * np.pi

        corners = [
            (latmin, lonmin),
            (latmin, lonmax),
            (latmax, lonmin),
            (latmax, lonmax),
        ]

        def aitoff_z(latitude, longitude):
            return np.sqrt(1 + np.cos(latitude) * np.cos(longitude / 2))

        def aitoff_x(latitude, longitude):
            return (
                2
                * np.cos(latitude)
                * np.sin(longitude / 2)
                / aitoff_z(latitude, longitude)
            )

        def aitoff_y(latitude, longitude):
            return np.sin(latitude) / aitoff_z(latitude, longitude)

        def to_aitoff_plane(latitude, longitude):
            return aitoff_x(latitude, longitude), aitoff_y(latitude, longitude)

        aitoff_corner_coords = [to_aitoff_plane(*corner) for corner in corners]

        xmin = ONE * min(x for x, y in aitoff_corner_coords)
        xmax = ONE * max(x for x, y in aitoff_corner_coords)

        # theta is the colatitude
        # What this branch is meant to do is check whether the equator (latitude = 0)
        # is included in the domain.

        if latmin < 0 < latmax:
            xmin = min(xmin, ONE * aitoff_x(0, lonmin))
            xmax = max(xmax, ONE * aitoff_x(0, lonmax))

        # the y direction is more straightforward because aitoff-projected parallels (y)
        # draw a convex shape, while aitoff-projected meridians (x) draw a concave shape
        ymin = ONE * min(y for x, y in aitoff_corner_coords)
        ymax = ONE * max(y for x, y in aitoff_corner_coords)

        return xmin, xmax, ymin, ymax

    def sanitize_center(self, center, axis):
        center, display_center = super().sanitize_center(center, axis)
        name = self.axis_name[axis]
        if name == "r":
            xxmin, xxmax, yymin, yymax = self._aitoff_bounds
            xc = (xxmin + xxmax) / 2
            yc = (yymin + yymax) / 2
            display_center = (0 * xc, xc, yc)
        elif name == "theta":
            xxmin, xxmax, yymin, yymax = self._conic_bounds
            xc = (xxmin + xxmax) / 2
            yc = (yymin + yymax) / 2
            display_center = (xc, 0 * xc, yc)
        elif name == "phi":
            Rmin, Rmax, zmin, zmax = self._poloidal_bounds
            xc = (Rmin + Rmax) / 2
            yc = (zmin + zmax) / 2
            display_center = (xc, yc)
        return center, display_center

    def sanitize_width(self, axis, width, depth):
        name = self.axis_name[axis]
        if width is not None:
            width = super().sanitize_width(axis, width, depth)
        elif name == "r":
            xxmin, xxmax, yymin, yymax = self._aitoff_bounds
            xw = xxmax - xxmin
            yw = yymax - yymin
            width = [xw, yw]
        elif name == "theta":
            # Remember, in spherical coordinates when we cut in theta,
            # we create a conic section
            xxmin, xxmax, yymin, yymax = self._conic_bounds
            xw = xxmax - xxmin
            yw = yymax - yymin
            width = [xw, yw]
        elif name == "phi":
            Rmin, Rmax, zmin, zmax = self._poloidal_bounds
            xw = Rmax - Rmin
            yw = zmax - zmin
            width = [xw, yw]
        return width
