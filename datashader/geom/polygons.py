import re
from functools import total_ordering

import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray


@total_ordering
class Polygons(Geom):
    @staticmethod
    def _polygon_to_array_parts(polygon):
        import shapely.geometry as sg
        shape = sg.polygon.orient(polygon)
        exterior = np.asarray(shape.exterior.ctypes)
        polygon_parts = [exterior]
        hole_separator = np.array([-np.inf, -np.inf])
        for ring in shape.interiors:
            interior = np.asarray(ring.ctypes)
            polygon_parts.append(hole_separator)
            polygon_parts.append(interior)
        return polygon_parts

    @staticmethod
    def _polygons_to_array_parts(polygons):
        import shapely.geometry as sg
        if isinstance(polygons, sg.Polygon):
            # Single polygon
            return Polygons._polygon_to_array_parts(polygons)
        elif isinstance(polygons, sg.MultiPolygon):
            polygons = list(polygons)
            polygon_parts = Polygons._polygon_to_array_parts(polygons[0])
            polygon_separator = np.array([np.inf, np.inf])
            for polygon in polygons[1:]:
                polygon_parts.append(polygon_separator)
                polygon_parts.extend(Polygons._polygon_to_array_parts(polygon))
            return polygon_parts
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must an instance of
shapely.geometry.Polygon or shapely.geometry.MultiPolygon""")

    @staticmethod
    def from_shapely(shape):
        polygon_parts = Polygons._polygons_to_array_parts(shape)
        return Polygons(np.concatenate(polygon_parts))

    def to_shapely(self):
        import shapely.geometry as sg
        ring_breaks = np.concatenate(
            [[-2], np.nonzero(~np.isfinite(self.array))[0][0::2], [len(self.array)]]
        )
        polygon_breaks = set(np.concatenate(
            [[-2], np.nonzero(np.isposinf(self.array))[0][0::2], [len(self.array)]]
        ))

        # Build rings for both outer and holds
        rings = []
        for start, stop in zip(ring_breaks[:-1], ring_breaks[1:]):
            ring_array = self.array[start + 2: stop]
            ring_pairs = ring_array.reshape(len(ring_array) // 2, 2)
            rings.append(sg.LinearRing(ring_pairs))

        # Build polygons
        polygons = []
        outer = None
        holes = []
        for ring, start in zip(rings, ring_breaks[:-1]):
            if start in polygon_breaks:
                if outer:
                    # This is the first ring in a new polygon, construct shapely polygon
                    # with already collected rings
                    polygons.append(sg.Polygon(outer, holes))

                # Start collecting new polygon
                outer = ring
                holes = []
            else:
                # Ring is a hole
                holes.append(ring)

        # Build final polygon
        polygons.append(sg.Polygon(outer, holes))

        if len(polygons) == 1:
            return polygons[0]
        else:
            return sg.MultiPolygon(polygons)


@register_extension_dtype
class PolygonsDtype(GeomDtype):
    _type_name = "Polygons"
    _subtype_re = re.compile(r"^polygons\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return PolygonsArray


class PolygonsArray(GeomArray):
    _element_type = Polygons

    @property
    def _dtype_class(self):
        return PolygonsDtype

    @classmethod
    def from_geopandas(cls, ga):
        polygon_parts = [
            Polygons._polygons_to_array_parts(shape) for shape in ga
        ]
        polygon_lengths = [
            sum([len(part) for part in parts])
            for parts in polygon_parts
        ]
        flat_array = np.concatenate(
            [part for parts in polygon_parts for part in parts]
        )
        start_indices = np.concatenate(
            [[0], polygon_lengths[:-1]]
        ).astype('uint').cumsum()
        return PolygonsArray({
            'start_indices': start_indices, 'flat_array': flat_array
        })
