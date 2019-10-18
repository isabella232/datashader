import re
from functools import total_ordering
import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray


@total_ordering
class Lines(Geom):
    @classmethod
    def _shapely_to_array_parts(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.LineString, sg.LinearRing)):
            # Single line
            return [np.asarray(shape.ctypes)]
        elif isinstance(shape, sg.MultiLineString):
            shape = list(shape)
            line_parts = [np.asarray(shape[0].ctypes)]
            line_separator = np.array([np.inf, np.inf])
            for line in shape[1:]:
                line_parts.append(line_separator)
                line_parts.append(np.asarray(line.ctypes))
            return line_parts
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of LineString,
MultiLineString, or LinearRing""".format(typ=type(shape).__name__))

    def to_shapely(self):
        import shapely.geometry as sg
        line_breaks = np.concatenate(
            [[-2], np.nonzero(~np.isfinite(self.array))[0][0::2], [len(self.array)]]
        )
        line_arrays = [self.array[start + 2:stop]
                       for start, stop in zip(line_breaks[:-1], line_breaks[1:])]

        lines = [sg.LineString(line_array.reshape(len(line_array) // 2, 2))
                 for line_array in line_arrays]

        if len(lines) == 1:
            return lines[0]
        else:
            return sg.MultiLineString(lines)


@register_extension_dtype
class LinesDtype(GeomDtype):
    _type_name = "Lines"
    _subtype_re = re.compile(r"^lines\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return LinesArray


class LinesArray(GeomArray):
    _element_type = Lines

    @property
    def _dtype_class(self):
        return LinesDtype
