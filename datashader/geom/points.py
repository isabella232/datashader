import re
from functools import total_ordering

from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray
from datashader.geom.line import LinesDtype


@total_ordering
class Points(Geom):
    pass


@register_extension_dtype
class PointsDtype(GeomDtype):
    _type_name = "Points"
    _subtype_re = re.compile(r"^points\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return PointsArray


class PointsArray(GeomArray):
    _element_type = Points

    @property
    def _dtype_class(self):
        return LinesDtype
