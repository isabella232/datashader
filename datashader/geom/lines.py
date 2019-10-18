import re
from functools import total_ordering

from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.geom.base import Geom, GeomDtype, GeomArray


@total_ordering
class Lines(Geom):
    pass


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
