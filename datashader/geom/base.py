import re
from functools import total_ordering
import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from datashader.datatypes import _RaggedElement, RaggedDtype, RaggedArray


@total_ordering
class Geom(_RaggedElement):
    def __repr__(self):
        data = [(x, y) for x, y in zip(self.xs, self.ys)]
        return "{typ}({data})".format(typ=self.__class__.__name__, data=data)

    @classmethod
    def _shapely_to_array_parts(cls, shape):
        raise NotImplementedError()

    @classmethod
    def from_shapely(cls, shape):
        shape_parts = cls._shapely_to_array_parts(shape)
        return cls(np.concatenate(shape_parts))

    @property
    def xs(self):
        return self.array[0::2]

    @property
    def ys(self):
        return self.array[1::2]


@register_extension_dtype
class GeomDtype(RaggedDtype):
    _type_name = "Geom"
    _subtype_re = re.compile(r"^geom\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return GeomArray


class GeomArray(RaggedArray):
    _element_type = Geom

    def __init__(self, *args, **kwargs):
        super(GeomArray, self).__init__(*args, **kwargs)
        # Validate that there are an even number of elements in each Geom element
        if (any(self.start_indices % 2) or
                (len(self.flat_array) - self.start_indices[-1]) % 2 > 0):
            raise ValueError("There must be an even number of elements in each row")

    @property
    def _dtype_class(self):
        return GeomDtype

    @property
    def xs(self):
        start_indices = self.start_indices // 2
        flat_array = self.flat_array[0::2]
        return RaggedArray({"start_indices": start_indices, "flat_array": flat_array})

    @property
    def ys(self):
        start_indices = self.start_indices // 2
        flat_array = self.flat_array[1::2]
        return RaggedArray({"start_indices": start_indices, "flat_array": flat_array})

    def to_geopandas(self):
        from geopandas.array import from_shapely
        return from_shapely([el.to_shapely() for el in self])

    @classmethod
    def from_geopandas(cls, ga):
        line_parts = [
            cls._element_type._shapely_to_array_parts(shape) for shape in ga
        ]
        line_lengths = [
            sum([len(part) for part in parts])
            for parts in line_parts
        ]
        flat_array = np.concatenate(
            [part for parts in line_parts for part in parts]
        )
        start_indices = np.concatenate(
            [[0], line_lengths[:-1]]
        ).astype('uint').cumsum()
        return cls({
            'start_indices': start_indices, 'flat_array': flat_array
        })
