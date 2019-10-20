from math import isnan, inf, isfinite

from toolz import memoize
import numpy as np

from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeomLike
from datashader.utils import ngjit


class PolygonGeom(_GeomLike):
    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_segment = _build_draw_polygon(
            append, map_onto_pixel, expand_aggs_and_cols
        )

        perform_extend_cpu = _build_extend_polygon_geom(
            draw_segment, expand_aggs_and_cols
        )
        geom_name = self.geometry

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df)
            geom_array = df[geom_name].array
            # line may be clipped, then mapped to pixels
            perform_extend_cpu(
                sx, tx, sy, ty,
                xmin, xmax, ymin, ymax,
                geom_array, *aggs_and_cols
            )

        return extend


def _build_draw_polygon(append, map_onto_pixel, expand_aggs_and_cols):
    @ngjit
    @expand_aggs_and_cols
    def draw_polygon(
            i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
            start_index, stop_index, flat, *aggs_and_cols
    ):
        """Draw a polygon using a winding-number scan-line algorithm
        """
        # First pass, compute bounding box in data coordinates and count number of edges
        num_edges = 0
        poly_xmin = inf
        poly_ymin = inf
        poly_xmax = -inf
        poly_ymax = -inf
        for j in range(start_index, stop_index - 2, 2):
            x = flat[j]
            y = flat[j + 1]
            if isfinite(x) and isfinite(y):
                poly_xmin = min(poly_xmin, x)
                poly_ymin = min(poly_ymin, y)
                poly_xmax = max(poly_xmax, x)
                poly_ymax = max(poly_ymax, y)
                if isfinite(flat[j + 2]) and isfinite(flat[j + 3]):
                    # Valid edge
                    num_edges += 1

        # skip polygon if outside viewport
        if (poly_xmax < xmin or poly_ymin > xmax
                or poly_ymax < ymin or poly_xmin > xmax):
            return

        # Compute pixel bounds for polygon
        startxi, startyi = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, poly_xmin, poly_ymin
        )
        stopxi, stopyi = map_onto_pixel(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, poly_xmax, poly_ymax
        )
        stopxi += 1
        stopyi += 1

        # Handle subpixel polygons (pixel width or height of polygon is 1)
        if (stopxi - startxi) == 1 or (stopyi - startyi) == 1:
            for yi in range(startyi, stopyi):
                for xi in range(startxi, stopxi):
                    append(i, xi, yi, *aggs_and_cols)
            return

        # Build arrays of edge pixel coordinates
        xs = np.zeros((num_edges, 2), dtype=np.int32)
        ys = np.zeros((num_edges, 2), dtype=np.int32)
        yincreasing = np.zeros(num_edges, dtype=np.int8)
        xdecreasing = np.zeros(num_edges, dtype=np.int8)
        ei = 0
        for j in range(start_index, stop_index - 2, 2):
            x0 = flat[j]
            y0 = flat[j + 1]
            x1 = flat[j + 2]
            y1 = flat[j + 3]
            if isfinite(x0) and isfinite(y0) and isfinite(y0) and isfinite(y1):
                x0i, y0i = map_onto_pixel(
                    sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y0
                )
                xs[ei, 0] = x0i
                ys[ei, 0] = y0i

                x1i, y1i = map_onto_pixel(
                    sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y1
                )
                xs[ei, 1] = x1i
                ys[ei, 1] = y1i

                if y1 > y0:
                    yincreasing[ei] = 1
                elif y1 < y0:
                    yincreasing[ei] = -1

                if x1 > x0:
                    xdecreasing[ei] = -1
                elif x1 < x0:
                    xdecreasing[ei] = 1

                ei += 1

        # Initialize array indicating which edges are still eligible for processing
        eligible = np.ones(num_edges, dtype=np.int8)

        # Perform scan-line algorithm
        for yi in range(startyi, stopyi):
            # All edges eligible at start of new row
            eligible.fill(1)
            for xi in range(startxi, stopxi):
                # Init winding number
                winding_number = 0
                for ei in range(num_edges):
                    if eligible[ei] == 0:
                        # We've already determined that edge is above, below, or left
                        # of edge for the current pixel
                        continue

                    # Get edge coordinates
                    x0i = xs[ei, 0]
                    x1i = xs[ei, 1]
                    y0i = ys[ei, 0]
                    y1i = ys[ei, 1]

                    # Reject edges that are above, below, or left of current pixel
                    if ((y0i > yi and y1i > yi) or
                            (y0i < yi and y1i < yi) or
                            (x0i < xi and x1i < xi)):
                        # Edge not eligible for any remaining pixel in this row
                        eligible[ei] = 0
                        continue

                    # Not correct, need next increasing?
                    if yincreasing[ei] != xdecreasing[ei] and y1i == yi:
                        eligible[ei] = 0
                        continue

                    if x0i > xi and x1i > xi:
                        # Edge is fully to the right of the pixel, so we know ray to the
                        # the right of pixel intersects edge.
                        winding_number += yincreasing[ei]
                    elif y0i == y1i or x0i == x1i:
                        # Horizontal or vertical in pixel space. Given prior checks
                        # we know that edge intersects with pixel
                        winding_number += (yincreasing[ei]
                                           if yincreasing[ei] != 0
                                           else xdecreasing[ei])
                    else:
                        # Now check if edge is to the right of pixel using cross product
                        # A is vector from pixel to first vertex
                        ax = x0i - xi
                        ay = y0i - yi

                        # B is vector from pixel to second vertex
                        bx = x1i - xi
                        by = y1i - yi

                        # Compute cross product of B and A
                        bxa = (bx * ay - by * ax)

                        if bxa * yincreasing[ei] < 0:
                            # Edge to the right
                            winding_number += yincreasing[ei]
                        else:
                            # Edge to left, not eligible for any remaining pixel in row
                            eligible[ei] = 0
                            continue

                if winding_number != 0:
                    # If winding number is not zero, point
                    # is inside polygon
                    append(i, xi, yi, *aggs_and_cols)

    return draw_polygon


def _build_extend_polygon_geom(
        draw_polygon, expand_aggs_and_cols
):
    def extend_cpu(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, geom_array, *aggs_and_cols
    ):
        start_i = geom_array.start_indices
        flat = geom_array.flat_array

        extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, start_i, flat, *aggs_and_cols
        )

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(
            sx, tx, sy, ty, xmin, xmax, ymin, ymax, start_i, flat, *aggs_and_cols
    ):
        nrows = len(start_i)
        flat_len = len(flat)

        for i in range(nrows):
            # Get x index range
            start_index = start_i[i]
            stop_index = (start_i[i + 1] if i < nrows - 1 else flat_len)
            draw_polygon(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax,
                         start_index, stop_index, flat, *aggs_and_cols)

    return extend_cpu
