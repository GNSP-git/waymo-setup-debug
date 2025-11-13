# obb_distance.pyx
# Cython-accelerated oriented bounding box (OBB) distance in 2D.

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: infer_types=True

from libc.math cimport cos, sin, sqrt, fabs

cdef inline void rect_corners(
    double cx, double cy,
    double hl, double hw,
    double c, double s,
    double* xs, double* ys
):
    """
    Compute 4 corners of an oriented rectangle.
    (hl, hw) = half-length, half-width.
    heading given via cos,sin.
    Order: +L+W, +L-W, -L-W, -L+W
    """
    cdef double dx0 = hl
    cdef double dy0 = hw

    # (+hl, +hw)
    xs[0] = cx + c*dx0 - s*dy0
    ys[0] = cy + s*dx0 + c*dy0

    # (+hl, -hw)
    xs[1] = cx + c*dx0 - s*(-dy0)
    ys[1] = cy + s*dx0 + c*(-dy0)

    # (-hl, -hw)
    xs[2] = cx + c*(-dx0) - s*(-dy0)
    ys[2] = cy + s*(-dx0) + c*(-dy0)

    # (-hl, +hw)
    xs[3] = cx + c*(-dx0) - s*(dy0)
    ys[3] = cy + s*(-dx0) + c*(dy0)


cdef inline void project_corners(
    double* xs, double* ys,
    double ax, double ay,
    double* out_min, double* out_max
):
    """
    Project 4 points onto axis (ax, ay).
    Axis need not be unit; scale cancels in overlap test.
    """
    cdef double p
    p = xs[0]*ax + ys[0]*ay
    out_min[0] = p
    out_max[0] = p

    p = xs[1]*ax + ys[1]*ay
    if p < out_min[0]:
        out_min[0] = p
    elif p > out_max[0]:
        out_max[0] = p

    p = xs[2]*ax + ys[2]*ay
    if p < out_min[0]:
        out_min[0] = p
    elif p > out_max[0]:
        out_max[0] = p

    p = xs[3]*ax + ys[3]*ay
    if p < out_min[0]:
        out_min[0] = p
    elif p > out_max[0]:
        out_max[0] = p


cdef inline bint rects_intersect(double* ax, double* ay,
                                 double* bx, double* by):
    """
    SAT overlap test for two oriented rectangles.

    ax, ay: 4 corners of rect A
    bx, by: 4 corners of rect B
    """
    cdef double minA, maxA, minB, maxB
    cdef double axis_x, axis_y
    cdef int i, j

    # 4 axes: 2 from A edges, 2 from B edges
    for i in range(2):
        # Edge i -> i+1 for A
        axis_x = ax[i+1] - ax[i]
        axis_y = ay[i+1] - ay[i]

        # Project
        project_corners(ax, ay, axis_x, axis_y, &minA, &maxA)
        project_corners(bx, by, axis_x, axis_y, &minB, &maxB)

        if maxA < minB or maxB < minA:
            return 0  # separated

    for i in range(2):
        # Edge i -> i+1 for B
        axis_x = bx[i+1] - bx[i]
        axis_y = by[i+1] - by[i]

        project_corners(ax, ay, axis_x, axis_y, &minA, &maxA)
        project_corners(bx, by, axis_x, axis_y, &minB, &maxB)

        if maxA < minB or maxB < minA:
            return 0

    return 1  # no separating axis => intersect


cdef inline double dist_point_seg(
    double px, double py,
    double x1, double y1,
    double x2, double y2
):
    """
    Squared distance from point P to segment [A,B].
    """
    cdef double vx = x2 - x1
    cdef double vy = y2 - y1
    cdef double wx = px - x1
    cdef double wy = py - y1

    cdef double c1 = vx*wx + vy*wy
    if c1 <= 0.0:
        # Closest to A
        return (px - x1)*(px - x1) + (py - y1)*(py - y1)

    cdef double c2 = vx*vx + vy*vy
    if c2 <= c1:
        # Closest to B
        return (px - x2)*(px - x2) + (py - y2)*(py - y2)

    cdef double t = c1 / c2
    cdef double projx = x1 + t * vx
    cdef double projy = y1 + t * vy
    return (px - projx)*(px - projx) + (py - projy)*(py - projy)


cdef inline double rect_rect_min_dist2(double* ax, double* ay,
                                       double* bx, double* by):
    """
    Min squared distance between two non-intersecting rectangles.
    We check point-to-segment distances both ways.
    """
    cdef double d2 = 1e30
    cdef double cand
    cdef int i, j, ni, nj

    ni = 4
    nj = 4

    # A vertices to B edges
    for i in range(ni):
        for j in range(nj):
            cand = dist_point_seg(
                ax[i], ay[i],
                bx[j],   by[j],
                bx[(j+1) & 3], by[(j+1) & 3]
            )
            if cand < d2:
                d2 = cand

    # B vertices to A edges
    for i in range(nj):
        for j in range(ni):
            cand = dist_point_seg(
                bx[i], by[i],
                ax[j],   ay[j],
                ax[(j+1) & 3], ay[(j+1) & 3]
            )
            if cand < d2:
                d2 = cand

    return d2


def obb_min_distance(
    double cx1, double cy1,
    double length1, double width1,
    double heading1,
    double cx2, double cy2,
    double length2, double width2,
    double heading2
):
    """
    Public API: minimum distance between two 2D oriented boxes in meters.
    Boxes are defined in Waymo ground-plane coordinates.

    length, width in meters.
    heading in radians (Waymo convention).
    """
    cdef double hl1 = 0.5 * length1
    cdef double hw1 = 0.5 * width1
    cdef double hl2 = 0.5 * length2
    cdef double hw2 = 0.5 * width2

    cdef double c1 = cos(heading1)
    cdef double s1 = sin(heading1)
    cdef double c2 = cos(heading2)
    cdef double s2 = sin(heading2)

    cdef double ax[4], ay[4]
    cdef double bx[4], by[4]
    cdef double d2

    rect_corners(cx1, cy1, hl1, hw1, c1, s1, ax, ay)
    rect_corners(cx2, cy2, hl2, hw2, c2, s2, bx, by)


    if rects_intersect(ax, ay, bx, by):
       return 0.0
    d2 = rect_rect_min_dist2(ax, ay, bx, by)
    if d2 < 0.0:
      d2 = 0.0

    return sqrt(d2)
