import itertools
import random
from functools import partial
from itertools import starmap
import math
import logging

import networkx as nx
import numpy as np

import mathutils
import bmesh
import bpy

from mathutils import Matrix, Vector

logger = logging.getLogger('geodesic')

TOL = 1e-4
VERT_TOL = 1e-2

def rotate_about_axis(axis, theta):
    """
    rodrigues formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis.normalized()
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return Matrix([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def build_graph_vert_pairs(G, it, vertex_group=None, min_weight=0.1):
    for u, v in it:
        if G.has_edge(u.index, v.index):
            continue
        d = (u.co - v.co).length
        if vertex_group is not None:
            # TODO the "right" thing to do is take the sum of a weighted average of weights encountered across the path for some sample amount
            try:
                half_weight = (vertex_group.weight(u.index) + vertex_group.weight(v.index)) / 2
            except Exception:
                continue
            # stretch weight range from [0, 1] to [min_weight, 2]
            multiplier = max(half_weight, min_weight / 2) * 2
            d *= multiplier
        G.add_edge(u.index, v.index, weight=d)

def build_graph(mesh, vertex_group=None, min_weight=0.1, cross_faces=True):
    G = nx.Graph()

    if cross_faces:
        it = itertools.chain.from_iterable(itertools.combinations(f.verts, 2) for f in mesh.faces)
    else:
        it = (e.verts for e in mesh.edges)

    build_graph_vert_pairs(G, it, vertex_group, min_weight)

    mesh.verts.ensure_lookup_table()
    for node in G:
        G.add_node(node, vert=mesh.verts[node].co)

    return G, mesh.verts

def subdivide_edge_n(e, n):
    # subdividing once is a 0.5 split, twice is a 0.33 .33 split etc
    split = 1 / (n + 1)
    pointing = (e.verts[1].co - e.verts[0].co)
    vert = e.verts[0].co
    for i in range(1, n + 1):
        yield vert + pointing * split * i

def remove_path(G, nodes):
    for i in range(len(nodes) - 1):
        G.remove_edge(nodes[i], nodes[i + 1])

# def build_graph_subdivide_n(mesh, n):

#     verts = [x.co for x in mesh.verts]
#     edges = []

#     for e in mesh.edges:
#         edges.append([i + len(verts) for i in range(n)])
#         l = list(subdivide_edge_n(e, n))
#         verts.extend(subdivide_edge_n(e, n))

#     G, _ = build_graph(mesh)

#     for f in mesh.faces:
#         for e1, e2 in itertools.combinations(f.edges, 2):
#             for v1, v2 in itertools.product(edges[e1.index], edges[e2.index]):
#                 if G.has_edge(v1, v2):
#                     continue
#                 d = (verts[v1] - verts[v2]).length
#                 G.add_edge(v1, v2, weight=d)

#     return G, verts

def make_empty_curve(name='Curve', type='BEZIER', handle_type='AUTO'):
    curve = bpy.data.objects.new(name, bpy.data.curves.new(name, 'CURVE'))
    bpy.context.collection.objects.link(curve)
    curve.data.dimensions = '3D'

    return curve

def make_spline(curve, points, name='Spline', type='BEZIER', handle_type='AUTO'):
    spline = curve.data.splines.new(type)

    spline_points = spline.bezier_points if type == 'BEZIER' else spline.points
    spline_points.add(len(points) - 1)

    assert len(points) == len(spline_points)

    for sp, p in zip(spline_points, points):
        if isinstance(p, bmesh.types.BMVert):
            p = p.co
        sp.co = p
        if type == 'BEZIER':
            sp.handle_left_type = handle_type
            sp.handle_right_type = handle_type

    return spline

def make_curve(points, name='Curve', type='BEZIER', handle_type='AUTO'):
    curve = make_empty_curve(name=name, type=type, handle_type=handle_type)
    make_spline(curve, points, type=type, handle_type=handle_type)

    return curve

def set_spline_handles(spline, handle_type):
    if not spline.type == 'BEZIER':
        return
    for p in spline.bezier_points:
        p.handle_left_type = handle_type
        p.handle_right_type = handle_type

def set_curve_handles(curve, handle_type):
    for spline in curve.data.splines:
        set_spline_handles(spline, handle_type)

def _vert_or_index(v):
    return getattr(v, 'index', v)

def try_shortest_path(G, a, b):
    a = _vert_or_index(a)
    b = _vert_or_index(b)

    try:
        return nx.algorithms.shortest_path(G, a, b, weight='weight')

    except nx.exception.NodeNotFound:
        logger.debug(f'NodeNotFound, {a} -> {b} vertex must have not been in the vertex group')

    except nx.exception.NetworkXNoPath:
        logger.debug(f'No such path {a} -> {b}')

    return None

def get_path_points(G, path):
    return [G.nodes[i]['vert'] for i in path]

def path_weight(G, path):
    ret = 0
    for i in range(len(path) - 1):
        ret += G[path[i]][path[i + 1]]['weight']
    return ret

def closest_vertex_on_face(mesh, face_index, point):
    mesh.faces.ensure_lookup_table()
    return min(mesh.faces[face_index].verts, key=lambda v: (v.co - point).length_squared)

def snap_curve_splines(G, mesh, obj, curve, vertex_group=None, cross_faces=False, closest_vert=True):
    remove = []

    splines = list(curve.data.splines)
    for spline in splines:
        points = spline.bezier_points if spline.type == 'BEZIER' else spline.points

        if len(points) < 2:
            continue
        start = points[0].co
        end = points[-1].co

        succ1, loc1, normal1, face_index1 = obj.closest_point_on_mesh(start)
        succ2, loc2, normal2, face_index2 = obj.closest_point_on_mesh(end)

        if not succ1 or not succ2:
            continue

        if closest_vert:
            a = closest_vertex_on_face(mesh, face_index1, obj.matrix_world @ loc1).index
            b = closest_vertex_on_face(mesh, face_index2, obj.matrix_world @ loc2).index
            path = try_shortest_path(G, a, b)
            if path is None:
                continue
            points = get_path_points(G, path)

        else:
            # the start and end are on a face, try each path from each vert to each other vert and take the one with least total path length
            p1 = loc1
            p2 = loc2
            mesh.faces.ensure_lookup_table()
            paths = filter(None, starmap(partial(try_shortest_path, G), itertools.product(mesh.faces[face_index1].verts, mesh.faces[face_index2].verts)))
            def score(path):
                return (
                    path_weight(G, path) +
                    (p1 - G.nodes[path[0]]['vert']).length +
                    (p2 - G.nodes[path[-1]]['vert']).length
                )

            path = min(paths, key=score, default=None)
            if path is None:
                continue
            points = [p1] + get_path_points(G, path) + [p2]

        make_spline(curve, points, type=spline.type)
        remove.append(spline)

    for x in remove:
        curve.data.splines.remove(x)

# def make_multiple_paths(G, n):
#     n = 5
#     while n:
#         if len(G) < 2:
#             break

#         # TODO could be more efficient
#         nodes = list(G)
#         random.shuffle(nodes)
#         a, b = nodes[:2]

#         try:
#             path = nx.algorithms.shortest_path(G, a, b, weight='weight')
#             if len(path) < 10:
#                 continue
#             remove_path(G, path)

#             curve = make_curve([verts[i] for i in path], handle_type='AUTO')
#             curve.data.bevel_depth = max(0.01, random.gauss(0.03, 0.01))
#             n -= 1

#         except nx.exception.NodeNotFound:
#             logger.debug('NodeNotFound, vertex must have not been in the vertex group')

#         except nx.exception.NetworkXNoPath:
#             logger.debug('No such path')

# expected to be called with only edges containing 1 or 2 faces
def next_face(face, edge):
    for f in edge.link_faces:
        if f != face:
            return f
    return None

def line_line_intersection(a, b, c, d):
    """3space segment intersection"""
    ret = mathutils.geometry.intersect_line_line(a, b, c, d)
    if ret is None:
        return None
    x, y = ret

    if ((x - y).length > TOL or        # lines dont intersect
        not point_on_line(x, a, b) or  # intersection not on line 1
        not point_on_line(y, c, d)     # intersection not on line 2
        ):
        return None

    return x

def point_on_line(pt, line1, line2):
    intersection, pct = mathutils.geometry.intersect_point_line(pt, line1, line2)

    return (
        (pt - intersection).length_squared < TOL and  # closest point on line is this point
        0 <= pct <= 1                                 # and it exists between the endpoints
    )

# I don't know of a nicer way to do this
def make_face_face_rotation_matrix(face1, face2, axis):
    dihedral = face1.normal.angle(face2.normal)
    m = rotate_about_axis(axis, dihedral)
    if math.isclose((m @ face1.normal).dot(face2.normal), 1, rel_tol=TOL):
        return m

    m = rotate_about_axis(axis, -dihedral)
    assert math.isclose((m @ face1.normal).dot(face2.normal), 1, rel_tol=TOL)

    return m

def closest_point_on_mesh(obj, point):
    succ, loc, normal, face_index = obj.closest_point_on_mesh(point)
    if not succ:
        raise ValueError('failed to get closest_point_on_mesh')
    return loc, normal, face_index

def vector_rejection(a, b):
    return a - a.project(b)

def walk_along_mesh(obj, mesh, start, heading):
    """
    Expects heading to be along the face its starting on already, otherwise we project it onto the face
    Returns Tuple[
        list of N points including start of the walk along a mesh in direction of heading with length of heading,
        list of N-1 face indices where the line from points[i] to points[i + 1] lies on face[i]
    ]
    """
    loc, normal, face_index = closest_point_on_mesh(obj, start)

    mesh.faces.ensure_lookup_table()

    points = [loc]
    face = mesh.faces[face_index]
    faces = []

    # TODO if we are given start at exactly a vert, the face is ambiguous, but maybe we should be nice and try each face
    # and take the one with the smallest dot since the heading may imply which face was "intended"
    # of course if you have coplanar faces the dot won't tell you enough, you'd then want to check which one the heading
    # actually produces a path that doesn't end right away

    if not math.isclose(heading.dot(face.normal), 0):
        # logger.debug('reprojection heading onto face because dot is {:.2f}'.format(heading.dot(face.normal)))
        l = heading.length
        heading = vector_rejection(heading, face.normal)
        heading.normalize()
        heading *= l

    while heading.length_squared:
        a = points[-1]
        b = a + heading

        # find first edge that intersects our heading
        intersection = None
        for edge in face.edges:
            v1 = edge.verts[0].co
            v2 = edge.verts[1].co

            if point_on_line(a, v1, v2):
                continue

            intersection = line_line_intersection(a, b, v1, v2)
            if intersection is not None:
                break

        # end of the road
        if intersection is None:
            # logger.debug('INTERSECTION IS NONE')
            points.append(b)
            faces.append(face.index)
            assert len(points) - 1 == len(faces)
            return points, faces

        # back to start
        # TODO this won't always be useful if the start is off an edge, we would have to check that an existing segment.dot(new_segment) == 0
        if (intersection - points[0]).length < TOL:
            # logger.debug('BACK TO START')
            assert len(points) - 1 == len(faces)
            return points, faces

        # hit a vert
        if (intersection - v1).length < VERT_TOL or (intersection - v2).length < VERT_TOL:
            # logger.debug('HIT A VERT')
            points.append(intersection)
            faces.append(face.index)
            assert len(points) - 1 == len(faces)
            return points, faces

        points.append(intersection)

        new_face = next_face(face, edge)
        if new_face is None:
            logger.debug('NEWFACE IS NONE')
            faces.append(face.index)
            assert len(points) - 1 == len(faces)
            return points, faces

        assert (heading.length) >= (intersection - a).length

        heading -= (intersection - a)  # subtract off the amount we have
        heading = make_face_face_rotation_matrix(face, new_face, v2 - v1) @ heading

        faces.append(face.index)
        face = new_face

    assert False

def const(n):
    return n

def generate_walks(curve, obj, mesh, starts, gen_n_spokes, gen_angles, gen_lengths):
    mesh.faces.ensure_lookup_table()

    for i, start in enumerate(starts):
        spokes = gen_n_spokes()
        angles = gen_angles(spokes)
        lengths = gen_lengths(spokes)

        if isinstance(start, tuple):
            start, heading = start
            heading = heading.normalized() * length
        else:
            loc, normal, face_index = closest_point_on_mesh(obj, start)
            # choose arb heading
            heading = rotate_about_axis(normal, random.uniform(-math.pi, math.pi)) @ normal.orthogonal()

        for length, angle in zip(lengths, angles):
            h = (rotate_about_axis(normal, angle) @ heading) * length
            points, faces = walk_along_mesh(obj, mesh, start, h)
            make_spline(curve, points)

def dev():
    C = bpy.context
    D = bpy.data

    to_remove = [x for x in D.objects if x.name.startswith('Curve')]
    for x in to_remove:
        D.objects.remove(x, do_unlink=True)

    obj = D.objects['Dodec']
    m = bmesh.new()
    m.from_mesh(obj.data)
    # m.transform(obj.matrix_world)

    # shortest path test
    G, verts = build_graph(m, vertex_group=obj.vertex_groups['Group'], cross_faces=True)
    bc = D.objects['BezierCurve']
    snap_curve_splines(G, m, obj, bc, closest_vert=False)
    bc.matrix_world = obj.matrix_world

    obj = D.objects['Plane']
    m = bmesh.new()
    m.from_mesh(obj.data)

    # path runs into edge which has no other face
    points, faces = walk_along_mesh(obj, m, Vector((-.99, -.99, 1)), Vector((1, 1.56, 0)).normalized() * 3)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    obj = D.objects['Cube.000']
    m = bmesh.new()
    m.from_mesh(obj.data)

    # ends on first face
    points, faces = walk_along_mesh(obj, m, Vector((-.99, -.99, 1)), Vector((1, 1.56, 0)).normalized() * 1)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    # goes to second face
    points, faces = walk_along_mesh(obj, m, Vector((-0.99, -0.99, 1)), Vector((1, .76, 0)).normalized() * 3)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    # hits a vert
    points, faces = walk_along_mesh(obj, m, Vector((-0.99, -0.99, 1)), Vector((1, 1, 0)).normalized() * 150)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    # initial heading is not along face
    points, faces = walk_along_mesh(obj, m, Vector((-0.99, -0.99, 1)), Vector((1, .33, 0.1)).normalized() * 3)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    # TODO we could have early stopping when we wrap around, but as in the case here, the starting point is off the edge
    # so just checking the intersection isnt sufficient, need to check the dot with all existing segments
    points, faces = walk_along_mesh(obj, m, Vector((-1, 0, -.99)), Vector((0, 0, 1)).normalized() * 30)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    obj = D.objects['Cube.001']
    m = bmesh.new()
    m.from_mesh(obj.data)
    points, faces = walk_along_mesh(obj, m, Vector((-0.99, -0.99, 1)), Vector((1, .56, 0)).normalized() * 100)
    curve = make_curve(points, handle_type='VECTOR')
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world

    obj = D.objects['Cube.002']
    m = bmesh.new()
    m.from_mesh(obj.data)
    curve = make_empty_curve(handle_type='VECTOR')
    generate_walks(
        curve,
        obj,
        m,
        [f.calc_center_median() for f in m.faces],
        partial(random.randint, 3, 5),
        partial(np.linspace, 0, np.pi * 2, endpoint=False),
        lambda n: [random.uniform(3, 5) for _ in range(n)],
    )
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world
    set_curve_handles(curve, 'VECTOR')

if __name__ == '__dev__':
    # I have a script in a testing blendfile with the following two lines in it to run this script
    # filename = "/path/to/origami.py"
    # exec(compile(open(filename).read(), filename, 'exec'), {'__name__': '__dev__'})
    logging.basicConfig(level=logging.DEBUG)
    dev()
    logger.debug('-' * 80)

# plan for operators
# edit mode - select shortest weighted path between two selected verts (allow selected vertex group)

# shortest path
# select mesh and curve, snap each spline's start and end to mesh and do shortest (weighted) path (allow vertex group selection)

# geodesic path
# select mesh and curve, snap each spline to mesh and then wrap/project the spline onto the mesh

# generators
# shortest path
# select mesh, select n random vert pairs (allow only selected) and draw spline, allow using particles and/or verts of another mesh as the basis for this pool
# options: min length (in terms of steps and/or weight), max length, no duplciate nodes, FUTURE no intersections

# geodesic path
# `generate_walks`
# select mesh, starting points snapped from particles (can include the particles X, Y, or Z as the heading guess) or verts of another mesh or face centers b/c why not
# options nspokes (constant, uniform, gaussian), angles (equi, random), lengths (constant, uniform, gaussian) FUTURE no intersections

# FUTURE detect interesections for either path discard or early stopping or some type of weaving (ie add intersection to each involved spline and shuffle their z height)
