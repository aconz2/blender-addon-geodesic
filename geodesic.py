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

# import sys
# os.system(f'{sys.executable} -m ensurepip')
# os.system(f'{sys.executable} -m pip install networkx')

logger = logging.getLogger('geodesic')

TOL = 1e-4
VERT_TOL = 1e-2

AXES = {
    'X': Vector((1, 0, 0)),
    'Y': Vector((0, 1, 0)),
    'Z': Vector((0, 0, 1)),
}

class RandomPairsWithReplacement:
    def __init__(self, xs):
        self.xs = xs
        self.i = 0
        random.shuffle(self.xs)

    def commit(self): pass
    def reject(self): pass

    def draw(self):
        if len(self.xs) - self.i < 2:
            self.i = 0
            random.shuffle(self.xs)

        a = self.xs[self.i]
        b = self.xs[self.i + 1]

        self.i += 2

        return a, b

class RandomPairsWithoutReplacement:
    def __init__(self, xs):
        # idk is it better to use a single deque and a marker of when the first element comes back around?
        self.primary = xs
        self.secondary = []
        self.i = 0

        self.a = None
        self.b = None
        self.done = False

        self.reload()

    def reload(self):
        self.primary.extend(self.secondary)
        random.shuffle(self.primary)
        self.secondary.clear()

        if len(self.primary) < 2:
            self.done = True

    def draw(self):
        if self.done:
            return None

        if len(self.primary) < 2:
            self.reload()
            if self.done:
                return None

        self.a = self.primary.pop()
        self.b = self.primary.pop()

        return self.a, self.b

    def commit(self):
        assert self.a is not None and self.b is not None
        self.a = None
        self.b = None

    def reject(self):
        assert self.a is not None and self.b is not None
        self.secondary.append(self.a)
        self.secondary.append(self.b)
        self.a = None
        self.b = None

def const(n):
    return n

def const_n(x, n):
    return [x] * n

def rotated(v, rot):
    v.rotate(rot)
    return v

def uniform_n(low, hi, n):
    return [random.uniform(low, hi) for _ in range(n)]

def vector_rejection(a, b):
    return a - a.project(b)

def one_mesh_one_curve(objects):
    if len(objects) != 2:
        return None
    a, b = objects
    if a.type == 'MESH' and b.type == 'CURVE':
        return a, b
    elif b.type == 'MESH' and a.type == 'CURVE':
        return b, a
    else:
        return None

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

def build_graph(mesh, vertex_group=None, min_weight=0.1, cross_faces=False):
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

def remove_path(G, nodes):
    for i in range(len(nodes) - 1):
        G.remove_edge(nodes[i], nodes[i + 1])

def make_empty_curve(name='Curve'):
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
    curve = make_empty_curve(name=name)
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

def snap_curve_splines_shortest_path(G, obj, mesh, curve, vertex_group=None, cross_faces=False, closest_vert=True):
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

# options will be added that make it hard to tell without exhaustive checks whether
# paths that fit criteria will be found in reasonable time. maxtries_multiplier bounds our efforts
def generate_multiple_paths(G, n, maxtries_multiplier=10, with_replacement=True, min_length=2):
    ret = []

    pairs = (RandomPairsWithReplacement if with_replacement else RandomPairsWithoutReplacement)(list(G))
    for _ in range(n * maxtries_multiplier):
        pair = pairs.draw()
        if pair is None:
            break
        a, b = pair
        # TODO future things might reject this path

        path = try_shortest_path(G, a, b)
        # TODO if this is frequently caused by disconnected components, it would be smarter to partition
        # the pairs up front and only try pairs with a connected component
        if path is None or len(path) < min_length:
            pairs.reject()
            continue

        ret.append(get_path_points(G, path))
        pairs.commit()

        if len(ret) == n:
            break

    return ret

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
            # logger.debug('NEWFACE IS NONE')
            faces.append(face.index)
            assert len(points) - 1 == len(faces)
            return points, faces

        assert (heading.length) >= (intersection - a).length

        heading -= (intersection - a)  # subtract off the amount we have
        heading = make_face_face_rotation_matrix(face, new_face, v2 - v1) @ heading

        faces.append(face.index)
        face = new_face

    assert len(points) - 1 == len(faces)
    return points, faces
    assert False

def snap_curve_splines_walk(obj, mesh, curve):
    remove = []

    splines = list(curve.data.splines)
    for spline in splines:
        points = spline.bezier_points if spline.type == 'BEZIER' else spline.points

        if len(points) < 2:
            continue
        start = points[0].co
        end = points[-1].co

        points, faces = walk_along_mesh(obj, mesh, start, end - start)

        make_spline(curve, points, type=spline.type)
        remove.append(spline)

    for x in remove:
        curve.data.splines.remove(x)

def generate_walks(obj, mesh, curve, starts, gen_n_spokes, gen_angles, gen_lengths):
    mesh.faces.ensure_lookup_table()

    for i, start in enumerate(starts):
        spokes = gen_n_spokes()
        angles = gen_angles(spokes)
        lengths = gen_lengths(spokes)

        if isinstance(start, tuple):
            start, heading = start
            heading.normalize()
            loc, normal, face_index = closest_point_on_mesh(obj, start)
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
    snap_curve_splines_shortest_path(G, obj, m, bc, closest_vert=False)
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
    curve = make_empty_curve()
    generate_walks(
        obj,
        m,
        curve,
        [f.calc_center_median() for f in m.faces],
        partial(random.randint, 3, 5),
        partial(np.linspace, 0, np.pi * 2, endpoint=False),
        lambda n: [random.uniform(3, 5) for _ in range(n)],
    )
    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world
    set_curve_handles(curve, 'VECTOR')

    obj = D.objects['Cube.Particles']
    m = bmesh.new()
    m.from_mesh(obj.data)
    depsg = C.evaluated_depsgraph_get()
    particles = obj.evaluated_get(depsg).particle_systems[0].particles
    curve = make_empty_curve('Curve.Particles')
    mat = obj.matrix_world.copy()
    mat.invert()

    # for p in particles[:10]:
    #     v = rotated(Vector((0, 1, 0)), p.rotation)
    #     p1 = p.location
    #     p2 = p1 + v * 2
    #     make_spline(curve, [mat @ p1, mat @ p2])
    # the mat stuff is to bring the particle location into object local space
    generate_walks(
        obj,
        m,
        curve,
        [(mat @ p.location, rotated(Vector((0, 1, 0)), p.rotation)) for p in particles],
        partial(const, 3),
        partial(np.linspace, 0, np.pi * 2, endpoint=False),
        partial(uniform_n, 1, 2),
    )

    curve.data.bevel_depth = 0.01
    curve.matrix_world = obj.matrix_world
    set_curve_handles(curve, 'VECTOR')

class GeodesicWeightedShortestPath(bpy.types.Operator):
    """Select shortest path between two vertices on a mesh using vertex weights"""

    bl_idname = 'mesh.geodesic_select_shortest_weighted_path'
    bl_label = 'Geodesic Select Shortest Weighted Path'
    bl_options = {'REGISTER', 'UNDO'}

    vertex_group: bpy.props.StringProperty(name='Vertex Group', default='')

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        obj = context.object
        self.layout.prop_search(self, 'vertex_group', obj, 'vertex_groups', text='Vertex Group')

    def execute(self, context):
        obj = context.object
        if len(obj.vertex_groups) == 0:
            self.report({'WARNING'}, 'This mesh has no vertex groups, use the builtin select shortest path')
            return {'CANCELLED'}

        if obj.data.total_vert_sel != 2:
            self.report({'WARNING'}, f'Select only 2 vertices, got {obj.data.total_vert_sel}')
            return {'CANCELLED'}

        if self.vertex_group == '':
            self.vertex_group = obj.vertex_groups[obj.vertex_groups.active_index].name

        m = bmesh.from_edit_mesh(obj.data)

        selected_verts = [x for x in m.verts if x.select]
        assert len(selected_verts) == 2

        G, verts = build_graph(m, vertex_group=obj.vertex_groups[self.vertex_group])
        path = try_shortest_path(G, selected_verts[0].index, selected_verts[1].index)

        if path is None:
            self.report({'WARNING'}, f'No path exists between the selected vertices {selected_verts[0].index} {selected_verts[1].index}')
            # we use FINISHED here to allow selecting another vertex group that might have a path
            return {'FINISHED'}

        m.verts.ensure_lookup_table()
        for p in path:
            m.verts[p].select = True

        bmesh.update_edit_mesh(obj.data, False, False)

        return {'FINISHED'}

class GeodesicSnapCurveToMeshShortestPath(bpy.types.Operator):
    """Snap each spline's in a curve to a mesh's face by optionally weighted shortest path"""

    bl_idname = 'mesh.geodesic_snap_curve_to_mesh_shortest_path'
    bl_label = 'Geodesic Snap Curve to Mesh Shortest Path'
    bl_options = {'REGISTER', 'UNDO'}

    vertex_group: bpy.props.StringProperty(name='Vertex Group', default='')
    cross_faces: bpy.props.BoolProperty(
        name='Cross Faces',
        default=False,
        description='Allow crossing faces in n-gons even if no edge connects the verts',
    )
    closest_vert: bpy.props.BoolProperty(
        name='Closest Vert',
        default=False,
        description='Snap the start and end to the nearest vert',
    )

    @classmethod
    def poll(cls, context):
        return one_mesh_one_curve(context.selected_objects) is not None

    def draw(self, context):
        self.layout.prop_search(self, 'vertex_group', context.object, 'vertex_groups', text='Vertex Group')
        self.layout.row().prop(self, 'cross_faces')
        self.layout.row().prop(self, 'closest_vert')

    def execute(self, context):
        mesh_curve = one_mesh_one_curve(context.selected_objects)
        if mesh_curve is None:
            self.report({'ERROR'}, 'You need to select one mesh and one curve object')
            return {'CANCELLED'}

        obj, curve = mesh_curve
        m = bmesh.new()
        m.from_mesh(obj.data)

        vertex_group = None if self.vertex_group == '' else obj.vertex_groups[self.vertex_group]

        G, verts = build_graph(m, vertex_group=vertex_group, cross_faces=self.cross_faces)

        snap_curve_splines_shortest_path(G, obj, m, curve, closest_vert=self.closest_vert)

        curve.matrix_world = obj.matrix_world

        return {'FINISHED'}

class GeodesicSnapCurveToMeshWalk(bpy.types.Operator):
    """Snap each spline's in a curve to the surface of mesh, using the splines start and endpoint as the heading"""

    bl_idname = 'mesh.geodesic_snap_curve_to_mesh_walk'
    bl_label = 'Geodesic Snap Curve to Mesh Walk'
    bl_options = {'REGISTER', 'UNDO'}

    handle_type: bpy.props.EnumProperty(
        name='Handle Type',
        items=[
            ('VECTOR', 'Vector', 'Vector'),
            ('AUTO', 'Auto', 'Auto'),
        ],
    )

    @classmethod
    def poll(cls, context):
        return one_mesh_one_curve(context.selected_objects) is not None

    def execute(self, context):
        mesh_curve = one_mesh_one_curve(context.selected_objects)
        if mesh_curve is None:
            self.report({'ERROR'}, 'You need to select one mesh and one curve object')
            return {'CANCELLED'}

        obj, curve = mesh_curve
        m = bmesh.new()
        m.from_mesh(obj.data)

        snap_curve_splines_walk(obj, m, curve)
        set_curve_handles(curve, self.handle_type)

        curve.matrix_world = obj.matrix_world

        return {'FINISHED'}

class GeodesicGenerateShortestPaths(bpy.types.Operator):
    """Generate shortest paths between random vertex pairs"""

    bl_idname = 'mesh.geodesic_generate_shortest_paths'
    bl_label = 'Geodesic Generate Shortest Paths'
    bl_options = {'REGISTER', 'UNDO'}

    n_paths: bpy.props.IntProperty(name='Number of Paths', min=1, default=1)
    with_replacement: bpy.props.BoolProperty(name='With Replacement', description='Re-use vertices if true', default=True)
    vertex_group: bpy.props.StringProperty(name='Vertex Group', default='')
    cross_faces: bpy.props.BoolProperty(
        name='Cross Faces',
        default=False,
        description='Allow crossing faces in n-gons even if no edge connects the verts',
    )
    handle_type: bpy.props.EnumProperty(
        name='Handle Type',
        items=[
            ('VECTOR', 'Vector', 'Vector'),
            ('AUTO', 'Auto', 'Auto'),
        ],
    )
    bevel_depth: bpy.props.FloatProperty(name='Bevel Depth', default=0, min=0, precision=3, step=1)
    seed: bpy.props.IntProperty(name='Seed', default=0)
    min_length: bpy.props.IntProperty(name='Min Length', default=2, description='Don\'t accept paths with fewer than this many vertices')

    def draw(self, context):
        self.layout.prop(self, 'n_paths')
        self.layout.prop_search(self, 'vertex_group', context.object, 'vertex_groups', text='Vertex Group')
        self.layout.prop(self, 'with_replacement')
        self.layout.prop(self, 'cross_faces')
        self.layout.prop(self, 'min_length')
        self.layout.prop(self, 'handle_type')
        self.layout.prop(self, 'bevel_depth')
        self.layout.prop(self, 'seed')

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def execute(self, context):
        random.seed(self.seed)

        obj = context.object
        m = bmesh.new()
        m.from_mesh(obj.data)

        vertex_group = None if self.vertex_group == '' else obj.vertex_groups[self.vertex_group]
        G, verts = build_graph(m, vertex_group=vertex_group, cross_faces=self.cross_faces)

        curve = make_empty_curve()
        pointss = generate_multiple_paths(G, self.n_paths, with_replacement=self.with_replacement, min_length=self.min_length)
        for points in pointss:
            make_spline(curve, points, type='BEZIER', handle_type=self.handle_type)

        if len(pointss) < self.n_paths:
            self.report({'WARNING'}, f'Only generated {len(pointss)} curves')

        curve.matrix_world = obj.matrix_world
        curve.data.bevel_depth = self.bevel_depth

        return {'FINISHED'}

class GeodesicGenerateWalks(bpy.types.Operator):
    """Generate walks on the surface of a mesh"""

    bl_idname = 'mesh.geodesic_generate_walks'
    bl_label = 'Geodesic Generate Walks'
    bl_options = {'REGISTER', 'UNDO'}

    n_spokes: bpy.props.IntProperty(name='Number of Paths', min=1, default=1)
    subset: bpy.props.IntProperty(name='Number of Sources to use', min=0, default=0, description='Only use this many sources (0 for all)')

    source: bpy.props.EnumProperty(
        name='Source',
        items=[
            ('FACE_CENTERS', 'Face Centers', 'Face Centers'),
            ('PARTICLES', 'Particles', 'Particles'),
        ],
    )
    particle_system: bpy.props.StringProperty(name='Particle System', default='')
    particle_axis: bpy.props.EnumProperty(
        name='Particle Axis',
        items=[
            ('X', 'X', 'X'),
            ('Y', 'Y', 'Y'),
            ('Z', 'Z', 'Z'),
        ]
    )

    path_length_type: bpy.props.EnumProperty(
        name='Path Length Type',
        items=[
            ('CONSTANT', 'Constant', 'Constant'),
            ('RANDOM_UNIFORM', 'Uniform Random', 'Uniform Random'),
        ]
    )
    path_length: bpy.props.FloatProperty(name='Length of Paths', min=0.001, default=1)
    path_length_random_uniform_min: bpy.props.FloatProperty(name='Uniform Random Min', min=0.001, default=1)
    path_length_random_uniform_max: bpy.props.FloatProperty(name='Uniform Random Max', min=0.001, default=1)

    seed: bpy.props.IntProperty(name='Seed', default=0)
    handle_type: bpy.props.EnumProperty(
        name='Handle Type',
        items=[
            ('VECTOR', 'Vector', 'Vector'),
            ('AUTO', 'Auto', 'Auto'),
        ],
    )
    bevel_depth: bpy.props.FloatProperty(name='Bevel Depth', default=0, min=0, precision=3, step=1)

    def draw(self, context):
        self.layout.prop(self, 'source')
        if self.source == 'PARTICLES':
            self.layout.prop_search(self, 'particle_system', context.object, 'particle_systems', text='Particle System')
            self.layout.prop(self, 'particle_axis', expand=True)

        self.layout.prop(self, 'subset')
        self.layout.prop(self, 'n_spokes')

        self.layout.prop(self, 'path_length_type', expand=True)
        if self.path_length_type == 'CONSTANT':
            self.layout.prop(self, 'path_length')
        else:
            self.layout.prop(self, 'path_length_random_uniform_min')
            self.layout.prop(self, 'path_length_random_uniform_max')

        self.layout.prop(self, 'handle_type')
        self.layout.prop(self, 'bevel_depth')
        self.layout.prop(self, 'seed')

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def execute(self, context):
        random.seed(self.seed)
        obj = context.object

        if self.source == 'PARTICLES':
            if len(obj.particle_systems) == 0:
                self.report({'ERROR'}, 'Object has no particle system')
                self.source = 'FACE_CENTERS'
            elif self.particle_system == '':
                self.particle_system = obj.particle_systems[obj.particle_systems.active_index].name

        m = bmesh.new()
        m.from_mesh(obj.data)
        curve = make_empty_curve()

        if self.source == 'FACE_CENTERS':
            source = [f.calc_center_median() for f in m.faces]

        else:
            depsg = context.evaluated_depsgraph_get()
            particles = obj.evaluated_get(depsg).particle_systems[0].particles
            mat = obj.matrix_world.copy()
            mat.invert()
            axis = AXES[self.particle_axis]
            source = [(mat @ p.location, rotated(axis, p.rotation)) for p in particles]

        if self.subset > 0:
            random.shuffle(source)
            source = source[:self.subset]

        if self.path_length_type == 'CONSTANT':
            lengths = partial(const_n, self.path_length)
        else:
            lengths = partial(uniform_n, self.path_length_random_uniform_min, self.path_length_random_uniform_max)

        # TODO - n_spokes should have constand and uniform options
        #      - angle of spokes should have equal and random with start/end options
        #      - investigate the bouncing/rotating of initial heading

        generate_walks(
            obj=obj,
            mesh=m,
            curve=curve,
            starts=source,
            gen_n_spokes=partial(const, self.n_spokes),
            gen_angles=partial(np.linspace, 0, np.pi * 2, endpoint=False),
            gen_lengths=lengths,
        )

        curve.matrix_world = obj.matrix_world
        curve.data.bevel_depth = self.bevel_depth

        set_curve_handles(curve, self.handle_type)

        return {'FINISHED'}

classes = [
    GeodesicWeightedShortestPath,
    GeodesicSnapCurveToMeshShortestPath,
    GeodesicSnapCurveToMeshWalk,
    GeodesicGenerateShortestPaths,
    GeodesicGenerateWalks,
]

# TODO figure out the right place for all the menu items
# class GeodesicMenu(bpy.types.Menu):
#     bl_label = 'Geodesic'
#     bl_idname = 'OBJECT_MT_geodesic'

#     def draw(self, context):
#         for klass in classes:
#             self.layout.operator(klass.bl_idname)

# def menu_func(self, context):
#     self.layout.menu(GeodesicMenu.bl_idname)

def register():
    # bpy.utils.register_class(GeodesicMenu)
    for klass in classes:
        bpy.utils.register_class(klass)
    # bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    # bpy.utils.unregister_class(GeodesicMenu)
    for klass in classes:
        bpy.utils.unregister_class(klass)
    # bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == '__dev__':
    # I have a script in a testing blendfile with the following two lines in it to run this script
    # filename = "/path/to/origami.py"
    # exec(compile(open(filename).read(), filename, 'exec'), {'__name__': '__dev__'})

    try:
        unregister()
    except Exception:
        pass

    register()

    logging.basicConfig(level=logging.DEBUG)
    # dev()
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
