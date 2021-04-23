# Geodesic Blender Addon

# WIP
TODO

- use modifiers everywhere
- walk along mesh is still buggy. numerical issues maybe? tolerance issues? just wrong? not sure yet

Addon for doing [geodesic](https://en.wikipedia.org/wiki/Geodesic)-like things in Blender like drawing shortest weighted path curves along meshes and drawing curves along a mesh given an initial heading.

These are in a menu under `Object>Geodesic` or by operator search (`F3`) just type `Geodesic` and you'll find them. I still need to put the `Selected Shortest Weighted Path` in the right menu so you can get to it from Edit mode but I'm not sure I'll ever understand the menu systems and operator search works great for me.

## Select Shortest Weighted Path

In edit mode, select two verts and this operator will select the shortest weighted path between them, using the vertex group of your choice to calculate the weights. The weight to move from one vertex to another is the average weight of each vertex times the distance between them.

`Cross Faces` enabled allows the path to traverse across n-gons even if no edge exists between them in the mesh.

Vertices that are not part of the selected vertex group are not included in the search graph, so if you get an error of `No path exists...`, check your verts are in the vertex group. And note that Blender seems to remove verts from a vertex group if you invert the weight to 0. To get around that, set the `Weight` to 0 when you go to assign them.

## Snap Curve To Mesh Shortest Path

Select one mesh and one curve (with potentially multiple splines). For each spline in the curve, we snap the start and end points to their closest point on the mesh. Then, we find the shortest weighted path as above and add points to the spline at those vertices.

`Cross Faces` is as above.

`Closest Vert` snaps the start and end spline to the closest vertex, not just closest point on the mesh.

## Snap Curve To Mesh Walk

Select one mesh and one curve (with potentially multiple splines). For each spline in the curve, we take the vector from its start point to end point as a heading vector. Then, snapping the start to the closest point on the mesh, we walk in that heading direction, bending around the mesh as we encounter edges which connect to another face. The walk continues for the length of the initial heading vector.

## Generate Shortest Paths

Select a mesh with a vertex group. We choose random pairs of vertices and draw a shortest path curve between them.

`Min Length` refers to the number of nodes involved.

`Cross Faces` is as above

This is a best-effort generator, we only try a limited number of times to create paths of a certain length before giving up.

## Generate Walks

Select a mesh. We create a bunch of curves that walk along the surface of the mesh.

`Source` will take starting locations from either face centers or the particle system

`Number of Sources to use` randomly selects a subset of the above sources. 0 means all

`Number of Spokes` each source will have this many walks emanating from it

`Length of Paths` max distance to walk along the mesh

`Spoke angle type` will either be equally spaced radially (like bicycle spokes) or have random angles

## Notes

Walks will terminate if you hit a vertex or an edge that doesn't have a joining face. I haven't tried with crazy geometries.

There are still some cases where things go weird with the walks so if you find good test cases, please open an issue!

Tested with 2.92
