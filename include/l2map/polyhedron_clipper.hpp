#pragma once
#include "types.hpp"
#include <vector>
#include <optional>

namespace l2map {

// A face of a convex polyhedron: vertices in CCW order when viewed from OUTSIDE.
using Face3D = std::vector<Point3D>;

// A convex polyhedron: list of faces.
struct Polyhedron {
    std::vector<Face3D> faces;
    bool empty() const { return faces.empty(); }
};

// Build a Polyhedron from the 8 node coordinates of a Hex8 element.
// node_coords: (8 x 3), Abaqus ordering:
//   Bottom (z-): N0(-,-,-), N1(+,-,-), N2(+,+,-), N3(-,+,-)
//   Top    (z+): N4(-,-,+), N5(+,-,+), N6(+,+,+), N7(-,+,+)
Polyhedron hex8_to_polyhedron(const MatrixXd& node_coords);

// ---------------------------------------------------------------------------
// PolyhedronClipper: 3D Sutherland-Hodgman on convex polyhedra.
// ---------------------------------------------------------------------------
class PolyhedronClipper {
public:
    // Clip a convex polyhedron by a half-space  n·x ≤ offset  (inside = kept).
    // Returns nullopt if nothing remains.
    std::optional<Polyhedron> clip_by_halfspace(
        const Polyhedron& poly,
        const Point3D& n,      // outward normal of clipping plane (unit vector)
        double offset) const;  // n·x = offset defines the plane; keep n·x ≤ offset

    // Intersect two convex polyhedra.
    // Clips subject by all 6 half-spaces defined by the faces of clip_poly.
    std::optional<Polyhedron> intersect(
        const Polyhedron& subject,
        const Polyhedron& clip_poly) const;

    // Signed volume of a polyhedron (positive if faces are CCW from outside).
    double signed_volume(const Polyhedron& poly) const;

    // Centroid of a polyhedron.
    Point3D centroid(const Polyhedron& poly) const;

private:
    static constexpr double kEps = 1e-14;

    // Clip one planar face polygon against the half-space n·x ≤ offset.
    // Returns the clipped face (may be empty).
    Face3D clip_face_(const Face3D& face,
                      const Point3D& n, double offset) const;

    // Sort cut_vertices CCW when viewed from direction outward_n.
    Face3D sort_cut_face_(std::vector<Point3D> cut_verts,
                          const Point3D& outward_n) const;

    // Intersect edge AB with plane n·x = offset; return parameter t in [0,1].
    double edge_plane_t_(const Point3D& A, const Point3D& B,
                         const Point3D& n, double offset) const;
};

} // namespace l2map
