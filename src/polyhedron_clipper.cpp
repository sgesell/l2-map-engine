#include "l2map/polyhedron_clipper.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace l2map {

// ---------------------------------------------------------------------------
// hex8_to_polyhedron
// ---------------------------------------------------------------------------

Polyhedron hex8_to_polyhedron(const MatrixXd& nc) {
    // nc: (8 x 3)
    auto v = [&](int i) -> Point3D { return nc.row(i); };

    // 6 faces with vertices in CCW order viewed from OUTSIDE (verified with
    // cross products in development — each gives the correct outward normal).
    Polyhedron p;
    p.faces = {
        {v(0), v(3), v(2), v(1)},  // bottom  outward normal: -z
        {v(4), v(5), v(6), v(7)},  // top     outward normal: +z
        {v(0), v(1), v(5), v(4)},  // front   outward normal: -y
        {v(2), v(3), v(7), v(6)},  // back    outward normal: +y
        {v(0), v(4), v(7), v(3)},  // left    outward normal: -x
        {v(1), v(2), v(6), v(5)},  // right   outward normal: +x
    };
    return p;
}

// ---------------------------------------------------------------------------
// PolyhedronClipper helpers
// ---------------------------------------------------------------------------

double PolyhedronClipper::edge_plane_t_(const Point3D& A, const Point3D& B,
                                        const Point3D& n, double offset) const
{
    double dA = n.dot(A) - offset;
    double dB = n.dot(B) - offset;
    double denom = dA - dB;
    if (std::abs(denom) < kEps) return 0.5;
    return dA / denom;
}

// Clip one face polygon against n·x ≤ offset.
// Uses 3D Sutherland-Hodgman on the face polygon.
Face3D PolyhedronClipper::clip_face_(const Face3D& face,
                                     const Point3D& n, double offset) const
{
    Face3D out;
    int m = static_cast<int>(face.size());
    if (m == 0) return out;

    for (int i = 0; i < m; ++i) {
        const Point3D& A = face[(i + m - 1) % m];   // previous
        const Point3D& B = face[i];                  // current

        double dA = n.dot(A) - offset;
        double dB = n.dot(B) - offset;

        bool A_in = (dA <= kEps);
        bool B_in = (dB <= kEps);

        if (B_in) {
            if (!A_in) {
                // A outside, B inside: add intersection then B
                double t = edge_plane_t_(A, B, n, offset);
                out.push_back(A + t * (B - A));
            }
            out.push_back(B);
        } else if (A_in) {
            // A inside, B outside: add intersection
            double t = edge_plane_t_(A, B, n, offset);
            out.push_back(A + t * (B - A));
        }
        // both outside: add nothing
    }
    return out;
}

// Sort cut vertices CCW when viewed from outward_n.
Face3D PolyhedronClipper::sort_cut_face_(std::vector<Point3D> verts,
                                          const Point3D& outward_n) const
{
    if (verts.size() < 3) return {};

    // Centroid
    Point3D c = Point3D::Zero();
    for (const auto& v : verts) c += v;
    c /= static_cast<double>(verts.size());

    // Reference direction on the cutting plane (perpendicular to outward_n)
    // Pick the axis furthest from outward_n to avoid degeneracy
    Point3D e_x = Point3D::UnitX();
    Point3D e_y = Point3D::UnitY();
    Point3D e_z = Point3D::UnitZ();
    Point3D ref;
    double cx = std::abs(outward_n.dot(e_x));
    double cy = std::abs(outward_n.dot(e_y));
    double cz = std::abs(outward_n.dot(e_z));
    if (cx <= cy && cx <= cz)      ref = e_x;
    else if (cy <= cx && cy <= cz) ref = e_y;
    else                           ref = e_z;

    // Local axes on the plane: eu, ev = outward_n × eu (both perpendicular to outward_n)
    Point3D eu = (ref - ref.dot(outward_n) * outward_n).normalized();
    Point3D ev = outward_n.cross(eu);

    // Compute angle for each vertex
    std::vector<std::pair<double, Point3D>> angle_verts;
    angle_verts.reserve(verts.size());
    for (const auto& v : verts) {
        Point3D d = v - c;
        double angle = std::atan2(d.dot(ev), d.dot(eu));
        angle_verts.push_back({angle, v});
    }

    // Sort by angle → CCW viewed from outward_n
    std::sort(angle_verts.begin(), angle_verts.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    Face3D result;
    result.reserve(verts.size());
    for (const auto& av : angle_verts)
        result.push_back(av.second);

    return result;
}

// ---------------------------------------------------------------------------
// clip_by_halfspace: 3D Sutherland-Hodgman pass
// ---------------------------------------------------------------------------

std::optional<Polyhedron> PolyhedronClipper::clip_by_halfspace(
    const Polyhedron& poly,
    const Point3D& n, double offset) const
{
    std::vector<Face3D> new_faces;
    std::vector<Point3D> cut_verts;  // intersection points accumulated per face

    for (const Face3D& face : poly.faces) {
        Face3D clipped = clip_face_(face, n, offset);
        if (clipped.size() >= 3)
            new_faces.push_back(clipped);

        // Collect NEW intersection vertices (on the cutting plane).
        // A vertex is genuinely new only if the face had at least one vertex strictly
        // outside (d > kEps). Faces entirely on the plane or entirely inside produce no
        // new boundary — their on-plane vertices are pre-existing and must NOT be added
        // as cut vertices (doing so duplicates the coinciding face).
        bool any_strictly_outside = false;
        for (const auto& v : face)
            if (n.dot(v) - offset > kEps) { any_strictly_outside = true; break; }

        if (any_strictly_outside) {
            for (const Point3D& v : clipped) {
                double d = n.dot(v) - offset;
                if (std::abs(d) < 1e-10) {
                    bool dup = false;
                    for (const auto& cv : cut_verts)
                        if ((v - cv).norm() < 1e-10) { dup = true; break; }
                    if (!dup) cut_verts.push_back(v);
                }
            }
        }
    }

    // Add the new face at the cutting plane (if the polyhedron was clipped)
    if (cut_verts.size() >= 3) {
        Face3D cut_face = sort_cut_face_(cut_verts, n);
        if (cut_face.size() >= 3)
            new_faces.push_back(std::move(cut_face));
    }

    if (new_faces.empty()) return std::nullopt;

    // Remove degenerate faces
    Polyhedron result;
    for (auto& f : new_faces)
        if (f.size() >= 3) result.faces.push_back(std::move(f));

    if (result.faces.empty()) return std::nullopt;
    return result;
}

// ---------------------------------------------------------------------------
// intersect: clip subject by all faces of clip_poly
// ---------------------------------------------------------------------------

std::optional<Polyhedron> PolyhedronClipper::intersect(
    const Polyhedron& subject,
    const Polyhedron& clip_poly) const
{
    Polyhedron current = subject;

    for (const Face3D& clip_face : clip_poly.faces) {
        if (clip_face.size() < 3) continue;

        // Compute outward normal of this clip face and plane offset
        Point3D e1 = clip_face[1] - clip_face[0];
        Point3D e2 = clip_face[2] - clip_face[1];
        Point3D n  = e1.cross(e2);
        double len = n.norm();
        if (len < 1e-14) continue;
        n /= len;  // outward unit normal

        double offset = n.dot(clip_face[0]);  // n·x = offset on the plane

        // Keep the half-space n·x ≤ offset (inside the clip hex)
        auto clipped = clip_by_halfspace(current, n, offset);
        if (!clipped) return std::nullopt;
        current = std::move(*clipped);
    }

    return current;
}

// ---------------------------------------------------------------------------
// signed_volume: divergence theorem — ∫_P dV = (1/3) Σ_faces ∫_face (n·x) dA
//                                            = (1/3) Σ_faces (n·v0) * face_area
// For a convex polyhedron, triangulate each face.
// ---------------------------------------------------------------------------

double PolyhedronClipper::signed_volume(const Polyhedron& poly) const {
    // ∫_P 1 dV = (1/3) Σ_faces (n_f · v_0f) * area_f
    // (using homogeneous formula with d=0: 1/(0+3) = 1/3)
    // For a triangulated face (fan from v0):
    //   area contribution = 0.5 * |(v_i - v_0) × (v_{i+1} - v_0)| · n
    // Combined: volume += (1/6) * (v0 · (v_i - v0) × (v_{i+1} - v0))
    double vol = 0.0;
    for (const Face3D& f : poly.faces) {
        int m = static_cast<int>(f.size());
        for (int i = 1; i < m - 1; ++i) {
            // Triangle: f[0], f[i], f[i+1]
            Point3D v0 = f[0], v1 = f[i], v2 = f[i+1];
            // Signed volume of tetrahedron with origin: (1/6) v0·(v1×v2)
            vol += v0.dot(v1.cross(v2));
        }
    }
    return vol / 6.0;
}

Point3D PolyhedronClipper::centroid(const Polyhedron& poly) const {
    // Centroid via tetrahedra with origin
    Point3D c = Point3D::Zero();
    double vol = 0.0;
    for (const Face3D& f : poly.faces) {
        int m = static_cast<int>(f.size());
        for (int i = 1; i < m - 1; ++i) {
            Point3D v0 = f[0], v1 = f[i], v2 = f[i+1];
            double tet_vol = v0.dot(v1.cross(v2));
            c += tet_vol * (v0 + v1 + v2);
            vol += tet_vol;
        }
    }
    if (std::abs(vol) < 1e-30) return Point3D::Zero();
    return c / (4.0 * vol);  // centroid = (1/4) * sum(tet_centroid * tet_vol) / total_vol
}

} // namespace l2map
