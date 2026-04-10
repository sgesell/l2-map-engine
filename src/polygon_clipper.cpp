#include "l2map/polygon_clipper.hpp"
#include <cmath>
#include <algorithm>

namespace l2map {

double PolygonClipper::signed_area(const Polygon2D& poly) const {
    double area = 0.0;
    int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const Point2D& a = poly[i];
        const Point2D& b = poly[(i + 1) % n];
        area += a[0] * b[1] - b[0] * a[1];
    }
    return 0.5 * area;
}

Polygon2D PolygonClipper::ensure_ccw(const Polygon2D& poly) const {
    if (signed_area(poly) < 0.0) {
        Polygon2D rev(poly.rbegin(), poly.rend());
        return rev;
    }
    return poly;
}

bool PolygonClipper::inside_(const Point2D& p, const Point2D& a, const Point2D& b) const {
    // Cross product (b-a) × (p-a) >= 0 means p is on the left side or on the line
    double cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    return cross >= 0.0;
}

Point2D PolygonClipper::line_intersection_(const Point2D& p1, const Point2D& p2,
                                            const Point2D& a,  const Point2D& b) const
{
    // Parametric intersection of line through p1,p2 with line through a,b
    double dx1 = p2[0] - p1[0], dy1 = p2[1] - p1[1];
    double dx2 = b[0]  - a[0],  dy2 = b[1]  - a[1];
    double denom = dx1 * dy2 - dy1 * dx2;
    // denom should not be zero if segments are not parallel (guaranteed by Sutherland-Hodgman)
    double t = ((a[0] - p1[0]) * dy2 - (a[1] - p1[1]) * dx2) / denom;
    return Point2D(p1[0] + t * dx1, p1[1] + t * dy1);
}

Polygon2D PolygonClipper::clip_by_halfplane_(const Polygon2D& subject,
                                              const Point2D& a,
                                              const Point2D& b) const
{
    Polygon2D output;
    if (subject.empty()) return output;

    int n = static_cast<int>(subject.size());
    for (int i = 0; i < n; ++i) {
        const Point2D& P = subject[i];
        const Point2D& Q = subject[(i + n - 1) % n]; // previous point

        bool P_inside = inside_(P, a, b);
        bool Q_inside = inside_(Q, a, b);

        if (P_inside) {
            if (!Q_inside)
                output.push_back(line_intersection_(Q, P, a, b));
            output.push_back(P);
        } else if (Q_inside) {
            output.push_back(line_intersection_(Q, P, a, b));
        }
    }
    return output;
}

std::optional<Polygon2D> PolygonClipper::intersect(const Polygon2D& polygon_a,
                                                     const Polygon2D& polygon_b) const
{
    if (polygon_a.empty() || polygon_b.empty()) return std::nullopt;

    // Ensure clip polygon (b) is CCW for correct half-plane test
    Polygon2D clip = ensure_ccw(polygon_b);

    Polygon2D output = polygon_a;  // subject
    int m = static_cast<int>(clip.size());

    for (int i = 0; i < m; ++i) {
        if (output.empty()) return std::nullopt;
        const Point2D& a = clip[i];
        const Point2D& b = clip[(i + 1) % m];
        output = clip_by_halfplane_(output, a, b);
    }

    if (output.empty()) return std::nullopt;

    // Ensure result is CCW
    output = ensure_ccw(output);

    // Reject degenerate intersections (point/line only)
    if (std::abs(signed_area(output)) < 1e-30) return std::nullopt;

    return output;
}

bool PolygonClipper::polygons_equal(const Polygon2D& a,
                                     const Polygon2D& b,
                                     double tol) const
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] - b[i]).norm() > tol) return false;
    }
    return true;
}

bool PolygonClipper::point_in_polygon(const Point2D& p, const Polygon2D& poly) const {
    // Ray casting from p in +x direction
    int n = static_cast<int>(poly.size());
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly[i][0], yi = poly[i][1];
        double xj = poly[j][0], yj = poly[j][1];
        if (((yi > p[1]) != (yj > p[1])) &&
            (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi))
            inside = !inside;
    }
    return inside;
}

} // namespace l2map
