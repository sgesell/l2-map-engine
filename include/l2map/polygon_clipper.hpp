#pragma once
#include "types.hpp"
#include <optional>

namespace l2map {

class PolygonClipper {
public:
    // Compute intersection of polygon_a and polygon_b using Sutherland-Hodgman.
    // Both polygons should be in CCW order.
    // Returns nullopt if intersection is empty or degenerate (area < 1e-30).
    // Result is guaranteed CCW if non-empty.
    std::optional<Polygon2D> intersect(const Polygon2D& polygon_a,
                                       const Polygon2D& polygon_b) const;

    // Signed area (positive = CCW)
    double signed_area(const Polygon2D& poly) const;

    // Check equality within tolerance
    bool polygons_equal(const Polygon2D& a,
                        const Polygon2D& b,
                        double tol = 1e-12) const;

    // Point-in-polygon test (ray casting)
    bool point_in_polygon(const Point2D& p, const Polygon2D& poly) const;

    // Ensure CCW orientation; reverse if CW
    Polygon2D ensure_ccw(const Polygon2D& poly) const;

private:
    // Clip subject polygon against a single half-plane defined by directed edge a→b.
    // "Inside" = left side of directed edge (cross product >= 0).
    Polygon2D clip_by_halfplane_(const Polygon2D& subject,
                                 const Point2D& a,
                                 const Point2D& b) const;

    // Line-line intersection: segment p1→p2 crossing the line through a→b.
    Point2D line_intersection_(const Point2D& p1, const Point2D& p2,
                                const Point2D& a,  const Point2D& b) const;

    // "Inside" test: is P on the left side (or on) the directed edge a→b?
    bool inside_(const Point2D& p, const Point2D& a, const Point2D& b) const;
};

} // namespace l2map
