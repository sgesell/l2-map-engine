#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/polygon_clipper.hpp"
#include <cmath>

using namespace l2map;

static Polygon2D make_square(double x0, double y0, double size) {
    // CCW unit square with bottom-left at (x0, y0)
    return {
        Point2D(x0,        y0),
        Point2D(x0 + size, y0),
        Point2D(x0 + size, y0 + size),
        Point2D(x0,        y0 + size)
    };
}

// -----------------------------------------------------------------------
// Test 1: Square-square partial overlap — area = 0.25
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: square-square partial overlap", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D a = make_square(0.0, 0.0, 1.0);  // [0,1]x[0,1]
    Polygon2D b = make_square(0.5, 0.5, 1.0);  // [0.5,1.5]x[0.5,1.5]

    auto result = clipper.intersect(a, b);
    REQUIRE(result.has_value());
    double area = clipper.signed_area(*result);
    CHECK_THAT(area, Catch::Matchers::WithinAbs(0.25, 1e-12));
}

// -----------------------------------------------------------------------
// Test 2: No overlap — two disjoint squares
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: no overlap returns nullopt", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D a = make_square(0.0, 0.0, 1.0);
    Polygon2D b = make_square(2.0, 2.0, 1.0);

    auto result = clipper.intersect(a, b);
    CHECK(!result.has_value());
}

// -----------------------------------------------------------------------
// Test 3: Equal polygons — intersection equals original
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: equal polygons", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D a = make_square(0.0, 0.0, 1.0);

    auto result = clipper.intersect(a, a);
    REQUIRE(result.has_value());
    double area = clipper.signed_area(*result);
    CHECK_THAT(area, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// -----------------------------------------------------------------------
// Test 4: CCW invariant — result must have positive signed area
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: result is always CCW", "[clipper]") {
    PolygonClipper clipper;
    // Try CW input polygon — clipper should handle and return CCW
    Polygon2D a_cw = {
        Point2D(0.0, 1.0),  // CW order (reversed from CCW)
        Point2D(1.0, 1.0),
        Point2D(1.0, 0.0),
        Point2D(0.0, 0.0)
    };
    Polygon2D b = make_square(0.5, 0.5, 1.0);

    auto result = clipper.intersect(a_cw, b);
    REQUIRE(result.has_value());
    double area = clipper.signed_area(*result);
    // Area must be positive (CCW)
    CHECK(area > 0.0);
    CHECK_THAT(area, Catch::Matchers::WithinAbs(0.25, 1e-10));
}

// -----------------------------------------------------------------------
// Test 5: Unit square area
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: signed_area unit square = 1.0", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D sq = make_square(0.0, 0.0, 1.0);
    double area = clipper.signed_area(sq);
    CHECK_THAT(area, Catch::Matchers::WithinAbs(1.0, 1e-14));
}

// -----------------------------------------------------------------------
// Test 6: CW polygon has negative area; ensure_ccw corrects it
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: ensure_ccw", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D cw = {
        Point2D(0.0, 1.0),
        Point2D(1.0, 1.0),
        Point2D(1.0, 0.0),
        Point2D(0.0, 0.0)
    };
    CHECK(clipper.signed_area(cw) < 0.0);
    Polygon2D ccw = clipper.ensure_ccw(cw);
    CHECK(clipper.signed_area(ccw) > 0.0);
    CHECK_THAT(clipper.signed_area(ccw), Catch::Matchers::WithinAbs(1.0, 1e-14));
}

// -----------------------------------------------------------------------
// Test 7: Point in polygon
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: point_in_polygon", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D sq = make_square(0.0, 0.0, 1.0);
    CHECK( clipper.point_in_polygon(Point2D(0.5, 0.5), sq));
    CHECK(!clipper.point_in_polygon(Point2D(1.5, 0.5), sq));
    CHECK(!clipper.point_in_polygon(Point2D(0.5, 1.5), sq));
}

// -----------------------------------------------------------------------
// Test 8: Full containment — smaller square inside larger
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: full containment", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D big   = make_square(0.0, 0.0, 2.0);
    Polygon2D small = make_square(0.5, 0.5, 1.0);

    auto result = clipper.intersect(small, big);
    REQUIRE(result.has_value());
    double area = clipper.signed_area(*result);
    // Intersection should be the smaller square
    CHECK_THAT(area, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// -----------------------------------------------------------------------
// Test 9: Adjacent squares (share only an edge) → nullopt (area < 1e-30)
// -----------------------------------------------------------------------
TEST_CASE("PolygonClipper: adjacent squares return nullopt", "[clipper]") {
    PolygonClipper clipper;
    Polygon2D a = make_square(0.0, 0.0, 1.0);
    Polygon2D b = make_square(1.0, 0.0, 1.0);

    auto result = clipper.intersect(a, b);
    // Shared edge → degenerate intersection → nullopt
    CHECK(!result.has_value());
}
