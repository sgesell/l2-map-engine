#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/poly_integrator.hpp"
#include "l2map/basis_builder.hpp"
#include <cmath>

using namespace l2map;

static Polygon2D unit_square() {
    return {
        Point2D(0.0, 0.0),
        Point2D(1.0, 0.0),
        Point2D(1.0, 1.0),
        Point2D(0.0, 1.0)
    };
}

static Polygon2D unit_triangle() {
    // Vertices: (0,0), (1,0), (0,1) — CCW
    return {
        Point2D(0.0, 0.0),
        Point2D(1.0, 0.0),
        Point2D(0.0, 1.0)
    };
}

// -----------------------------------------------------------------------
// Test 1: ∫_[0,1]² 1 dA = 1.0
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: constant 1 over unit square", "[integrator]") {
    PolyIntegrator integ;
    // Constant polynomial: coeffs = [1], mono = {(0,0)}
    MonomialBasis2D mono = get_monomial_basis_2d(1);
    VectorXd coeffs(1);
    coeffs << 1.0;
    double result = integ.integrate(unit_square(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(1.0, 1e-12));
}

// -----------------------------------------------------------------------
// Test 2: ∫_[0,1]² x dA = 0.5
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: integral of x over unit square = 0.5", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono = get_monomial_basis_2d(3);  // {1, x, y}
    VectorXd coeffs = VectorXd::Zero(3);
    coeffs[1] = 1.0;  // coefficient of x
    double result = integ.integrate(unit_square(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.5, 1e-12));
}

// -----------------------------------------------------------------------
// Test 3: ∫_[0,1]² y dA = 0.5
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: integral of y over unit square = 0.5", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono = get_monomial_basis_2d(3);
    VectorXd coeffs = VectorXd::Zero(3);
    coeffs[2] = 1.0;  // coefficient of y
    double result = integ.integrate(unit_square(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.5, 1e-12));
}

// -----------------------------------------------------------------------
// Test 4: ∫_[0,1]² x*y dA = 0.25
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: integral of xy over unit square = 0.25", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono = get_monomial_basis_2d(4);  // {1, x, y, x²} — need xy
    // Actually for xy we need n=5: {1,x,y,x²,xy}
    MonomialBasis2D mono5 = get_monomial_basis_2d(5);
    VectorXd coeffs = VectorXd::Zero(5);
    // Find index of (1,1) = xy
    for (int i = 0; i < 5; ++i) {
        if (mono5.monomials[i] == std::make_pair(1, 1)) {
            coeffs[i] = 1.0;
            break;
        }
    }
    double result = integ.integrate(unit_square(), coeffs, mono5);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.25, 1e-12));
}

// -----------------------------------------------------------------------
// Test 5: ∫_triangle x dA = 1/6   (triangle (0,0),(1,0),(0,1))
// Exact: ∫∫ x dx dy over {x+y ≤ 1, x≥0, y≥0} = 1/6
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: integral of x over unit triangle = 1/6", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono = get_monomial_basis_2d(3);
    VectorXd coeffs = VectorXd::Zero(3);
    coeffs[1] = 1.0;
    double result = integ.integrate(unit_triangle(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(1.0 / 6.0, 1e-12));
}

// -----------------------------------------------------------------------
// Test 6: ∫_triangle 1 dA = 0.5
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: constant 1 over unit triangle = 0.5", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono = get_monomial_basis_2d(1);
    VectorXd coeffs(1);
    coeffs << 1.0;
    double result = integ.integrate(unit_triangle(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.5, 1e-12));
}

// -----------------------------------------------------------------------
// Test 7: Polynomial multiplication (1+x)*(1+y) = 1 + x + y + xy
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: polynomial multiplication (1+x)*(1+y)", "[integrator]") {
    PolyIntegrator integ;
    MonomialBasis2D mono3 = get_monomial_basis_2d(3);  // {1, x, y}

    VectorXd a = VectorXd::Zero(3);
    a[0] = 1.0; a[1] = 1.0;  // 1 + x

    VectorXd b = VectorXd::Zero(3);
    b[0] = 1.0; b[2] = 1.0;  // 1 + y

    MonomialBasis2D mono_out;
    VectorXd prod = integ.multiply_polynomials(a, mono3, b, mono3, mono_out);

    // Expected: 1 + x + y + xy  → coeffs in Pascal order for 5 monomials
    // {1, x, y, x², xy}: coeff[0]=1, coeff[1]=1, coeff[2]=1, coeff[3]=0, coeff[4]=1
    REQUIRE(mono_out.n_monomials >= 5);

    // Find xy index
    int idx_1=0, idx_x=0, idx_y=0, idx_xy=-1;
    for (int i = 0; i < mono_out.n_monomials; ++i) {
        auto [px, py] = mono_out.monomials[i];
        if (px==0 && py==0) idx_1 = i;
        if (px==1 && py==0) idx_x = i;
        if (px==0 && py==1) idx_y = i;
        if (px==1 && py==1) idx_xy = i;
    }
    REQUIRE(idx_xy >= 0);
    CHECK_THAT(prod[idx_1],  Catch::Matchers::WithinAbs(1.0, 1e-14));
    CHECK_THAT(prod[idx_x],  Catch::Matchers::WithinAbs(1.0, 1e-14));
    CHECK_THAT(prod[idx_y],  Catch::Matchers::WithinAbs(1.0, 1e-14));
    CHECK_THAT(prod[idx_xy], Catch::Matchers::WithinAbs(1.0, 1e-14));

    // Verify via integration: ∫_[0,1]² (1+x+y+xy) dA = 1 + 0.5 + 0.5 + 0.25 = 2.25
    double integral = integ.integrate(unit_square(), prod, mono_out);
    CHECK_THAT(integral, Catch::Matchers::WithinAbs(2.25, 1e-12));
}

// -----------------------------------------------------------------------
// Test 8: Higher-degree integration — ∫_[0,1]² x²y² dA = 1/9
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: integral of x2y2 over unit square = 1/9", "[integrator]") {
    PolyIntegrator integ;
    // x²y² needs monomial basis including (2,2)
    // Pascal: degree 4 includes (4,0),(3,1),(2,2),(1,3),(0,4) — 15 total monomials
    MonomialBasis2D mono = get_monomial_basis_2d(15);
    VectorXd coeffs = VectorXd::Zero(15);
    for (int i = 0; i < 15; ++i) {
        if (mono.monomials[i] == std::make_pair(2, 2)) {
            coeffs[i] = 1.0;
            break;
        }
    }
    double result = integ.integrate(unit_square(), coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(1.0 / 9.0, 1e-10));
}

// -----------------------------------------------------------------------
// Test 9: Symmetry — integrating a basis function product matches for (i,j) and (j,i)
// -----------------------------------------------------------------------
TEST_CASE("PolyIntegrator: mass matrix is symmetric", "[integrator]") {
    PolyIntegrator integ;
    Polygon2D sq = unit_square();

    std::vector<Point2D> pts = {
        Point2D(0.1, 0.1), Point2D(0.5, 0.1), Point2D(0.9, 0.1),
        Point2D(0.1, 0.5), Point2D(0.5, 0.5), Point2D(0.9, 0.5),
        Point2D(0.1, 0.9), Point2D(0.5, 0.9), Point2D(0.9, 0.9)
    };
    BasisBuilder bb;
    BasisMatrix B = bb.build(pts);
    MonomialBasis2D mono = get_monomial_basis_2d(9);

    // Compute V[0,1] and V[1,0] — should be equal
    MonomialBasis2D mono_prod;
    VectorXd prod01 = integ.multiply_polynomials(B.row(0), mono, B.row(1), mono, mono_prod);
    VectorXd prod10 = integ.multiply_polynomials(B.row(1), mono, B.row(0), mono, mono_prod);

    // Shift polygon to match (last pt = origin)
    Point2D origin = pts.back();
    Polygon2D sq_sh;
    for (auto& p : sq) sq_sh.push_back(p - origin);

    double v01 = integ.integrate(sq_sh, prod01, mono_prod);
    double v10 = integ.integrate(sq_sh, prod10, mono_prod);
    CHECK_THAT(v01, Catch::Matchers::WithinAbs(v10, 1e-12));
}
