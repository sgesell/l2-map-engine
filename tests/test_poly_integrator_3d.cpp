#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "l2map/poly_integrator_3d.hpp"
#include "l2map/basis_builder_3d.hpp"
#include "l2map/polyhedron_clipper.hpp"
#include <cmath>

using namespace l2map;

// =============================================================================
// Helpers
// =============================================================================

// Unit cube [0,1]^3 as a hex8 polyhedron with correct outward normals.
static Polyhedron unit_cube()
{
    MatrixXd nc(8, 3);
    nc << 0,0,0,  1,0,0,  1,1,0,  0,1,0,
          0,0,1,  1,0,1,  1,1,1,  0,1,1;
    return hex8_to_polyhedron(nc);
}

// Find the index of monomial (px,py,pz) in a MonomialBasis3D; return -1 if absent.
static int find_mono(const MonomialBasis3D& mono, int px, int py, int pz)
{
    for (int i = 0; i < mono.n_monomials; ++i) {
        auto [a, b, c] = mono.monomials[i];
        if (a == px && b == py && c == pz) return i;
    }
    return -1;
}

// =============================================================================
// Integration over the unit cube
// =============================================================================

// ∫_[0,1]^3 1 dV = 1
TEST_CASE("PolyIntegrator3D: integral of 1 over unit cube", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    // Monomial basis {1}
    MonomialBasis3D mono = get_tensor_basis_3d(1);
    VectorXd coeffs(1); coeffs << 1.0;

    double result = integ.integrate(cube, coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(1.0, 1e-12));
}

// ∫_[0,1]^3 x dV = 0.5
TEST_CASE("PolyIntegrator3D: integral of x over unit cube", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    // Pascal tetrahedron for 4 monomials = {1, x, y, z}
    MonomialBasis3D mono = get_tensor_basis_3d(4);
    VectorXd coeffs = VectorXd::Zero(4);
    int idx_x = find_mono(mono, 1, 0, 0);
    REQUIRE(idx_x >= 0);
    coeffs[idx_x] = 1.0;

    double result = integ.integrate(cube, coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.5, 1e-12));
}

// ∫_[0,1]^3 xyz dV = 0.125
TEST_CASE("PolyIntegrator3D: integral of xyz over unit cube", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    // Hex8 tensor basis {1,x,y,z,xy,xz,yz,xyz}
    MonomialBasis3D mono = get_tensor_basis_3d(8);
    VectorXd coeffs = VectorXd::Zero(8);
    int idx_xyz = find_mono(mono, 1, 1, 1);
    REQUIRE(idx_xyz >= 0);
    coeffs[idx_xyz] = 1.0;

    // ∫_[0,1]^3 xyz dV = (1/2)^3 = 1/8
    double result = integ.integrate(cube, coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(0.125, 1e-12));
}

// ∫_[0,1]^3 (1+x+y+z) dV = 1 + 0.5 + 0.5 + 0.5 = 2.5
TEST_CASE("PolyIntegrator3D: integral of (1+x+y+z) over unit cube", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    MonomialBasis3D mono = get_tensor_basis_3d(4);  // {1, x, y, z}
    VectorXd coeffs = VectorXd::Ones(4);            // 1*1 + 1*x + 1*y + 1*z

    double result = integ.integrate(cube, coeffs, mono);
    CHECK_THAT(result, Catch::Matchers::WithinAbs(2.5, 1e-12));
}

// =============================================================================
// multiply_polynomials — correctness
// =============================================================================

// x * y = xy  (all other coefficients zero)
TEST_CASE("PolyIntegrator3D: multiply x * y = xy", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    MonomialBasis3D mono = get_tensor_basis_3d(4);  // {1, x, y, z}

    VectorXd cx = VectorXd::Zero(4);
    VectorXd cy = VectorXd::Zero(4);
    int ix = find_mono(mono, 1, 0, 0); REQUIRE(ix >= 0); cx[ix] = 1.0;
    int iy = find_mono(mono, 0, 1, 0); REQUIRE(iy >= 0); cy[iy] = 1.0;

    MonomialBasis3D mono_out;
    VectorXd prod = integ.multiply_polynomials(cx, mono, cy, mono, mono_out);

    int idx_xy = find_mono(mono_out, 1, 1, 0);
    REQUIRE(idx_xy >= 0);
    CHECK_THAT(prod[idx_xy], Catch::Matchers::WithinAbs(1.0, 1e-14));

    // Every other coefficient must be zero
    double max_other = 0.0;
    for (int i = 0; i < prod.size(); ++i)
        if (i != idx_xy) max_other = std::max(max_other, std::abs(prod[i]));
    CHECK(max_other < 1e-14);
}

// (1+x+y+z)^2 integrated over unit cube = 6.5
//   = ∫(1 + 2x+2y+2z + x²+y²+z² + 2xy+2xz+2yz) dV
//   = 1 + 3*1 + 3*(1/3) + 3*2*(1/4) = 1+3+1+1.5 = 6.5
TEST_CASE("PolyIntegrator3D: multiply (1+x+y+z)^2 integrates to 6.5", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    MonomialBasis3D mono = get_tensor_basis_3d(4);  // {1, x, y, z}
    VectorXd a = VectorXd::Ones(4);                 // 1 + x + y + z

    MonomialBasis3D mono_out;
    VectorXd prod = integ.multiply_polynomials(a, mono, a, mono, mono_out);

    double integral = integ.integrate(cube, prod, mono_out);
    CHECK_THAT(integral, Catch::Matchers::WithinAbs(6.5, 1e-10));
}

// Symmetry: multiply_polynomials(a, ma, b, mb) == multiply_polynomials(b, mb, a, ma)
TEST_CASE("PolyIntegrator3D: multiply_polynomials is symmetric", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    Polyhedron cube = unit_cube();

    MonomialBasis3D mono = get_tensor_basis_3d(8);  // Hex8 trilinear basis

    VectorXd ca(8), cb(8);
    ca << 1.0, -0.5,  0.3,  0.7, -0.2,  0.1,  0.4, -0.8;
    cb << 0.5,  0.2, -0.1,  0.6,  0.3, -0.4,  0.9, -0.3;

    MonomialBasis3D mono_ab, mono_ba;
    VectorXd prod_ab = integ.multiply_polynomials(ca, mono, cb, mono, mono_ab);
    VectorXd prod_ba = integ.multiply_polynomials(cb, mono, ca, mono, mono_ba);

    REQUIRE(prod_ab.size() == prod_ba.size());
    // Integrals must match (the product is commutative)
    double int_ab = integ.integrate(cube, prod_ab, mono_ab);
    double int_ba = integ.integrate(cube, prod_ba, mono_ba);
    CHECK_THAT(int_ab, Catch::Matchers::WithinAbs(int_ba, 1e-12));
}

// =============================================================================
// Product-basis cache consistency
// =============================================================================

// A warmed-up integrator must produce bit-identical results to a cold one.
TEST_CASE("PolyIntegrator3D: warm_up_product_cache gives same result as cold", "[poly_integrator_3d]")
{
    MonomialBasis3D mono = get_tensor_basis_3d(8);
    int deg = mono.max_degree();

    VectorXd ca(8), cb(8);
    ca << 1.0, -0.5,  0.3,  0.7, -0.2,  0.1,  0.4, -0.8;
    cb << 0.5,  0.2, -0.1,  0.6,  0.3, -0.4,  0.9, -0.3;

    // Cold integrator: builds cache lazily on first multiply call
    PolyIntegrator3D integ_cold;
    MonomialBasis3D mono_out_cold;
    VectorXd prod_cold = integ_cold.multiply_polynomials(ca, mono, cb, mono, mono_out_cold);

    // Pre-warmed integrator: cache populated before multiply call
    PolyIntegrator3D integ_warm;
    integ_warm.warm_up_product_cache(deg, deg);
    MonomialBasis3D mono_out_warm;
    VectorXd prod_warm = integ_warm.multiply_polynomials(ca, mono, cb, mono, mono_out_warm);

    REQUIRE(prod_cold.size() == prod_warm.size());
    double max_diff = (prod_cold - prod_warm).cwiseAbs().maxCoeff();
    CHECK(max_diff < 1e-14);

    // Integrals over unit cube must also match
    Polyhedron cube = unit_cube();
    double int_cold = integ_cold.integrate(cube, prod_cold, mono_out_cold);
    double int_warm = integ_warm.integrate(cube, prod_warm, mono_out_warm);
    CHECK_THAT(int_cold, Catch::Matchers::WithinAbs(int_warm, 1e-14));
}

// Repeated warm_up calls for the same degree pair are idempotent (no crash).
TEST_CASE("PolyIntegrator3D: repeated warm_up is safe", "[poly_integrator_3d]")
{
    PolyIntegrator3D integ;
    integ.warm_up_product_cache(3, 3);
    integ.warm_up_product_cache(3, 3);  // second call — must be a no-op
    integ.warm_up_product_cache(1, 3);  // different pair — must also work

    MonomialBasis3D mono8 = get_tensor_basis_3d(8);
    MonomialBasis3D mono4 = get_tensor_basis_3d(4);

    VectorXd a8 = VectorXd::Ones(8);
    VectorXd a4 = VectorXd::Ones(4);

    MonomialBasis3D mo1, mo2;
    VectorXd p1 = integ.multiply_polynomials(a8, mono8, a8, mono8, mo1);
    VectorXd p2 = integ.multiply_polynomials(a4, mono4, a8, mono8, mo2);

    CHECK(p1.size() > 0);
    CHECK(p2.size() > 0);
}

// =============================================================================
// BasisBuilder3D — Lagrange delta property with ColPivHouseholderQR
// =============================================================================

// After switching to ColPivHouseholderQR, basis functions must still satisfy
// φ_i(x_j) = δ_{ij} for all 8 Gauss points of a unit-cube Hex8.
TEST_CASE("BasisBuilder3D: Lagrange delta property (ColPivHouseholderQR)", "[basis_builder_3d]")
{
    // Build a simple 8-point set from a unit cube Hex8
    MatrixXd nc(8, 3);
    nc << 0,0,0,  1,0,0,  1,1,0,  0,1,0,
          0,0,1,  1,0,1,  1,1,1,  0,1,1;

    // Use the 8 Gauss points of this element as the interpolation nodes
    // (mimic what the engine does)
    std::vector<Point3D> pts;
    pts.reserve(8);
    for (int i = 0; i < 8; ++i) pts.push_back(nc.row(i));

    BasisBuilder3D bb;
    BasisMatrix basis = bb.build(pts);

    MonomialBasis3D mono = get_tensor_basis_3d(8);
    Point3D origin = pts.back();

    int N = 8;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Point3D p_sh = pts[j] - origin;
            VectorXd m = mono.evaluate(p_sh[0], p_sh[1], p_sh[2]);
            double val = basis.row(i).dot(m);
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK_THAT(val, Catch::Matchers::WithinAbs(expected, 1e-10));
        }
    }
}
