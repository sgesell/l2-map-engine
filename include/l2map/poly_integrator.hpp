#pragma once
#include "types.hpp"
#include "basis_builder.hpp"  // for MonomialBasis2D
#include <vector>
#include <utility>

namespace l2map {

class PolyIntegrator {
public:
    // n_gauss_pts: number of Gauss-Legendre points per edge (default 5)
    explicit PolyIntegrator(int n_gauss_pts = 5);

    // Integrate a polynomial over an arbitrary 2D polygon using the Stokes theorem reduction.
    //
    // poly:   polygon vertices in CCW order (first != last, closed implicitly)
    // coeffs: polynomial coefficients in MonomialBasis2D ordering
    // mono:   the monomial basis corresponding to coeffs
    //
    // Returns the area integral ∫_poly g(x,y) dA.
    double integrate(const Polygon2D& poly,
                     const VectorXd& coeffs,
                     const MonomialBasis2D& mono) const;

    // Multiply two polynomials expressed in MonomialBasis2D.
    // Output monomial basis is written to mono_product_out.
    VectorXd multiply_polynomials(const VectorXd& coeffs_a,
                                  const MonomialBasis2D& mono_a,
                                  const VectorXd& coeffs_b,
                                  const MonomialBasis2D& mono_b,
                                  MonomialBasis2D& mono_product_out) const;

private:
    int                 n_gauss_pts_;
    std::vector<double> gauss_pts_;      // nodes in [-1, 1]
    std::vector<double> gauss_weights_;  // sum = 2

    void init_gauss_points_();

    // Decompose polynomial into homogeneous parts.
    // Returns vector indexed by degree: element d = coeffs of degree-d part
    // (length d+1 each, in decreasing x-power order)
    std::vector<VectorXd> decompose_homogeneous_(
        const VectorXd& coeffs,
        const MonomialBasis2D& mono) const;

    // Integrate one homogeneous polynomial of given degree over directed edge v0→v1
    // using Gauss quadrature on the parametric segment [0,1].
    double integrate_homo_over_edge_(const VectorXd& homo_coeffs,
                                     int degree,
                                     const Point2D& v0,
                                     const Point2D& v1) const;

    // Evaluate a homogeneous polynomial at (x, y)
    // homo_coeffs: length (degree+1), ordered as x^d, x^{d-1}y, ..., y^d
    double eval_homo_(const VectorXd& homo_coeffs, int degree,
                      double x, double y) const;
};

} // namespace l2map
