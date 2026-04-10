#include "l2map/poly_integrator.hpp"
#include <cmath>
#include <stdexcept>

namespace l2map {

PolyIntegrator::PolyIntegrator(int n_gauss_pts) : n_gauss_pts_(n_gauss_pts) {
    init_gauss_points_();
}

void PolyIntegrator::init_gauss_points_() {
    if (n_gauss_pts_ == 5) {
        // 5-point Gauss-Legendre on [-1, 1]
        gauss_pts_     = {-0.9061798459386640, -0.5384693101056831, 0.0,
                           0.5384693101056831,  0.9061798459386640};
        gauss_weights_ = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                          0.4786286704993665, 0.2369268850561891};
    } else if (n_gauss_pts_ == 3) {
        gauss_pts_     = {-0.7745966692414834, 0.0, 0.7745966692414834};
        gauss_weights_ = { 0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
    } else if (n_gauss_pts_ == 1) {
        gauss_pts_     = {0.0};
        gauss_weights_ = {2.0};
    } else {
        throw std::runtime_error("PolyIntegrator: unsupported n_gauss_pts " +
                                 std::to_string(n_gauss_pts_) + "; use 1, 3, or 5");
    }
}

// ---------------------------------------------------------------------------
// Homogeneous polynomial decomposition
// ---------------------------------------------------------------------------

std::vector<VectorXd> PolyIntegrator::decompose_homogeneous_(
    const VectorXd& coeffs,
    const MonomialBasis2D& mono) const
{
    int max_deg = mono.max_degree();
    std::vector<VectorXd> parts(max_deg + 1);
    for (int d = 0; d <= max_deg; ++d)
        parts[d] = VectorXd::Zero(d + 1);

    for (int i = 0; i < mono.n_monomials; ++i) {
        int px = mono.monomials[i].first;
        int py = mono.monomials[i].second;
        int d  = px + py;
        // Within homogeneous poly of degree d, index by decreasing x-power
        int k = d - px; // k = py = position in {x^d, x^{d-1}y, ..., y^d}
        parts[d][k] += coeffs[i];
    }
    return parts;
}

// ---------------------------------------------------------------------------
// Evaluate a homogeneous polynomial of degree d at (x, y)
// homo_coeffs indexed as: [x^d, x^{d-1}y, x^{d-2}y^2, ..., y^d]
// ---------------------------------------------------------------------------

double PolyIntegrator::eval_homo_(const VectorXd& homo_coeffs, int degree,
                                   double x, double y) const
{
    double result = 0.0;
    // k-th term: x^{d-k} * y^k
    double xpow = std::pow(x, degree); // x^d
    double ypow = 1.0;                  // y^0
    double y_over_x = (x == 0.0) ? 0.0 : y / x;

    // Direct computation to avoid pow() cost
    for (int k = 0; k <= degree; ++k) {
        // x^{d-k} * y^k
        double xk = 1.0, yk = 1.0;
        for (int j = 0; j < degree - k; ++j) xk *= x;
        for (int j = 0; j < k;          ++j) yk *= y;
        result += homo_coeffs[k] * xk * yk;
    }
    (void)xpow; (void)ypow; (void)y_over_x;
    return result;
}

// ---------------------------------------------------------------------------
// Integrate homogeneous polynomial of degree d over directed edge v0→v1
// using Gauss quadrature on [0,1] parametrisation.
//
// ∫_Ω f_d dA = 1/(d+2) * Σ_edges ∫_0^1 f_d(F(t)) dt
// where F(t) = v0 + t*(v1 - v0).
// ---------------------------------------------------------------------------

double PolyIntegrator::integrate_homo_over_edge_(const VectorXd& homo_coeffs,
                                                  int degree,
                                                  const Point2D& v0,
                                                  const Point2D& v1) const
{
    double dx = v1[0] - v0[0];
    double dy = v1[1] - v0[1];
    // Stokes formula cross-product factor: (x₀·dy − y₀·dx)
    double cross = v0[0] * dy - v0[1] * dx;
    double sum = 0.0;

    for (int l = 0; l < n_gauss_pts_; ++l) {
        // Map Gauss node from [-1,1] to [0,1]
        double t  = (gauss_pts_[l] + 1.0) * 0.5;
        double wt = gauss_weights_[l] * 0.5;  // Jacobian dt/ds = 1/2

        double x = v0[0] + t * dx;
        double y = v0[1] + t * dy;
        sum += wt * eval_homo_(homo_coeffs, degree, x, y);
    }
    return cross * sum;
}

// ---------------------------------------------------------------------------
// Main integration via Stokes theorem reduction:
//   ∫_Ω g dA = Σ_d [1/(d+2) * Σ_edges ∫_edge f_d dσ]
// ---------------------------------------------------------------------------

double PolyIntegrator::integrate(const Polygon2D& poly,
                                  const VectorXd& coeffs,
                                  const MonomialBasis2D& mono) const
{
    if (poly.size() < 3) return 0.0;

    std::vector<VectorXd> homo_parts = decompose_homogeneous_(coeffs, mono);
    int max_deg = static_cast<int>(homo_parts.size()) - 1;
    int n = static_cast<int>(poly.size());

    double total = 0.0;
    for (int d = 0; d <= max_deg; ++d) {
        const VectorXd& hc = homo_parts[d];
        // Check if this homogeneous part is non-zero
        if (hc.norm() == 0.0) continue;

        double edge_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const Point2D& v0 = poly[i];
            const Point2D& v1 = poly[(i + 1) % n];
            edge_sum += integrate_homo_over_edge_(hc, d, v0, v1);
        }
        total += edge_sum / static_cast<double>(d + 2);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Polynomial multiplication
// ---------------------------------------------------------------------------

VectorXd PolyIntegrator::multiply_polynomials(
    const VectorXd& coeffs_a,
    const MonomialBasis2D& mono_a,
    const VectorXd& coeffs_b,
    const MonomialBasis2D& mono_b,
    MonomialBasis2D& mono_product_out) const
{
    int deg_a = mono_a.max_degree();
    int deg_b = mono_b.max_degree();
    int deg_out = deg_a + deg_b;

    // Count output monomials: (deg_out+1)*(deg_out+2)/2
    int n_out = (deg_out + 1) * (deg_out + 2) / 2;
    mono_product_out = get_monomial_basis_2d(n_out);

    // Build a lookup: (px, py) -> index in output basis
    // The Pascal ordering guarantees: index = sum of row sizes before degree d + position in row
    auto mono_idx = [&](int px, int py) -> int {
        int d = px + py;
        int row_start = d * (d + 1) / 2;
        int pos = d - px;  // position within degree-d row (decreasing x = increasing y)
        return row_start + pos;
    };

    VectorXd result = VectorXd::Zero(n_out);
    for (int i = 0; i < mono_a.n_monomials; ++i) {
        if (coeffs_a[i] == 0.0) continue;
        for (int j = 0; j < mono_b.n_monomials; ++j) {
            if (coeffs_b[j] == 0.0) continue;
            int px = mono_a.monomials[i].first  + mono_b.monomials[j].first;
            int py = mono_a.monomials[i].second + mono_b.monomials[j].second;
            result[mono_idx(px, py)] += coeffs_a[i] * coeffs_b[j];
        }
    }
    return result;
}

} // namespace l2map
