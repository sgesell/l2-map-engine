#include "l2map/basis_builder.hpp"
#include <cmath>
#include <sstream>

namespace l2map {

// ---------------------------------------------------------------------------
// MonomialBasis2D
// ---------------------------------------------------------------------------

VectorXd MonomialBasis2D::evaluate(double x, double y) const {
    VectorXd v(n_monomials);
    for (int i = 0; i < n_monomials; ++i) {
        int px = monomials[i].first;
        int py = monomials[i].second;
        double val = 1.0;
        for (int k = 0; k < px; ++k) val *= x;
        for (int k = 0; k < py; ++k) val *= y;
        v[i] = val;
    }
    return v;
}

MonomialBasis2D get_monomial_basis_2d(int n_points) {
    MonomialBasis2D basis;
    basis.n_monomials = n_points;
    basis.monomials.reserve(n_points);

    // Fill Pascal triangle row by row
    for (int deg = 0; basis.n_monomials > static_cast<int>(basis.monomials.size()); ++deg) {
        for (int px = deg; px >= 0; --px) {
            int py = deg - px;
            basis.monomials.push_back({px, py});
            if (static_cast<int>(basis.monomials.size()) == n_points) break;
        }
    }
    return basis;
}

MonomialBasis2D get_serendipity_basis_2d(int n_points) {
    int k = static_cast<int>(std::round(std::sqrt(static_cast<double>(n_points))));
    if (k * k != n_points)
        return get_monomial_basis_2d(n_points);  // non-square: fall back to Pascal

    // Tensor-product basis: iterate by total degree, include only (px,py) with
    // px <= k-1 and py <= k-1.  This gives exactly k² monomials and is unisolvent
    // on any k×k grid of distinct (x_i, y_j) values.
    MonomialBasis2D basis;
    basis.n_monomials = n_points;
    basis.monomials.reserve(n_points);
    for (int deg = 0; static_cast<int>(basis.monomials.size()) < n_points; ++deg) {
        for (int px = deg; px >= 0; --px) {
            int py = deg - px;
            if (px <= k - 1 && py <= k - 1) {
                basis.monomials.push_back({px, py});
                if (static_cast<int>(basis.monomials.size()) == n_points) break;
            }
        }
    }
    return basis;
}

// ---------------------------------------------------------------------------
// BasisBuilder
// ---------------------------------------------------------------------------

MatrixXd BasisBuilder::build_vandermonde_(const std::vector<Point2D>& points,
                                          const MonomialBasis2D& mono) const
{
    int N = static_cast<int>(points.size());
    MatrixXd A(N, N);
    for (int i = 0; i < N; ++i) {
        VectorXd row = mono.evaluate(points[i][0], points[i][1]);
        A.row(i) = row;
    }
    return A;
}

BasisMatrix BasisBuilder::build(const std::vector<Point2D>& points) const {
    int N = static_cast<int>(points.size());
    if (N == 0) return BasisMatrix(0, 0);

    // Shift: subtract last point so it becomes the origin
    Point2D origin = points.back();
    std::vector<Point2D> shifted(N);
    for (int i = 0; i < N; ++i)
        shifted[i] = points[i] - origin;

    MonomialBasis2D mono = get_serendipity_basis_2d(N);
    MatrixXd A = build_vandermonde_(shifted, mono);

    // Solve A * U = I  =>  U = A^{-1}
    Eigen::FullPivLU<MatrixXd> lu(A);
    if (lu.rank() < N) {
        std::ostringstream ss;
        ss << "BasisBuilder: Vandermonde matrix is rank-deficient (rank=" << lu.rank()
           << " < " << N << ") for " << N << " points. First point: ("
           << points[0][0] << ", " << points[0][1] << "). "
           << "This typically indicates a degenerate element geometry.";
        throw std::runtime_error(ss.str());
    }

    // lu.solve(I) = A^{-1}, where COLUMN j = coefficients of φ_j.
    // We need ROW i = coefficients of φ_i, so we return the TRANSPOSE: A^{-T}.
    //
    // Proof: φ_j(x_i) = Σ_k (A^{-1})[k,j] * m_k(x_i) = (A * A^{-1})[i,j] = δ_{ij}
    // → column j of A^{-1} is the coefficient vector of φ_j.
    // Transposing: row j of A^{-T} is the coefficient vector of φ_j. ✓
    MatrixXd I = MatrixXd::Identity(N, N);
    return lu.solve(I).transpose();  // shape (N, N): row i = coefficients of φ_i
}

double BasisBuilder::evaluate_basis(const BasisMatrix& basis,
                                    int i,
                                    const Point2D& p,
                                    const MonomialBasis2D& mono) const
{
    VectorXd m = mono.evaluate(p[0], p[1]);
    return basis.row(i).dot(m);
}

} // namespace l2map
