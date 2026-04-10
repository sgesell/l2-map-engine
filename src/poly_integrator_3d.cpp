#include "l2map/poly_integrator_3d.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace l2map {

PolyIntegrator3D::PolyIntegrator3D(int n_gauss_pts)
    : integrator_2d_(n_gauss_pts)
{}

// ---------------------------------------------------------------------------
// Decompose a 3D polynomial into homogeneous parts.
// Returns one entry per distinct degree present.
// ---------------------------------------------------------------------------

std::vector<PolyIntegrator3D::HomoPart>
PolyIntegrator3D::decompose_homogeneous_(const VectorXd& coeffs,
                                          const MonomialBasis3D& mono) const
{
    // Find max degree
    int max_deg = 0;
    for (int i = 0; i < mono.n_monomials; ++i) {
        auto [px,py,pz] = mono.monomials[i];
        max_deg = std::max(max_deg, px+py+pz);
    }

    // Collect terms by degree
    std::vector<HomoPart> parts(max_deg + 1);
    for (int d = 0; d <= max_deg; ++d) parts[d].degree = d;

    for (int i = 0; i < mono.n_monomials; ++i) {
        double c = coeffs[i];
        if (c == 0.0) continue;
        auto [px,py,pz] = mono.monomials[i];
        int d = px+py+pz;
        parts[d].terms.push_back({{px,py,pz}, c});
    }
    return parts;
}

// ---------------------------------------------------------------------------
// Expand (c0 + cu*u + cv*v)^n into Pascal-ordered 2D coefficients.
// Multinomial theorem: coefficient of u^j v^k = C(n;n-j-k,j,k) c0^{n-j-k} cu^j cv^k
// Pascal index for u^j v^k: d*(d+1)/2 + k  where d = j+k
// ---------------------------------------------------------------------------

VectorXd PolyIntegrator3D::expand_linear_power_(double c0, double cu, double cv, int n)
{
    int sz = (n+1)*(n+2)/2;
    VectorXd result = VectorXd::Zero(sz);

    // Precompute factorial table up to n
    std::vector<double> fact(n+1, 1.0);
    for (int i = 1; i <= n; ++i) fact[i] = fact[i-1] * i;

    for (int j = 0; j <= n; ++j) {
        for (int k = 0; k <= n-j; ++k) {
            int a = n - j - k;   // power of c0
            double coeff = fact[n] / (fact[a] * fact[j] * fact[k]);
            double val = coeff;
            for (int ii = 0; ii < a; ++ii) val *= c0;
            for (int ii = 0; ii < j; ++ii) val *= cu;
            for (int ii = 0; ii < k; ++ii) val *= cv;

            int d = j + k;
            int idx = d*(d+1)/2 + k;
            result[idx] += val;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Multiply two Pascal-ordered 2D polynomials of degree deg_a and deg_b.
// Output degree = deg_a + deg_b.
// Pascal index for u^j v^k: d*(d+1)/2 + k  where d = j+k
// ---------------------------------------------------------------------------

VectorXd PolyIntegrator3D::multiply_poly2d_(const VectorXd& a, int deg_a,
                                             const VectorXd& b, int deg_b)
{
    int deg_out = deg_a + deg_b;
    int sz_out  = (deg_out+1)*(deg_out+2)/2;
    VectorXd result = VectorXd::Zero(sz_out);

    // Iterate over a by (ja, ka) and b by (jb, kb)
    // Pascal index for (j,k): d*(d+1)/2+k, d=j+k
    for (int da = 0; da <= deg_a; ++da) {
        for (int ka = 0; ka <= da; ++ka) {
            int ja  = da - ka;
            int ia  = da*(da+1)/2 + ka;
            double ca = (ia < a.size()) ? a[ia] : 0.0;
            if (ca == 0.0) continue;

            for (int db = 0; db <= deg_b; ++db) {
                for (int kb = 0; kb <= db; ++kb) {
                    int jb  = db - kb;
                    int ib  = db*(db+1)/2 + kb;
                    double cb = (ib < b.size()) ? b[ib] : 0.0;
                    if (cb == 0.0) continue;

                    int j_out = ja + jb;
                    int k_out = ka + kb;
                    int d_out = j_out + k_out;
                    int i_out = d_out*(d_out+1)/2 + k_out;
                    result[i_out] += ca * cb;
                }
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Integrate one homogeneous 3D polynomial over one planar face.
//
// Strategy:
//   1. Set up local orthonormal (u,v) axes on the face plane.
//   2. Map face vertices to 2D.
//   3. Express each 3D monomial (px,py,pz) as a 2D polynomial in (u,v).
//   4. Accumulate into one 2D polynomial.
//   5. Use existing 2D PolyIntegrator (Stokes) to integrate.
// ---------------------------------------------------------------------------

double PolyIntegrator3D::integrate_homo_on_face_(const HomoPart& hp,
                                                  const Face3D& face) const
{
    if (hp.terms.empty()) return 0.0;
    if (face.size() < 3) return 0.0;

    // --- Local coordinate system on the face plane ---
    // Origin at first vertex x0
    Point3D x0 = face[0];

    // e_u = direction of first edge
    Point3D e_u = (face[1] - x0).normalized();

    // Face normal from first two edges
    Point3D edge2 = face[2] - face[1];
    Point3D n_face = e_u.cross(edge2);
    double nlen = n_face.norm();
    if (nlen < 1e-14) return 0.0;
    n_face /= nlen;

    // e_v = n × e_u  (orthogonal to both n and e_u, pointing into the face)
    Point3D e_v = n_face.cross(e_u);

    // --- Map face vertices to 2D ---
    Polygon2D poly_2d;
    poly_2d.reserve(face.size());
    for (const auto& v : face) {
        Point3D d = v - x0;
        poly_2d.push_back(Point2D(d.dot(e_u), d.dot(e_v)));
    }

    // --- Determine degree of the homogeneous part ---
    int degree = hp.degree;

    // --- Build the 2D polynomial for this homogeneous part ---
    // For each term (px,py,pz): expand x^px * y^py * z^pz in (u,v) coords.
    // x = x0.x + e_u.x*u + e_v.x*v  (and same for y, z)

    int sz_out = (degree+1)*(degree+2)/2;
    VectorXd poly2d_coeffs = VectorXd::Zero(sz_out);

    for (const auto& [exps, coeff] : hp.terms) {
        auto [px, py, pz] = exps;

        // Expand each factor as a 2D polynomial
        VectorXd fx = expand_linear_power_(x0[0], e_u[0], e_v[0], px);
        VectorXd fy = expand_linear_power_(x0[1], e_u[1], e_v[1], py);
        VectorXd fz = expand_linear_power_(x0[2], e_u[2], e_v[2], pz);

        // Multiply the three 2D polynomials
        VectorXd fxy  = multiply_poly2d_(fx, px, fy, py);
        VectorXd fxyz = multiply_poly2d_(fxy, px+py, fz, pz);

        poly2d_coeffs += coeff * fxyz;
    }

    // --- Build the MonomialBasis2D corresponding to the Pascal order used ---
    MonomialBasis2D mono2d = get_monomial_basis_2d(sz_out);

    // --- Integrate using the existing 2D PolyIntegrator ---
    return integrator_2d_.integrate(poly_2d, poly2d_coeffs, mono2d);
}

// ---------------------------------------------------------------------------
// integrate: divergence theorem over convex polyhedron
//
//   ∫_P g dV = Σ_d [1/(d+3)] * Σ_faces (n_f · v_0f) * ∫_face g_d dA
// ---------------------------------------------------------------------------

double PolyIntegrator3D::integrate(const Polyhedron& poly,
                                    const VectorXd& coeffs,
                                    const MonomialBasis3D& mono) const
{
    if (poly.faces.empty()) return 0.0;

    auto homo_parts = decompose_homogeneous_(coeffs, mono);

    double total = 0.0;

    for (const Face3D& face : poly.faces) {
        if (face.size() < 3) continue;

        // Outward unit normal of this face
        Point3D e1 = face[1] - face[0];
        Point3D e2 = face[2] - face[1];
        Point3D n_raw = e1.cross(e2);
        double nlen = n_raw.norm();
        if (nlen < 1e-14) continue;
        Point3D n_f = n_raw / nlen;

        // n_f · v_0 = constant offset of the face plane from origin
        double n_dot_v0 = n_f.dot(face[0]);

        // Contribution from each homogeneous degree
        for (const auto& hp : homo_parts) {
            if (hp.terms.empty()) continue;
            double face_integral = integrate_homo_on_face_(hp, face);
            total += (1.0 / (hp.degree + 3)) * n_dot_v0 * face_integral;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// multiply_polynomials (3D)
// ---------------------------------------------------------------------------

// Helper: count Pascal-tetrahedron monomials up to (inclusive) degree d.
static int n_monomials_up_to_deg(int d) {
    return (d + 1) * (d + 2) * (d + 3) / 6;
}

// Build and cache the product-basis descriptor for (deg_a, deg_b).
// After the first call the result lives in product_cache_ and subsequent
// calls are a single unordered_map::find — no allocation, no hashing of
// individual monomial coordinates.
void PolyIntegrator3D::warm_up_product_cache(int deg_a, int deg_b) const
{
    int key = deg_a * 100 + deg_b;
    if (product_cache_.count(key)) return;  // already cached

    int deg_out = deg_a + deg_b;
    int n_out   = n_monomials_up_to_deg(deg_out);
    int stride  = deg_out + 1;  // each per-direction power is in [0, deg_out]

    ProductBasisCache pc;
    pc.mono_out = get_tensor_basis_3d(n_out);
    pc.stride   = stride;
    pc.idx_table.assign(stride * stride * stride, -1);

    for (int i = 0; i < n_out; ++i) {
        auto [px, py, pz] = pc.mono_out.monomials[i];
        pc.idx_table[px * stride * stride + py * stride + pz] = i;
    }

    product_cache_.emplace(key, std::move(pc));
}

VectorXd PolyIntegrator3D::multiply_polynomials(const VectorXd& ca,
                                                  const MonomialBasis3D& ma,
                                                  const VectorXd& cb,
                                                  const MonomialBasis3D& mb,
                                                  MonomialBasis3D& mono_out) const
{
    int deg_a = ma.max_degree();
    int deg_b = mb.max_degree();

    // Ensure cache entry exists (no-op if already warmed up).
    warm_up_product_cache(deg_a, deg_b);

    const ProductBasisCache& pc = product_cache_.at(deg_a * 100 + deg_b);
    mono_out    = pc.mono_out;
    int n_out   = pc.mono_out.n_monomials;
    int stride  = pc.stride;

    // Flat-array lookup: idx_table[ox*stride^2 + oy*stride + oz] → index in mono_out.
    // This replaces the previous per-call unordered_map (hash + allocation).
    VectorXd result = VectorXd::Zero(n_out);

    for (int i = 0; i < ma.n_monomials; ++i) {
        if (ca[i] == 0.0) continue;
        auto [ax, ay, az] = ma.monomials[i];
        for (int j = 0; j < mb.n_monomials; ++j) {
            if (cb[j] == 0.0) continue;
            auto [bx, by, bz] = mb.monomials[j];
            int ox = ax + bx, oy = ay + by, oz = az + bz;
            int flat_idx = pc.idx_table[ox * stride * stride + oy * stride + oz];
            if (flat_idx >= 0)
                result[flat_idx] += ca[i] * cb[j];
        }
    }
    return result;
}

} // namespace l2map
