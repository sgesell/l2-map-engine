# L2MapEngine — Math Notes

## Core equation

The L²-projection satisfies:

```
∫_Ω u_h χ dV  =  ∫_Ω u_n χ dV      for all χ ∈ V_n
```

where `u_h` is the source field and `u_n` is the projected field on the new mesh.

## Element-local system

For each new element `e_n`, the local system is:

```
V * α = M

V[i,j] = ∫_{e_n} φ_i φ_j dA

M[j,l] = Σ_{e_o : e_o ∩ e_n ≠ ∅}  ∫_{e_o ∩ e_n} φ_j · (Σ_k β_{k,l} ψ_k) dA
```

where:
- `φ_i` are basis polynomials for the new element
- `ψ_k` are basis polynomials for the old element
- `β_{k,l}` are field values at old element integration points (component `l`)
- `α_{i,l}` are the solved field values at new element integration points

## Stokes theorem for polygon integration

```
∫_Ω g(x,y) dA = Σ_d [1/(d+2)] * Σ_edges ∫_0^1 f_d(F(t)) dt
```

where `f_d` is the homogeneous degree-`d` component of `g`, and `F(t) = v_k + t(v_{k+1} - v_k)`.

## Coordinate shifting

All computations shift coordinates so that the last Gauss point of the **new** element is at the origin:

```
x_shifted = x_global - x_origin
```

This is critical for numerical stability when absolute coordinates are large.

## Monomial basis (Pascal triangle order)

```
degree 0:  1                           → 1 term  (total: 1)
degree 1:  x, y                        → 2 terms (total: 3)
degree 2:  x², xy, y²                  → 3 terms (total: 6)
degree 3:  x³, x²y, xy², y³           → 4 terms (total: 10)
degree 4:  x⁴, x³y, x²y², xy³, y⁴   → 5 terms (total: 15)
```

For a Quad8 element with 9 integration points, we use 9 monomials:
`{1, x, y, x², xy, y², x³, x²y, xy²}` — the first 9 in Pascal order.

Note: this is NOT the serendipity 8-monomial set. The Python reference also uses 9 monomials.

## Vandermonde inversion

```
A_P[i][j] = m_j(x_i_shifted, y_i_shifted)
U_P = A_P^{-1}    (via Eigen::FullPivLU)
```

Row `k` of `U_P` gives the coefficient vector of basis polynomial `φ_k`.

## References

- Chin, E.B., Lasserre, J.B., Sukumar, N. (2015). "Numerical integration of homogeneous
  functions on convex and nonconvex polygons and polyhedra." *Computational Mechanics*.
- ABAQUS Theory Manual, Section on element formulations.
