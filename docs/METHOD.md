# Method contract

Let `q` be the query direction for `(h, r)`, so a candidate tail embedding `e_t`
receives score `qᵀe_t`. Let `b` be the non-negative increase required to clear
the current top-k threshold by margin `1e-6`. Protected query directions form the
rows of `C`.

## Rank-truncated projection

Compute `C = UΣVᵀ` in float64. The numerical constraint rank counts singular
values strictly greater than `1e-6`. For embedding width `d` and fraction `f`,
the requested rank is

```text
m_requested = floor(f d + 0.5)
m_used      = min(m_requested, rank(C)).
```

With `V_m` containing the first `m_used` right-singular vectors,

```text
q_projected = q - V_mᵀ V_m q
delta       = b q_projected / (qᵀ q_projected).
```

The update is a no-op when `b <= 0` or the denominator has magnitude below
`1e-12`. Fraction zero is exactly direct promotion. Fraction one is the strict
null-space endpoint whenever a feasible direction remains.

## Ridge preservation

For preservation weight `lambda > 0`, form

```text
B = lambda CᵀC + 1e-8 I
z = B⁻¹q
delta = b z / (1 + qᵀz).
```

The extra `1` and the `1e-8` jitter are frozen parts of the evaluated method.
They are retained for numerical equivalence with the experiment implementation.

## Metrics

- `success_at_k`: edited target rank is at most k.
- `damage_count`: protected facts initially in the top k but outside it after
  the candidate-column intervention.
- `no_damage`: `damage_count == 0`.
- `safe_success`: `success_at_k and no_damage`.

Ties use competition ranking: `rank = 1 + count(score > target_score)`.
