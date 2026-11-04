"""max_iter: when the simplex runs out of pivots it must warn instead of
silently returning the cost of a non-optimal plan (an upper bound on W_1)."""
import warnings

import numpy as np
import pytest

import ricci_numpy as rn

try:
    import ricci_numpy.fast as rnf
    HAS_FAST = True
except ImportError:
    rnf = None
    HAS_FAST = False

IMPLS = [
    pytest.param(rn, id="core"),
    pytest.param(rnf, id="fast", marks=pytest.mark.skipif(
        not HAS_FAST, reason="numba not installed")),
]


def _problem(seed=0, k=8):
    """Random transport problem whose northwest-corner start is not optimal."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k))
    q = rng.dirichlet(np.ones(k))
    C = rng.integers(1, 6, size=(k, k)).astype(float)
    return p, q, C


def _solve(impl, p, q, C, max_iter):
    """W_1, and whether the max_iter RuntimeWarning was emitted."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w = impl.wasserstein(p, q, C, max_iter=max_iter)
    warned = any(issubclass(r.category, RuntimeWarning)
                 and "max_iter" in str(r.message) for r in rec)
    return w, warned


@pytest.mark.parametrize("impl", IMPLS)
def test_warns_exactly_when_pivots_run_out(impl):
    p, q, C = _problem()
    w_opt, warned = _solve(impl, p, q, C, max_iter=5000)
    assert not warned
    needed = next(k for k in range(200) if not _solve(impl, p, q, C, k)[1])
    assert needed >= 1, "test problem should need at least one pivot"
    for k in range(needed):
        w, warned = _solve(impl, p, q, C, k)
        assert warned, f"no warning with max_iter={k} < {needed} pivots needed"
        assert w >= w_opt - 1e-12  # truncated plan is feasible: upper bound
    # Exactly enough pivots: optimality is checked after the last one.
    w, warned = _solve(impl, p, q, C, needed)
    assert not warned
    assert w == pytest.approx(w_opt, abs=1e-12)


@pytest.mark.parametrize("impl", IMPLS)
def test_max_iter_is_passed_through(impl):
    A = impl.adjacency([(0, 1), (1, 2), (0, 2), (2, 3),
                        (3, 4), (3, 5), (4, 5)])  # two triangles + bridge
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        k_default = impl.all_curvatures(A)
    with pytest.warns(RuntimeWarning, match="max_iter=0"):
        k_capped = impl.all_curvatures(A, max_iter=0)
    # Truncation can only overestimate W_1, i.e. underestimate curvature.
    assert np.all(k_capped <= k_default + 1e-12)
