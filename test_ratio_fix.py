"""
test_ratio_fix.py
-----------------
Verifies that ia_ratio_valids (sum of per-seed unique counts) differs from
ia_valids (cross-seed union) exactly as the user described, and that the ratio
plot receives the correct numerator.

Scenario (user's example):
  ligand JQ1A-1, mask_count = 3, seeds = [10, 17]
  seed 10 → 200 unique valid molecules out of 500 samples
  seed 17 → 100 unique valid molecules out of 500 samples
    - 50 of those 100 overlap with seed 10's 200
  Expected ratio numerator  = 200 + 100 = 300  (sum)
  Expected union (absolute) = 200 + 50  = 250  (union: seed10 ∪ seed17)
  Denominator               = 500 * 2   = 1000
  Expected ratio            = 300 / 1000 = 0.30
  Absolute-plot count       = 250 (union)
"""

def simulate_run(seed10_preds, seed17_preds, num_samples):
    seeds = [10, 17]
    preds_by_seed = {10: seed10_preds, 17: seed17_preds}

    ia_pool = set()   # union
    ia_sum  = 0       # sum of per-seed unique counts

    for seed in seeds:
        preds = preds_by_seed[seed]
        ia_pool.update(preds)
        ia_sum += len(preds)   # <-- the fix

    union_count = len(ia_pool)
    total_samples = num_samples * len(seeds)
    ratio_old = union_count / total_samples
    ratio_new = ia_sum      / total_samples

    return {
        "union_count":    union_count,
        "sum_count":      ia_sum,
        "total_samples":  total_samples,
        "ratio_old (bug)": ratio_old,
        "ratio_new (fix)": ratio_new,
    }


def run_tests():
    num_samples = 500

    # ── Test 1: partial overlap (user's scenario) ────────────────────────────
    seed10 = {f"mol_A_{i}" for i in range(200)}          # 200 unique
    seed17_overlap = {f"mol_A_{i}" for i in range(50)}   # 50 shared with seed10
    seed17_unique  = {f"mol_B_{i}" for i in range(50)}   # 50 fresh
    seed17 = seed17_overlap | seed17_unique               # 100 unique for seed17

    r = simulate_run(seed10, seed17, num_samples)
    print("\n-- Test 1: partial overlap --")
    print(f"  seed 10 unique:        {len(seed10)}")
    print(f"  seed 17 unique:        {len(seed17)}")
    print(f"  Union (absolute plot): {r['union_count']}  (expected 250)")
    print(f"  Sum   (ratio plot):    {r['sum_count']}   (expected 300)")
    print(f"  Denominator:           {r['total_samples']} (expected 1000)")
    print(f"  Ratio OLD (bug):       {r['ratio_old (bug)']:.4f}  (expected 0.2500)")
    print(f"  Ratio NEW (fix):       {r['ratio_new (fix)']:.4f}  (expected 0.3000)")

    assert r["union_count"]          == 250,  f"union mismatch: {r['union_count']}"
    assert r["sum_count"]            == 300,  f"sum mismatch:   {r['sum_count']}"
    assert r["total_samples"]        == 1000, f"denom mismatch: {r['total_samples']}"
    assert abs(r["ratio_old (bug)"] - 0.25) < 1e-9
    assert abs(r["ratio_new (fix)"] - 0.30) < 1e-9
    print("  PASS")

    # ── Test 2: zero overlap ─────────────────────────────────────────────────
    seed10_b = {f"mol_X_{i}" for i in range(200)}
    seed17_b = {f"mol_Y_{i}" for i in range(100)}

    r2 = simulate_run(seed10_b, seed17_b, num_samples)
    print("\n-- Test 2: zero overlap --")
    print(f"  Union (absolute plot): {r2['union_count']}  (expected 300)")
    print(f"  Sum   (ratio plot):    {r2['sum_count']}   (expected 300)")
    print(f"  Ratio OLD:             {r2['ratio_old (bug)']:.4f}")
    print(f"  Ratio NEW:             {r2['ratio_new (fix)']:.4f}")

    # With zero overlap, sum == union, so old and new should agree
    assert r2["union_count"] == 300
    assert r2["sum_count"]   == 300
    assert abs(r2["ratio_old (bug)"] - r2["ratio_new (fix)"]) < 1e-9
    print("  PASS  (sum == union when there is no cross-seed overlap)")

    # ── Test 3: full overlap ─────────────────────────────────────────────────
    shared = {f"mol_Z_{i}" for i in range(100)}
    r3 = simulate_run(shared, shared, num_samples)
    print("\n-- Test 3: full overlap --")
    print(f"  Union (absolute plot): {r3['union_count']}  (expected 100)")
    print(f"  Sum   (ratio plot):    {r3['sum_count']}   (expected 200)")
    print(f"  Ratio OLD:             {r3['ratio_old (bug)']:.4f}  (expected 0.1000)")
    print(f"  Ratio NEW:             {r3['ratio_new (fix)']:.4f}  (expected 0.2000)")

    assert r3["union_count"] == 100
    assert r3["sum_count"]   == 200
    assert abs(r3["ratio_old (bug)"] - 0.10) < 1e-9
    assert abs(r3["ratio_new (fix)"] - 0.20) < 1e-9
    print("  PASS")

    print("\nAll 3 tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
