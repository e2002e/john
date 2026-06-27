# Spec: 0-order shell unranker for GPU OMEN-style candidate generation

## Goal
Implement a GPU-parallel password candidate generator that emits words in
**ascending probability order** (OMEN-style), where each work-item computes its
own candidate from a global index with **no sequential dependency**. Target:
OpenCL, integrating into an existing JtR OpenCL fork; hash is MD5; the design
metric is cracks-per-joule, not candidates/sec.

## Model (0-order / positional)
- Word length `len`. Each position `i` has a charset; the charset is sorted by
  probability (index 0 = most probable char).
- Each char gets an integer **level** (cost). Simplest: `level = rank`. Faithful
  OMEN: `level = round(-log P / quantum)` so common chars cluster at level 0.
  Store as `lvl[i][c]` (per-position, so positions may differ).
- A word's cost = **sum** of its per-position levels.
- A **shell** = all words with a given total level. Enumerating shell 0, then 1,
  then 2, ... is ascending probability order. Within a shell, order is irrelevant
  (the quantization treats a shell as equiprobable).

## Algorithm

### Host precompute 1 — count table `N` (build once)
`N[i][s]` = number of ways to fill positions `i..len-1` with levels summing to `s`.
```
N[len][0] = 1;  N[len][s>0] = 0
for i = len-1 down to 0:
    for each reachable s in N[i+1]:
        for c in 0..charset[i]-1:
            N[i][ s + lvl[i][c] ] += N[i+1][s]
```
Size is `len * (maxlevel+1)` integers — tiny, upload to constant memory.
`N[0][s]` is the size of shell `s`.

### Host precompute 2 — cumulative shell offsets
Prefix-sum the shell sizes: `cum[s] = number of words in all shells < s`.
This maps a global linear index `g` to `(shell, k)`:
find the largest `s` with `cum[s] <= g`, then rank-within-shell `k = g - cum[s]`.

### Device per-thread unrank — `(shell, k) -> word`
Pure table lookups, no recursion:
```
s = shell
for i = 0..len-1:
    for c = 0..charset[i]-1:           # chars in level order
        need = s - lvl[i][c]
        ways = (need >= 0) ? N[i+1][need] : 0
        if k < ways:
            word[i] = c;  s = need;  break
        k -= ways
```
Cost: `O(len * charset)` lookups per candidate — far below MD5's cost, so
generation never starves the hash cores.

### Device kernel flow
`thread_id -> (shell,k)  ->  unrank  ->  word  ->  hash`. No inter-thread
communication, no carry chain, no divergence beyond the bounded inner loop.

## Verified reference (Python — port this, use as oracle)
```python
import collections
def build(lvl, charset, L):
    N = [collections.defaultdict(int) for _ in range(L+1)]
    N[L][0] = 1
    for i in range(L-1, -1, -1):
        for s_next, cnt in N[i+1].items():
            for c in range(charset):
                N[i][s_next + lvl[i][c]] += cnt
    shells = sorted(N[0]); cum = {}; run = 0
    for s in shells: cum[s] = run; run += N[0][s]
    return N, shells, cum, run

def unrank(N, lvl, charset, L, shell, k):
    s = shell; word = []
    for i in range(L):
        for c in range(charset):
            need = s - lvl[i][c]
            ways = N[i+1].get(need, 0) if need >= 0 else 0
            if k < ways:
                word.append(c); s = need; break
            k -= ways
    return tuple(word)

def gen(N, shells, cum, lvl, charset, L, g):
    sh = shells[0]
    for s in shells:
        if cum[s] <= g: sh = s
        else: break
    return unrank(N, lvl, charset, L, sh, g - cum[sh])
```

## Acceptance tests
1. Over `g = 0 .. total-1`, all words distinct and count == charset**len
   (complete bijection).
2. Total level of `gen(g)` is non-decreasing in `g` (ascending shells).
3. GPU port matches the Python oracle word-for-word for every `g`.

## Design rationale (do NOT re-litigate these)
- **"It's slower" is wrong here.** Per-candidate cost is `O(len*charset)` table
  lookups, far below one MD5. Native OMEN enumeration is recursive/sequential and
  can't saturate the cores; this is fully parallel, so it is *faster* on GPU. The
  correct metric is expected candidates-to-hit x joules, not candidates/sec.
- **"It loses ordering" is wrong here.** The 0-order shell traversal *is*
  probability order (ascending total level). Ordering is supplied by walking
  shells in order; the unranker only orders within a shell, where order is moot.
- Ordering does NOT come from the diagonal sweep ("rain") itself — rain's native
  order is not probability order. It comes from the shell loop. The unranker
  replaces the sweep so the same diagonals are walked in ascending order AND in
  parallel.

## n-gram extension (later, if 0-order proves insufficient)
Same skeleton, but levels are conditional: `lvl[i][c | context]`. Index the count
table by `(position, context, remaining-level)` and carry the context through the
unrank loop. The unrank loop body is unchanged; only the table dimensionality grows.
Decide whether it's worth it by measuring cracks-per-joule against the 0-order
version on real targets first.
