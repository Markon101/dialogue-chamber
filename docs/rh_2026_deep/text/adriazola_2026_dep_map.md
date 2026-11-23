# Proof dependency map

> **Note, 22 November 2026.** The status marks below are those of 21 November 2026. The internal packages PC-1 to PC-7 that the marks *[reconstructed above PC-n]* refer to have since been closed at the human-proof boundary. See [`STATUS.md`](STATUS.md).

This is the argument of the archive in mathematical order, from the analytic criterion to the conclusion. It is a map of mathematics, not a history of the audit and not a list of Lean declarations. Lean names appear in `code` where they help a reader find the corresponding statement in the manuscript or the archive. Section references (§) are to [`RH_Route_Notes.pdf`](RH_Route_Notes.pdf).

Status marks used below.

- **[proved here]** a complete proof is given in the manuscript from the stated classical inputs.
- **[reconstructed]** a theorem of the archive whose proof has been read from the Lean text and written in ordinary mathematics down to classical statements or named published theorems.
- **[reconstructed above PC-n]** the same, down to the internal package PC-n of [`RECONSTRUCTION_STATUS.md`](RECONSTRUCTION_STATUS.md), whose proof bodies are still under audit.

Every named published theorem at which a branch stops is on the external boundary, [`EXTERNAL_THEOREMS.md`](EXTERNAL_THEOREMS.md).

## 1. The spine

```text
 1  analytic RH criterion                      one-sided bound on windows ⇒ zero-free half-plane
            ↓
 2  smoothed prime sums / explicit formula     G, G_χ, sampled sums, the inert-prime identity
            ↓
 3  arithmetic Stepanov mechanism              chain of m theta lifts, lattices N_u, seven inequalities (A1)–(A7)
            ↓
 4  theta-lift divisibility                    kernel law, valuation estimate at inert primes
            ↓
 5  local p-adic gain and Hecke loss           gain_j(p) on a sublattice; colength bound from LPprime
            ↓
 6  integral period map / LinkDIV              B_u : N_u → R^∨, its height, BesselId;
                                               LinkS → PuNZ → MIstar2 → BesselId → LinkDIV, FrameF
            ↓
 7  slope and covolume inequalities            FrameF (A3), covolume of R^∨ in (A4): HSAt, DeepStR, FUN
            ↓
 8  automorphic / Galois / local lifting       R4 = VWin + LinkN: automorphy of deformation rings,
    inputs                                     Wach modules, Theorems B and C, integral theta series
            ↓
 9  records and the host field                 seesaw identities, existence of records, HostEx / R5
            ↓
10  positive prime-sum combination             excess ⇒ G_Ξ(x) ≤ x^{1/2+Cε} on windows (R6)
            ↓
11  zero-free half-plane                       Re s > 1/2 + Cε for ζ and the L(s,χ), χ in Ξ, every ε
            ↓
12  RH / corresponding GRH conclusion          the final theorem of the archive
```

Stages 1 to 3 and 10 to 11 form the **analytic frame**, which the manuscript proves. Stages 4 to 9 form the **geometric core**, the internal theorems of the archive between the frame and the published theorems it quotes, which the manuscript reconstructs from the Lean text. The two meet at the interface statements R1 to R6 of §6.

## 2. The stages

### Stage 1. The analytic criterion [proved here]

For $x\ge1$ let $G(x)=\sum_{n\le x}\Lambda(n)(x/n-1)-x\log x+(1+\gamma)x$, and let $G_\chi$ be the analogue with $\Lambda(n)\operatorname{Re}\chi(n)$ and the linear term $\operatorname{Re}(L'(1,\chi)/L(1,\chi))\,x$. The Mellin transform of $G$ has a pole at each nontrivial zero of $\zeta$, and that of $G_\chi$ at each nontrivial zero of $L(s,\chi)$ and $L(s,\bar\chi)$.

- **Window theorem** (§2.3, Theorem `engine`, Corollary `corA`; Result 1). If a positive combination $G_+=G+\sum_iw_iG_{\chi_i}$ satisfies $G_+(x)\le Cx^\theta$ for all $x$ in windows $Y\le x\le Y^\lambda$ with $Y$ arbitrarily large, then $\zeta$ and every $L(s,\chi_i)$ are free of zeros in $\operatorname{Re}s>\theta$.
- Input: Landau's theorem on Dirichlet integrals of one sign, and classical properties of $\zeta$ and $L(s,\chi)$.

### Stage 2. Smoothed prime sums and the explicit formula [proved here]

- **Sampled sums** (§2.5, Propositions `samp`, `dil`). Versions of the criterion for sums sampled at finitely many points near $x$.
- **The inert-prime identity** (§5, Proposition `inert`; Result 3). For a prime $d>3$, $d\equiv3\pmod 4$, with quadratic character $\chi$, the sum over inert primes $2\sum_{\chi(p)=-1}\sum_{i\text{ odd}}\log p\,\max(a/p^i-w,0)$ plus the term at $d$ equals an explicit main term plus $w\,(G(a/w)-G_\chi(a/w))$. So a bound for the sum over primes of the route is a bound for $G-G_\chi$ at the points $a/w$.

### Stage 3. The arithmetic Stepanov mechanism [proved here, hypotheses from the core]

- **Model** (§3). Bombieri's form of Stepanov's method over a finite field, and Bost's slope inequality over $\mathbb Z$, which plays the role of the count of zeros against poles.
- **The chain** (§4.1). A chain of $m$ theta lifts between unitary groups of growing rank, over the integers of a CM field. To each index $u$ it attaches a hermitian lattice $N_u$ and a period pairing $B_u:N_u\to R^\vee$ into the dual of a lattice of forms.
- **Accounting** (Proposition `acct`; Result 2). Seven inequalities (A1) to (A7) among slopes, heights and local terms bound the sum $S$ over primes by its main term, with error shapes $x\ell/m$, $m\,x^{c\varepsilon}$ and $x\ell m/k$. The first two balance at $m=\sqrt x$, the Stepanov balance with the chain length as the multiplicity.
- The hypotheses (A2), (A3), (A4) are supplied by the core (Stages 4 to 8) through the interface statements R1 (`LinkMu`), R2 (`WordFacts`, contained in `LinkDIV`), R3 (sizes) and R4 (`VWin`, `LinkN`).

### Stage 4. Theta-lift divisibility [reconstructed]

(§4.5.) At a prime $p$ inert in $\mathbb Q(\sqrt{-d})$:

- **Kernel law.** The theta kernel of one lift is $\exp(-\Upsilon)$ times an integral series, where $\Upsilon$ is a sum of values of a pairing that is semilinear for the Frobenius of $\mathbb Q_{p^2}$. The pairing is the coboundary, for the formal group law, of the logarithm of the normalized sigma function of an elliptic curve with CM by $\mathbb Q(\sqrt{-d})$, whose formal group has height two at $p$. The integrality reduces to theta functions of rank one (`lubin_tate_transfer_integral`).
- **Normal form** (`BetaIsG`, `TwoSumsCondition`). In a parameter with logarithm $\sum_j x^{p^{2j}}/p^j$ every such pairing has an explicit normal form. Semilinearity forces odd shifts.
- **Valuation estimate** (`valest`). At a top of weight $\alpha$ the credit is $\alpha p/(p^2-1)=\sum_{i\text{ odd}}\alpha/p^i$, which matches the charge of an inert prime in Proposition `inert`.

### Stage 5. Local $p$-adic gain and Hecke loss [reconstructed]

- **Gain.** The composite of the lifts is divisible by $p^{\mathrm{gain}_j(p)}$ on a sublattice of the base lattice. The local model of the chain at an inert place (`SysTw`) and the passage from the model to the column map (`CmpV`) carry this divisibility to the period pairing and to `LinkDIV`.
- **Loss** (`LPprime`). A Hecke operator attached to a Frobenius orbit is nondegenerate up to a bounded power of $p$ on most of the rank. This bounds the colength of the sublattice. The archive proves it from standard monomial theory, Pieri and branching rules and the extremal projector.
- **Hypothesis (A2).** The colength bound and the degree formula give `SlopesN`, which is (A2). With `LinkG` and R1 (`LinkMu`) this closes the gain side.

### Stage 6. The integral period map and `LinkDIV` [reconstructed above PC-4, PC-5]

- **Height** (§4.6). $B_u$ is the composite of the $m$ lifts followed by a pairing with forms on a second unitary group. Its height is at most the sum of the logarithms of the norms of these maps (`height_chain_atR`), which gives the term $m\,\mathrm{ph}(a_t)$ of (A4). The constant $\log\Omega$ is a mean logarithmic period of the CM curve, given by the formula of Chowla and Selberg (`MeanPeriod`). Its term $L'(1,\chi_d)/L(1,\chi_d)$ cancels against the same term of Proposition `inert`.
- **Identification with a period** (`BesselId`), a condition of `LinkDIV`.
- **Corrected dependency order.** The pairing identity is established in the order

  ```text
  LinkS → PuNZ → MIstar2 → BesselId → LinkDIV, FrameF
  ```

  `LinkS` gives near orthogonality of Poincaré series on the unitary group, from the decay of Bessel transforms (Theorem P, with the radial identities and the quantum-determinant collapse) and from a pre-trace formula that counts discrete series (the counting engine of `LinkS` and `LinkIRR`). `PuNZ`, the nonvanishing $P_u\ne0$, comes from the diagonal dominance of the Gram matrix of Poincaré series. `MIstar2` is an identity of theta integrals obtained from seesaw identities. `BesselId` identifies $B_u$ with a period, and `LinkDIV` and `FrameF` use it. This order has no cycle. `BesselId` stays on the `LinkDIV` side.
- Remaining packages attached here: PC-5 (lattice counting below `LinkS` and `LinkIRR`) and PC-4 (finite-place harmonic analysis below `MIstar2`).

### Stage 7. Slope and covolume inequalities [reconstructed above PC-1, PC-2, PC-6]

- **Frame theorem, (A3)** (`FrameF`) [reconstructed]. Slope theory with Hilbert–Schmidt control of $Q=\sum_u\gamma_uB_uB_u^*$. Inputs are the Harder–Narasimhan filtration and Bost's slope inequality.
- **Slope of $R^\vee$, the second half of (A4)** (`L3`, `CovolumeBoundOfX`). Covolumes of lattices of cusp forms on unitary Shimura varieties, bounded by arithmetic intersection theory.
  - **Effective arithmetic Hilbert–Samuel inequality** (`HSAt`) [reconstructed]. A Chow-form determinant argument in the style of Nesterenko and Philippon, with Mahler heights, Hadamard's inequality and a monomial count. Inputs are Kawamata–Viehweg vanishing, Bertini and Chow forms.
  - **Uniform lower bound for mean volume slopes on deep covers** (`DeepStR`) [reconstructed above PC-1, PC-2]. Uniformity in the level comes from a prime cutoff that tends to infinity together with an exact local identity at good primes. With $w=\zeta_p-1$, $w^{p-1}=up$ and $\mathrm{Lie}\subseteq W\subseteq w^{-1}\mathrm{Lie}$, the identity reduces to $\mathrm{ck}(\mathrm{Lie},W)+\mathrm{ck}(\omega,\omega^{\mathrm{HT}})=h_\tau-\lambda_\tau/(p-1)$ and $p^kh_\tau=h_{\mathrm{Fr}\tau}+\lambda_{\mathrm{Fr}\tau}$, solved on finite Frobenius orbits by $h_\tau=\sum_{n\ge0}p^{-k(n+1)}\lambda_{\mathrm{Fr}^{n+1}\tau}$.
  - **Slope bound for the target lattice $R$** (`FUN`) [reconstructed above PC-6]. $\mu_{\max}(R)\le x^{C_H}b-\operatorname{mean}_v\log\kappa_v$, through denominators of zeta values and toric periods.

### Stage 8. Automorphic, Galois and local lifting inputs [reconstructed above PC-3, PC-7]

These give the two bounds of R4, which control the local terms of the lifts at places where their local type is twisted.

- **`VWin`, the Selmer bound** [reconstructed, small leaves remain]. Lengths of Tate–Shafarevich groups of the cyclotomic field are bounded by valuations of $\operatorname{tr}\rho(\sigma)-t$ at the automorphic points of a deformation ring of a residually reducible representation. This needs the automorphy of every such point, which the archive proves internally (`P3core`, `x1`, `red`, `V0ok`), with Greenberg–Wiles and explicit local conditions, Mazur's presentation, Bellaïche–Chenevier generalized matrix algebras and Chebotarev. Rank four has its own chain and does not involve the rank-three statement.
- **`LinkN`, the congruence-module bound** [reconstructed above PC-3, PC-7]. The congruence ideal of a Hecke summand is bounded by Selmer lengths through a universal cocycle, lifted to integral crystalline lattices with exponents from Wach modules.
  - torsion bound for crystalline extensions (`torsion_bound_of_weight_hyp`), with a constant logarithmic in the Hodge weights;
  - $\gamma$-descent along a generator of the cyclotomic Galois group (`coreWA`), with $\lfloor H/(l-1)\rfloor+v_l(\lfloor H/(l-1)\rfloor!)\le Hl/(l-1)^2$;
  - Wach realization through Berger's filtration conditions at $\zeta_{l^{s+1}}-1$;
  - **Theorems B and C**, two local lifting theorems for Wach modules with alternating Hodge jumps [reconstructed above PC-3], with case bounds at most $40\,\mathrm{Sg}$, $\sigma\le3\,\mathrm{Sg}$ for $l\ge11$, and the constant $32$ for Theorem C;
  - integrality of theta series of level structures (`series_bound`, `SerN`) [reconstructed above PC-7].

### Stage 9. The records and the host field [reconstructed above PC-4, PC-6]

- **Records** (§4.7). The lattices, heights and local terms of R1 and R2 are fields of a record, and the content of the core is the existence of a family of records. The archive builds them from `MIstar2`, matched local data, the covolume bound and R4.
- **Seesaw value** (`valX`). Six exact identities give $c_V(j)=\kappa\,\mathrm{sc}_j\,\mathrm{sc}_{j+1}\,c_Q\,c_G\,c_T$. The level scalars cancel, the twist-place laws reduce to one split place in the linear model of the Weil representation, and root exchange is used in type.
- **Integral structure** (`DescIntAt`, `exists_forms_of_transSum`). Galois descent of integral forms, and the theta kernel at CM points as a classical short theta series.
- **Host field** (R5, `HostExW`, `hostOfFieldOf`) [reconstructed]. The field $E$ of degree $4n$ with $x^{2+8\varepsilon}\le n\le2x^{2+8\varepsilon}$ and its auxiliary primes $q$, $d$. Nonvacuity goes through `HostEx`, whose nine analytic inputs are on the external boundary.
- **Auxiliary primes** (§8; Result 5) [proved here from (C1)–(C8)]. Most primes $q\equiv3\pmod8$ in a thin range give characters with $\lvert G_\chi(y)\rvert\le y^{1/2+\tau}$ on the window and a zero-free box, by a mean square over prime moduli and seven classical statements, without assuming any zero-free half-plane.

### Stage 10. The positive prime-sum combination [proved here, given R1–R5]

(§7; Result 4, Theorem `step5`, Corollary `asm`.) A bound for the excess of the sum over primes, averaged over translations by a subgroup of the Galois group, gives a bound for a positive combination $\sum_\psi c_\psi\operatorname{Re}G_\psi(x)$ over a family of characters. With the sizes of R3 and the auxiliary characters of Stage 9 this is R6,

$$G_\Xi(x)\le x^{1/2+C\varepsilon}\quad\text{on the windows of R5,}$$

for a finite set $\Xi$ of primitive characters containing the trivial character.

### Stage 11. The zero-free half-plane [proved here]

Corollary `corA` applied to $G_\Xi$ with $\theta=\tfrac12+C\varepsilon$ gives that $\zeta$ and $L(s,\chi)$, $\chi\in\Xi$, have no zeros with $\operatorname{Re}s>\tfrac12+C\varepsilon$. Since $\varepsilon$ is arbitrary, the half-plane is $\operatorname{Re}s>\tfrac12$.

### Stage 12. The conclusion of the archive

The final theorem of the archive is `theorem riemannHypothesis : RiemannHypothesis`, stated with Mathlib's definition, together with statements on the zeros of Dirichlet $L$-functions (see [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md)). This repository reconstructs the route to that statement. It does not assert the statement.

## 3. Where the current audit boundary meets the map

```text
 stage                               internal package under audit
 ─────────────────────────────────   ──────────────────────────────────────────────────────────
 1–3   analytic frame                none (proved here)
 4     theta-lift divisibility       none
 5     gain and loss                 none
 6     period map, LinkS → … LinkDIV PC-5 lattice counting (below LinkS, LinkIRR)
                                     PC-4 finite-place seesaw (below MIstar2)
                                     SL-3 small frame lemmas, model algebra (Theorem P side)
 7     slope and covolume            PC-1 local Hodge–Tate / finite flat (below DeepStR's exact identity)
                                     PC-2 global tower and covolume (below DeepStR, L3)
                                     PC-6 record construction (FUN, below L3 and LinkDIV)
                                     SL-1 flag and test leaves of DeepStR
 8     automorphic / Galois / local  PC-3 shared Wach-module layer of Theorems B and C (below LinkN)
                                     PC-7 FR-10 markings and integrality producers (below SerN → LinkN)
                                     SL-2 rank-four leaves; SL-4 FR-01/02, FR-18, FR-19 leaves
 9     records and host              PC-4, PC-6, PC-7 as above; SL-5 bookkeeping
                                     (auxiliary primes: proved here; small bookkeeping leaves at the host)
10–11  combination, half-plane       none (proved here)
```

Downstream chains of the packages, as recorded by the audit.

- PC-1: `FamilyExistsPos` → `HodgeLatticeFlagBound` → `Prop6` → level uniformity of `DeepStR` → `HeightAt` → `UAt` → `lowAt_of_cover` → `L3` → `LinkDIV`.
- PC-2: tower axioms of `DeepStR`; fields of `topCovolG_alt_alt` → `L3` → `LinkDIV`.
- PC-3: Theorems B and C → `LengthBoundAtPlaces` → `LinkN`.
- PC-4: `valX`, $c_V$, `ClsPinD` → `KerX`, `PerX`, `RdX` → `MIstar2` → `BesselId` → `LinkDIV`, `FrameF`.
- PC-5: `MainTheorem`, `KVIdentity`, row counts, the dimension clause of `LinkS`, `LinkIRR`.
- PC-6: `HostExW`, `CBHyp`, `adm2`, `DIVinN`; `FUN` → `L3`, `LinkDIV`.
- PC-7: `SerN` → crystalline lattices → `LinkN`.

Every remaining package lies below a reconstructed mechanism and above named published mathematics. None of them sits in the analytic frame. The external boundary, which is separate from these packages, is in [`EXTERNAL_THEOREMS.md`](EXTERNAL_THEOREMS.md).

---

# External theorems: the verification boundary

> **Note, 22 November 2026.** The comparison status below is that of 21 November 2026, before the source-fidelity audit began. That audit has since matched, corrected or reclassified part of this boundary, but not all of it. See [`STATUS.md`](STATUS.md), Section 5.

This file lists the named published or classical theorems at which the reconstruction stops. It is drawn from [`audit/Post_Closure_Verification_Boundary.md`](audit/Post_Closure_Verification_Boundary.md), which is the authoritative and more detailed list. Internal theorems of the archive whose proofs are still unread are a separate list, in [`RECONSTRUCTION_STATUS.md`](RECONSTRUCTION_STATUS.md). The two lists are disjoint.

**No fidelity check against the literature has yet been made for any entry.** Every entry still needs a comparison in both directions. The archive's statement must not be stronger than the published theorem, and the published hypotheses must hold where the archive applies it.

## 1. How to read the table

Column *Hypotheses matched?* records whether anyone has compared the archive's statement and its use with the published hypotheses, and "not recorded" means that the audit recorded no comparison. Column *Verification status* uses the following terms.

| Term | Meaning |
|---|---|
| `source match pending` | The exact published source has not been located or opened. The attribution was recorded from memory or through a secondary source. These entries most need a literature match. |
| `not compared` | A source is named, and the statement has not been compared with it. |
| `partially compared` | A specific feature (a constant, a normalization, a hypothesis) has been checked, and the full comparison has not been made. |
| `proved in archive` | The archive contains its own Lean proof of a statement of published type. The reconstruction stops at the published statement. The archive's proof is a separate task (Layer 3), and the comparison with print is still pending. |
| `classical` | A standard statement (textbook level). It is still on the boundary, and its exact form should be confirmed. |

Some inputs enter the archive through `instance : Fact _` declarations (54 in the archive). The final theorem has no instance binders, so each is discharged by a term, but a `Fact` hides the input from the statement that uses it. Locating the term for each instance on the route is part of the verification (V16).

## 2. Priority checks

| # | External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|---|
| V1 | Primes in progressions with relative error, (C7) = `PNTAPRelW` | Gallagher, Invent. Math. 11 (1970), Thm 7, at level $\max(M,e^{\sqrt{\log X}})$ with partial summation. The form was taken from Thorner–Zaman (arXiv:2108.10878, (3.3)–(3.4)) and arXiv:2508.16400, Lemma 7.3 | Second stage of the auxiliary primes (§8) | No. Derivation written, not compared line by line with Gallagher, including the two exceptional-zero windows | `source match pending` (Gallagher's paper not opened) |
| V2 | Prime ideal theorem for ray classes of $\mathbb Q(\sqrt{-d})$ in the Landau–Page range, `ShortNorm` | Hecke–Landau; Davenport ch. 20 for $\mathbb Q$; Fogels, Lagarias–Montgomery–Odlyzko, Weiss, Thorner–Zaman | Primes $p=\pi\bar\pi$, $\pi\equiv1\pmod M$, for the host field; used with $h=1$, $\log B\le10(\log Y)^{2/5}$ | No | `source match pending` (citation from memory); proved in archive by an elementary convolution |
| V3 | `GdEndAlt` and `SmallClassNumbers.Bound` against Skinner–Wiles, Thorne, Allen–Newton–Thorne, and Soulé | Skinner–Wiles; Thorne; Allen–Newton–Thorne; Soulé | Weakened Vandiver statement; residually reducible automorphy | No. The test against $K_4(\mathbb Z)$, $K_8(\mathbb Z)$ was not run. These are comparisons: the archive proves `GdEndAlt` internally (Section 4) | `not compared` |
| V4 | Kato's theorem in the range used, with `exists_admissible_hecke_projection_nonvanishing` | Kato; Rubin | Nonvanishing $L$-value in the Selmer bound | No. The $L$-value must meet Kato's hypotheses | `not compared`; proved in archive on the operative branch |
| V5 | Siegel–Weil formula, `SiegelWeilStd`, `EisBL` | Siegel–Weil; Gan–Takeda §7.1 | Theta tower, records | Partially. Constant matched with Gan–Takeda §7.1. Tamagawa number 2, probability Haar measure and holomorphy at $s_0=(m-n)/2$ to confirm | `partially compared` |
| V6 | Siegel–Weil in the convergent range, two declarations `SiegelWeilUnitaryConvergent` | Weil; Ichino | `RowDiag` (Fourier-coefficient form, $2n<M$, $n=1$); doubling form | No. The doubling form needs its own check | `not compared` |
| V7 | Published bundles of the covolume branch: `PublishedHodgeBundleTheorems` (28 fields), `HSCore.Quoted` (8), the eleven statements under `deepStR_of_use`, `GrothendieckDeformation`, `SiegelToroidalIntegral`, `ToroidalForFanKS`, `SemiAbelianLieSeminegative` | Kawamata–Viehweg vanishing and the other fields of `HSCore.Quoted`; Nadel vanishing, principalisation, Hirzebruch–Mumford, Faltings–Chai, Lan, Grothendieck–Messing as fields of `PublishedHodgeBundleTheorems` | Covolume bound (`HSAt`, `topCovolG_alt_alt`, PC-2) | No. Several fields not unfolded. This is the largest published bundle | `not compared` |
| V8 | Wach and Berger clauses (`p3`, `p4` with rational converse, `p5`, `p9`, `p10`, `SATcore`, `F1`, `F2`, $\Delta$-invariant forms) | Wach (1996); Berger (2004) | Relay of Theorem B, half of `ArcCore`, `LinkN` | Partially. Ring relations of `p9` ($\pi_0(\zeta_p-1)=-p$, $q=\varepsilon q_A$) checked classically | `proved in archive`; `partially compared` |
| V9 | Local class field theory, `LocAb` / `LocCFT`: $\sum_v d'_v=[L^+:\mathbb Q]$ | Local class field theory | Exponent $D_0$ of FR-01 (cancels exactly); rank four through `b1Ar` | No | `not compared` |
| V10 | Standard monomial theory in exact integral form; extremal projector constants | Standard monomial theory; Asherova–Smirnov–Tolstoy, Zhelobenko | `LPprime` (loss side) | Classical in substance; constants unchecked | `proved in archive`; constants `not compared` |
| V11 | Slope bound $\mu_{\max}(\mathrm{Lie})\le C_s\log(2+\mathrm{rk}+\max(-\deg,0))$, `SlopeBoundSt` | Candidate sources Bost, Graftieaux, Gaudron–Rémond | Constant $C_h$ of `DeepStR` | No. Printed source not matched | `source match pending`; proved in archive |
| V12 | Route-level leaves of `deep_slim_use` | Hironaka (resolution, principalisation in characteristic zero); inverse Poincaré–Lelong; `HSUse`; `MoriwakiBigStatement`; Fargues (`FarguesExact`, `FarguesHodgeTate`); Zarhin's trick (`ZarhinStatement`); `NefRR` | `DeepStR`; `FarguesHodgeTate` and `ZarhinStatement` are the stopping points of PC-1 | No | `not compared` |
| V13 | Rallis's constant-term formula for theta integrals in the convergent range; `SiegelSetUnitary` | Rallis (1984); Kudla–Rallis | `GenBig`, `GenAniso`, `GenSmall` of the theta tower; `WeilConv`, `KDecay` | No | `not compared` |
| V14 | Weighted integrability of discrete-series matrix coefficients, `CoefWeightedLoc` | Harish-Chandra; Trombi–Varadarajan type | `Wu_spec` (through a `Fact` instance), counting engine | No | `not compared` |
| V15 | Thorne's localized $R=\mathbb T$ at nice points, $n=3,4$, clause `P1core` | Thorne | Rank four, at every link through `b1Ar`, `b2Ar`, `raw`, `N4` | No | `not compared` |
| V16 | The 54 `Fact` instances | (various) | Throughout | Term for each instance on the route not yet located | open |
| V17 | Nine analytic inputs of `hostEx_of_zeroFreeRegion`: explicit formulae (`ExplZeta`, `ExplChi`), `LogDerOne`, zero counting (`ZeroCount`), zero-free region for Dirichlet $L$-functions (`ZFRDirichlet`), `TZRat`, Page's theorem (`PageThm`), Jutila's zero density $N(\alpha,Q,T)\ll(Q^2T)^{(2+\varepsilon)(1-\alpha)}$ for $\alpha\ge4/5$, `QuadraticHeckeZeroEstimates`, and `RayField` | Page; Jutila; standard explicit-formula and zero-free-region literature | Nonemptiness of the records through `HostEx` (live route) | No | `source match pending` (attribution and year of Jutila from memory) |
| V18 | Prime number theorem for primitive characters in the form of `psi_thin`, $\lvert\psi_0(u,\chi)\rvert\le Cu(\log u+\lambda)^2\exp(-c\log u/(\lambda+\sqrt{\log u}))$ outside an exceptional zero | de la Vallée Poussin's zero-free region; truncated explicit formula; Davenport ch. 14, 17, 19, 20 | `QH.thinH`, host field | No | `source match pending` (cited from memory); proved in archive |
| V19 | Root exchange for $\mathrm{GL}_n$ over a nonarchimedean local field; linear-model seesaw | Ginzburg–Rallis–Soudry, *The descent map from automorphic representations of GL(n) to classical groups* (2011), Ch. 7; Kudla, *Seesaw dual reductive pairs* (1984) and *Notes on the local theta correspondence* (1996), §§1–3 | `exchange_isPos`; candidate stopping point for the PC-4 leaf `exchange_support` | No. The archive proves root exchange internally (`RootEx`) | `not compared` |
| V20 | Raynaud's classification of $\mathbb F$-vector space schemes with Fargues's integral Hodge–Tate sequence | Raynaud, *Schémas en groupes de type (p, …, p)* (1974); Fargues | Candidate stopping point for `testFacts` (PC-1). Not an input of the archive today | No | candidate; `not compared` |

## 3. The boundary by family

### Analytic frame, auxiliary primes and host field

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Landau's theorem; functional equations; (C1)–(C6), (C8) | Landau; as cited in the manuscript | Window criterion, auxiliary primes (§§2, 8) | Quoted in the form used; (C2), (C4), (C5) not compared | `not compared` |
| Gallagher's prime number theorem, (C7) | Gallagher (1970) | §8 | See V1 | `source match pending` |
| Landau–Page prime ideal theorem for ray classes (`ShortNorm`, `NormPrimes`, `LevelPrime`) | Hecke–Landau; Page; see V2 | Host field | See V2 | `source match pending` |
| Linnik's theorem (`SmallOne`, via `primesRel_holds`) | Linnik (1944) | Least prime $\equiv1\bmod M$ at most $M^A$, relative form for $X\ge M^{c_2}$ | No | `not compared` |
| Chebyshev $\psi-\vartheta\ll\sqrt X$; Dirichlet's theorem | classical | §8 | not recorded | `classical` |
| Prime number theorem for primitive characters; analytic inputs of `HostEx` | see V17, V18 | Host, records | See V17, V18 | `source match pending` |
| $\int_1^\infty(\psi(u)-u)u^{-2}\,du=-1-\gamma$ | from the Laurent expansion of $-\zeta'/\zeta$ at $s=1$ | `g_integral` | not recorded | `classical`; the archive proves `l21a` |
| $L(1,\chi)\ne0$ and continuity at $s=1$; Dirichlet character theory; quadratic reciprocity; Galois theory of $\mathbb Q(\zeta_N)$; quadratic Gauss sums | classical; Mathlib | `QH.small_zeros`, host moduli | Spot-checked | `classical` |
| Behrend's construction | Behrend (1946) | `thickFrame2`, `hostThickExists` | Proved in the manuscript; key identity checked symbolically | proved in manuscript |

### Gain and loss (theta-lift divisibility, `LPprime`)

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Global Artin reciprocity, ray class fields, main theorem of CM; good models of CM elliptic curves at $l\ge5$, $l\nmid d$ | classical CM theory | Gain at inert primes (`CM2`, FR-19) | Use read | `not compared` |
| Weak approximation; Jacobowitz's classification at unramified and inert primes | Jacobowitz | `intInert_of` | No | `not compared` |
| Standard monomial theory, Pieri, branching, Weyl dimension formula; extremal projector; straightening law (Doubilet–Rota–Stein, De Concini–Eisenbud–Procesi) | as named | `LPprime`, `adm7` | Constants unchecked (V10) | `proved in archive` / `not compared` |

### Slopes, frame theorem and covolumes

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Harder–Narasimhan theory; slope inequality | Stuhler, Grayson, Bost, Gaudron | `FrameF`, (A3) | Hand proof in the audit | `not compared` |
| Kawamata–Viehweg vanishing; Bertini for simple zeros of generic sections over number fields; Chow forms | as named | `HSAt` (`HSCore.Quoted`) | See V7 | `not compared` |
| Chowla–Selberg formula; Kronecker limit formula | Chowla–Selberg (1967) | Mean period in (A4) | Normalization consistent; Colmez conjecture not used | `proved in archive`; `partially compared` |
| Fargues's integral Hodge–Tate sequence; Zarhin's trick; quotients by finite flat group schemes | Fargues; Zarhin | `familyExistsPos_holds`, PC-1 | See V12, V20 | `not compared` |
| Raynaud's classification of $\mathbb F$-vector space schemes | Raynaud (1974) | Candidate for PC-1 | See V20 | candidate |
| Layer 3 statements under `DeepStR`: arithmetic intersection theory (Gillet–Soulé, Burgos–Kramer–Kühn type), Zhang's successive minima, arithmetic Hilbert–Samuel bounds, Moret-Bailly's key formula, toroidal compactifications and integral models, strong approximation, Hirzebruch–Mumford proportionality, heights of flag varieties | as named | `DeepStR`, PC-2 | No | `proved in archive` |
| Slope bound of Bost type | Bost, Graftieaux, Gaudron–Rémond | `DeepStR` | See V11 | `source match pending` |
| Mertens-type bound $\sum_{p\le B}\log p/(p-1)\le\log B+3.4$ | elementary | `sum_log_div_pred_SB` | Checked numerically | `classical` |
| Ray class fields; product formula; Mahler measure; Hadamard; Schwartz–Zippel; Dedekind's complementary module | classical | Covolume bound | not recorded | `classical` |

### `LinkS`, `LinkIRR`, Theorem P and discrete-series counting

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Quantum minors, Capelli's theorem, centrality of the quantum determinant | Molev, *Yangians and Classical Lie Algebras*, ch. 1, 7 | Theorem P, radial identities | Identities confirmed numerically in nine low-rank cases; RLL relation and quantum determinant checked symbolically | `partially compared` |
| Cauchy–Binet; Laplace expansion; multivariate Lagrange interpolation; A. A. Markov's inequality; Cauchy's estimate; Fatou; dominated convergence | classical | Theorem P | not recorded | `classical` |
| Siegel–Weil in the convergent range | Weil; Ichino | `RowDiag` → `PuNZ` | See V6 | `not compared` |
| Siegel mass formula in ratio form; Siegel mean value theorem; mean counts of primitive sublattices over a genus | Siegel, Ann. of Math. 46 (1945); W. M. Schmidt, Duke Math. J. 35 (1968) | Counting engine, PC-5 | No | `not compared` |
| Schmid's characterisation of discrete series; Harish-Chandra (square integrability, formal degree, finiteness); Borel–Harish-Chandra; Schur orthogonality; weighted integrability | Schmid; Harish-Chandra; see V14 | Discrete-series counting | No | `not compared` |
| Landherr with Jacobowitz; rank-one class number formula; Louboutin; Minkowski | as named | Lattice and volume bounds | Louboutin: archive form follows | `proved in archive` / `partially compared` |
| Green's characters of $\mathrm{GL}_r(\mathbb F_g)$; Gelfand–Graev multiplicity one; degenerate Whittaker models (Zelevinsky, Gelfand–Kazhdan); uniqueness of Bessel models (Jiang–Sun–Zhu, Sun–Zhu) | as named | Bessel periods | No | `not compared` |

### Automorphy and Galois cohomology (`VWin`)

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Greenberg–Wiles; local Tate duality; local and global Euler characteristic; Poitou–Tate | Greenberg–Wiles; Tate | `x1` ($h^2+h^0+2d\le h^1$), `CohP` | No | `not compared` |
| Mazur's presentation lemma; Noetherian finiteness under Mazur's $\Phi_l$ condition | Mazur | `obs_pres`, `Det.matFg` | not recorded | `proved in archive` / `classical` |
| Chebotarev density theorem; Brauer–Nesbitt | classical | `cheb_holds`, `V0ok` | not recorded | `classical` |
| Bellaïche–Chenevier generalized matrix algebras | Bellaïche–Chenevier | Residually multiplicity-free Cayley–Hamilton algebra | Hypotheses match (audit) | `partially compared` |
| Taylor–Wiles–Kisin patching | Taylor–Wiles; Kisin | `pmodEx` | not recorded | `proved in archive` (through `litN`) |
| Hida theory for definite unitary groups, $2\le n\le4$ | Hida; Geraghty | Automorphy | not recorded | `proved in archive` |
| Thorne's localized $R=\mathbb T$ | Thorne | `P1core` | See V15 | `not compared` |
| Kato's Euler system; Rubin | Kato; Rubin | See V4 | See V4 | `proved in archive` |
| Adjoint Selmer vanishing | Allen; Newton–Thorne | `AdjointLiftingTheorem` | not recorded | `proved in archive` |
| Skinner–Wiles, Thorne, Allen–Newton–Thorne (residually reducible automorphy) | as named | Comparison only for `GdEndAlt` | See V3 | comparison, not an input |
| Soulé; Dwyer–Friedlander; Borel; Clozel | as named | Exponent $l^{c(2e)}$, $c(m)=m(m+5)/2$ | Archive weaker than Soulé (safe direction) | `partially compared` |
| Ferrero–Washington | Ferrero–Washington | $\mu=0$ behind `Bounds` in `x2` | Field abelian over $\mathbb Q$ | `partially compared` |
| Mazur–Wiles, Rubin; Stickelberger; Herbrand-type triviality | as named | Kolyvagin system of Gauss sums; `CorX` | No | `not compared` |
| Automorphic induction and cyclic base change (Arthur–Clozel); Labesse; Ramakrishnan's $\mathrm{GL}_2\times\mathrm{GL}_2\to\mathrm{GL}_4$; Deligne–Serre; Gelbart–Jacquet | as named | Seeds, descent | No | `not compared` |
| Galois representations of RACSDC representations, local–global compatibility, purity | Harris–Taylor, Shin, Chenevier–Harris, Caraiani; Clozel | Clauses `pur`, `aut` | No | `not compared` |
| Krull, Nakayama, prime avoidance; Grothendieck connectedness (SGA 2, XIII); Serre's multiplicities | classical | `redDim`, `clChain`, `rulePos` | not recorded | `classical` / `proved in archive` |

### Local $p$-adic Hodge theory (`LinkN`, Theorems B and C)

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Tate's results on $H^1(H_K,\mathcal O_C)$ and the normalized decomposition; Ax–Sen–Tate | Tate (1967) | Torsion bound, Theorem TB (a) | Explicit-constant versions derived | `proved in archive` |
| Hodge–Tate decomposition of de Rham representations; $\mathrm{gr}^iB_{\mathrm{dR}}^+\cong\mathbb C_p(i)$ | Sen (1980); Fontaine (1994) | Theorem TB | not recorded | `proved in archive` |
| Tate's filtered invariants; Fontaine's finiteness and de Rham property | Tate; Fontaine | Step A of Theorem TB (a) | No | `not compared` |
| Bloch–Kato, Cor. 3.8.4; dual exponential; crystalline extension criterion | Bloch–Kato (1990) | $H^1_f$, `exponent_le`, Theorem C record | No | `not compared` |
| Local Tate duality; local Euler characteristic | Tate (1962) | Theorems B and C producers | No | `not compared` |
| Prismatic F-crystals for crystalline lattices | Bhatt–Scholze (2023), Thm 1.2 | Crystalline lattices | No | `not compared` |
| Wach lattices; specialization at $\zeta_{p^n}-1$; level maps and Hodge filtration; Berger's membership criterion | Wach (1996); Berger (2004) | See V8; candidate stopping points for PC-3 | See V8 | `proved in archive` |
| Herr complex; class formula; naturality | Herr (1998); Benois (2000); Berger (2003) | `local_lemmas_of_literature`, Theorem C | not recorded | `proved in archive` |
| Exactness and strictness of $D_{\mathrm{cris}}$; $D_{\mathrm{dR}}=D_{\mathrm{cris}}\otimes K$; full faithfulness and exactness of $D$ | Fontaine (1990, 1994) | `SplitCore`, Theorem C | not recorded | `classical` |
| Wach module as value of the prismatic F-crystal | Abhinandan (2023); Du–Liu–Moon–Shimizu (2022) | Wach realization | not recorded | the archive reproves it |
| Lifting-the-exponent lemma; Legendre; Kummer; Weierstrass preparation in $\mathcal O[[\pi]]$; $\mathcal O[[\pi]]^\Delta=\mathcal O[[\pi_0]]$ | classical | Ring relations, case exponents | Proofs written in the audit | `classical` |
| CM theory for the Weber function; Shimura; Deligne | as named | `mul_twist_mem_intForms` | No | `not compared` |
| Local Langlands for $\mathrm{GL}_n$; Bernstein–Zelevinsky; Moy–Prasad | as named | Classification, genericity, depth zero | No | `not compared` |
| Class field theory (characters with prescribed local components) | classical | `LinkN` | not recorded | `classical` |

### Theta correspondence, seesaw and records

| External input | Source / authors | Used in | Hypotheses matched? | Verification status |
|---|---|---|---|---|
| Siegel–Weil, `SiegelWeilStd`, `EisBL` | see V5 | Theta tower | Constant matched | `partially compared` |
| Rallis's constant-term formula; Kudla–Rallis | Rallis (1984); Kudla–Rallis | See V13 | No | `not compared` |
| Rallis inner product formula; doubling method | Rallis | `AForm`, `BForm`, `LamLow` (FR-14); `nonZeroBottom_of_doubling_integral` | No | `not compared` |
| Yamana | not named | The boundary does not name Yamana. Whether the doubling-form check V6 needs his results has not been determined | not recorded | not on the boundary |
| $(\mathrm{GL}_a,\mathrm{GL}_b)$ Howe duality on $\mathbb C[M_{b\times a}]$; multiplicity-one branching $U(n+1)\downarrow U(n)$; irreducibility of Weyl modules; Bernstein–Zelevinsky geometric lemma; Kudla's filtration of the Jacquet module | as named | `KerX`, `PerX`, `finRam`, `finSplit` | not recorded | `proved in archive` |
| Howe duality (Howe, Waldspurger, Mínguez, Gan–Takeda); unramified Howe duality at inert places | as named | `howeInert` | No | `not compared` |
| Root exchange; Kudla's linear-model seesaw | Ginzburg–Rallis–Soudry; Kudla | See V19 | No | `not compared` |
| Haar measure on quotients and Fubini over places | A. Weil, *L'intégration dans les groupes topologiques*, ch. II | `collapse_factor`, `iterated_integral_product_groups` | Haar pair at $Q$ normalized by $\mathrm{vol}(K_Q)=1$ | `not compared` |
| Adelic and classical theta dictionary | A. Weil, Acta Math. 111 (1964) | `thetaKO_transSum_eq_short_series`, `kernel_identity_tuple` | Read at statement level | `not compared` |
| Projective lifting; density of a global lattice in its completion | classical; Mathlib | `AdMark.exists_marking` | not recorded | `classical` |
| Level structures on PEL moduli and lifting through level covers | Kottwitz, J. AMS 5 (1992), §5; Lan (2013), ch. 1 | `Dict.MarkLift.exists_lift_marked` (PC-7) | No | `not compared` |
| Hasse's norm theorem; orthonormal integral bases of unramified extensions (Serre, *Corps locaux*, I §6); Galois descent | classical | `IsUnfEx`, `DescIntAt` | not recorded | `classical` |
| `SiegelMassRatioCM`, `RayField`, `PrincipalIdealSt`, `CMCurveGoodOutsidePrimeSt`; the thirteen hypotheses of `miStar2_of_allTrue_alt_alt` | as stated | Record nonemptiness, `window_lower`, `MIstar2` | No | `not compared` |
| Tate's thesis; Hecke $L$ nonvanishing at 1 | Tate | Functional equations | Local statements are closed terms | `classical` |

## 4. Not inputs

These are comparisons or internal theorems. They should not be cited as places where the route stops.

- Skinner–Wiles, Thorne, Allen–Newton–Thorne as a source for residually reducible automorphy lifting. `GdEndAlt`, `SymplecticFourImageDense` and `ThmS6Core_s` are theorems of the archive (V3).
- The arithmetic Hilbert–Samuel formula `q9`. The explicit inequality `HSAt` is internal and reconstructed.
- The Capelli identity and the RTT presentation as models for the radial and model-algebra identities.
- The Siegel–Walfisz range. None of `ShortNorm`, (C7), `SmallOne` uses it.
- Raynaud's classification, until V20 is settled.
- `KillerLev`, `KillerKap` (internal theorems of PC-3) and `CVEq` (a definition).

## 5. Separate campaigns

- The archive's own Lean proofs of published theorems (Layer 3), including `litN`, the explicit-constant versions behind Theorem TB (a), the archive's proof of V18 and the internal `RootEx` proof of root exchange.
- Refereeing of Sections 7 to 9 of the manuscript.
