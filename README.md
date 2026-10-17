# Cognigraph: Probabilistic Constraint Graphs for Fault Localisation

**22AIE301 Probabilistic Reasoning, course project.** A Bayesian fault localiser for small Python programs. Given a function and a test suite it computes, for every statement, the **posterior probability that the statement is faulty**, handles **several simultaneous faults whose number is unknown**, and says how sure it is. Inference is exact; every parameter is learned from labelled single-fault programs; every result in this file comes from a saved, checksummed result folder in this repository.

> **Status in one paragraph.** On 362 fresh MBPP programs, with a protocol fixed before the model was run, the model ranks two-fault programs better than every one of five strong baselines (over the best, MUSE: +0.030 expected average precision, 95% interval [+0.012, +0.046], Holm-corrected p = 0.004). The gain is **small**, does **not** extend to three faults, and the probabilities fall below the base rate at three faults. The study uses injected faults, not naturally occurring multi-fault bugs, and no language model. Negative results are kept in the repository and listed below.

**Contents**
1. [Problem statement](#1-problem-statement)
2. [Data](#2-data)
3. [Probability background](#3-probability-background)
4. [The mathematics of the model](#4-the-mathematics-of-the-model)
5. [Models and baselines](#5-models-and-baselines)
6. [Experimental design and evaluation mathematics](#6-experimental-design-and-evaluation-mathematics)
7. [Results](#7-results)
8. [Negative results](#8-negative-results)
9. [Limitations](#9-limitations)
10. [Future scope](#10-future-scope)
11. [Install, run, reproduce](#11-install-run-reproduce)
12. [Repository layout](#12-repository-layout)
13. [Documents](#13-documents)
14. [References](#14-references)

---

## 1. Problem statement

A program is run against a test suite and some tests fail. A failing test does not say which line is wrong: it says only that *some statement it executed is probably wrong*. **Fault localisation** ranks statements by how suspicious they are.

Why this is a probability problem:
* The evidence is indirect and noisy (a fault may sit behind a rarely taken branch and be hidden by easy test inputs).
* Statements compete to explain the same failure, so beliefs about different lines are dependent (*explaining away*).
* A useful answer is not only a ranking but a *calibrated probability*, including "there may be no fault" and "there may be several".

**Existing approaches and their limits**

| Family | Examples | Limit |
|---|---|---|
| Spectrum-based | Ochiai, Tarantula | formulas over coverage; one line at a time; not probabilities |
| Bayesian spectrum | Barinel | coverage only; maximises the failure parameter per candidate |
| Mutation-based | Metallaxis, MUSE | formulas over what an edit does to the tests; designed for one fault |
| Learned | logistic regression on features | black box; probabilities transfer poorly across programs |

**The research gap this project addresses.** Bayesian localisers that use the outcomes of *program variants* (single-edit mutants or patches) as evidence assume **one fault**. With two faults, the exact repair of one fault leaves the tests broken by the other fault still failing, and a single-fault model reads that as proof of *innocence*. We model each variant as an **intervention** and derive how likely a repair is to fix a failing test **as a function of how many other faulty statements that test executes**. Parameters are estimated from single-fault programs only, so the multi-fault behaviour of the model is a prediction rather than something fitted. The novelty claim is deliberately narrow and search-limited (about 30 web searches; an earlier pass missed two closely related papers): see [research/RELATED_WORK_AND_NOVELTY.md](research/RELATED_WORK_AND_NOVELTY.md).

**Scope.** In: single Python functions of 5 to 30 statements, 1 to 3 simultaneous faults, test-suite and single-edit-variant evidence, exact inference, calibration. Out: project-scale programs, naturally occurring multi-fault bug data, language-model evaluation, comparison with SmartFL or Prosecutor (Java, project scale).

**Research questions, with their answers**
1. Can mutation evidence be used in a Bayesian localiser when the number of faults is unknown? **Partly:** yes at one and two faults; not at three.
2. Does it beat strong baselines? **At two faults on fresh data, yes, by a small, statistically solid margin; at three faults no.**
3. Are the probabilities trustworthy? **At one and two faults better than the alternatives; at three faults worse than a constant guess.**
4. Does it work on human-written bugs? **Weakly:** when no single edit repairs the bug, it approaches the coverage-only model.

---

## 2. Data

Nothing is downloaded at run time: all sources are stored in `data/external/` with source URL, commit or version, licence and SHA-256 (see [data/external/README.md](data/external/README.md)). **Parameters of the proposed model are fitted on the project's own development set (D1) only; every other set is held out.**

### 2.1 Overview

| Set | Role | Programs | Faults / cases | Fault origin | Tests per program |
|---|---|---|---|---|---|
| **D1** (project benchmark, `pcg/benchmark_programs.py`) | **development**: parameters, model design | 34 | 594 detected single faults; 410 two-fault programs | injected (10 operators) | 7 to 16 hand-written (mean 10.1) |
| **QuixBugs** | held-out; human bugs | 31 | 31 human bugs; injected sets with 1, 2, 3 faults | human-written and injected | JSON tests of the benchmark (231 of 242 kept) |
| **HumanEvalPack / HumanEvalFix** (Python) | held-out; human bugs | 164 problems; 147 usable, 139 with injected-fault sets | **146 human-written bugs**; injected sets with 1, 2, 3 faults | human-written and injected | the benchmark's own tests |
| **MBPP** | **confirmatory**; fresh, never used to build the model | **362** | 1, 2, 3 injected faults (4, 4, 2 versions per program) | injected | about 12.6 after amplification |

Programs used in the version-4 held-out study: 3,622 (cases). Statistical unit everywhere: the **program**.

### 2.2 D1: the project's development benchmark
* 34 small Python functions (`binary_search`, `gcd`, `is_prime`, `merge_sorted`, `classify_triangle`, `median`, `has_close_elements`, ...), 4 to 16 statements (mean 7.1).
* **Oracle:** the expected outcome of a test is whatever the reference implementation returns or raises; a test passes when the mutant's behaviour (`return:<value>` or `raise:<exception type>`) equals the oracle's. The reference implementations are pinned by hand-written expectations in the tests.
* **Fault injection:** each mutant changes exactly one AST node. Ten operators:

| Operator | Category | Transformation |
|---|---|---|
| ROR-flip | comparison | `<` to `>`, `<=` to `>=`, `==` and `!=` swapped |
| ROR-bound | boundary | `<` and `<=` swapped, `>` and `>=` swapped |
| CRP | off-by-one | integer constant `c` to `c+1` and `c-1` |
| LCR | boolean | `and` and `or` swapped |
| NEG | boolean | condition of `if` or `while` negated |
| MEM | boolean | `in` and `not in` swapped |
| AOR | arithmetic | `+` and `-` swapped; `*` to `+`; `/`, `//`, `%` to `*` |
| BCR | return value | `True` and `False` swapped |
| SDL | omission | assignment or expression statement replaced by `pass` |
| JMP | control flow | `break` and `continue` swapped (produced no mutants here) |

* **Counts:** 625 mutants generated; 1 excluded (timeout); 624 valid; **594 detected** by at least one test and used; 30 not detected (equivalent or missed), never counted as localised. Detected by category: off-by-one 244, omission 83, comparison 73, boolean 79, arithmetic 64, boundary 35, return value 16.
* **Ground truth:** the single changed line, stored separately and read only by the metric code and by the parameter estimation for *other* programs.
* **Two-fault set:** 415 sampled, 410 evaluated.
* **Protocol:** leave-one-program-out in versions 2 and 3; seed 20261013; a test is stopped and counted as failed after 20,000 executed lines.

### 2.3 QuixBugs
* Source: <https://github.com/jkoppel/QuixBugs>, commit `4257f44b0ff1181dedaedee6a447e133219fcebf`, MIT. Python programs (buggy), `correct_python_programs/` (reference), `json_testcases/`.
* 40 programs exist; **31 have JSON test cases and all 31 are used** (the other 9 need object graphs as input). 231 of the 242 tests are kept: 9 dropped because the benchmark's own correct program exceeds the step limit of 100,000 lines, 2 because its output differs from the expected output.
* **Human-written bugs:** one single-line bug per program; the fault line is found by comparing the buggy and the correct version (29 are a modified line, 2 a missing line, blamed on the statement after the gap). For only **5 of 31** is a single edit a plausible repair.
* Injected-fault sets Q1, Q2, Q3 (1, 2, 3 faults), QH (31 human bugs), QHM (human bug plus one injected); D3 in version 3 (15 sampled mutants per correct program, 382 detected).
* Used already in version 3, so it is **less clean** as held-out data than MBPP.

### 2.4 HumanEvalPack / HumanEvalFix (Python)
* Source: <https://huggingface.co/datasets/bigcode/humanevalpack>, file `python/test-00000-of-00001.parquet` (164 problems, MIT per the data set card). SHA-256 of the file as downloaded `ed5f15d7...f8075`, of the JSON-lines copy the code reads `523fb78c...e10`.
* HumanEvalFix gives, for each problem, a correct solution, a **human-inserted bug** (146 usable) and tests. Sets E1, E2, E3 (injected 1, 2, 3 faults), **EH** (146 human bugs), EHM (human bug plus one injected).
* Restoring-variant statistics: a single edit repairs **46 of the 146** human bugs, against 0.77 to 0.83 of injected single faults. This is the main reason the model is weaker on human bugs.

### 2.5 MBPP (confirmatory data)
* Source: <https://huggingface.co/datasets/google-research-datasets/mbpp>, configuration `full` (974 problems, 4 splits), CC BY 4.0. Used **only** for the version 5 confirmatory study.
* **Selection rules (fixed in the protocol before the model was run):** a single function with no setup code, at least 2 literal inputs, at least **6 usable tests after amplification**, **5 to 30 statements**, name not clashing with D1. Result: **362 programs**.
* **Test amplification.** MBPP gives only about 3 asserted calls per problem, too few to localise anything. We perturb one argument of each asserted call to reach about **12 tests per program** (12.6 on average). The expected output of a perturbed test is the output of the **reference program**, so test outcomes are correct by construction.
* **Fault injection:** the ten operators above, on distinct lines; **4, 4 and 2 faulty versions per program** for 1, 2, 3 faults (360 to 362 programs per fault count). Programs where an injected fault is not detected are skipped.
* **Statement cap of 30** because exact inference on the largest case (28 statements, 49 variants) already takes 52 s (median 0.03 s).

### 2.6 What the data can and cannot show
Injected faults are generated by the same ten operators that generate the analysis variants, so a restoring variant exists far more often than for human bugs. **Every result on injected faults is therefore an upper bound for the variant evidence**; the human-bug sets (177 bugs) are the counter-check, and there the model is weaker.

---

## 3. Probability background

Each concept below is used in the model.

| Concept | Definition | Use in this project |
|---|---|---|
| Random variable, distribution | variable with probabilities summing to 1 | $F_i\in\{0,1\}$ statement faulty; $y_j\in\{\text{pass},\text{fail}\}$; $\Theta$ on a 10-point grid |
| Joint, marginal, conditional | $P(A\mid B)=P(A,B)/P(B)$; $P(A)=\sum_B P(A,B)$ | the answer $P(F_i=1\mid y)$ is a marginal of the posterior |
| Bayes' rule | $P(H\mid D)=P(H)P(D\mid H)/P(D)$ | posterior over fault configurations |
| Odds form | posterior odds = prior odds × likelihood ratio | every per-statement evidence source folds into the prior exactly |
| Independence, conditional independence | $P(A,B)=P(A)P(B)$ | faults independent a priori; tests independent given faults and $\Theta$ |
| Bayesian network, d-separation | DAG factorisation; unconnected nodes do not influence each other | arrows $F_i\to y_j$; an unexecuted statement keeps its prior |
| Noisy-OR | independent causes, effect if any fires | likelihood of a test outcome |
| Latent variables, marginalisation | sum over unobserved values | $F$ and $\Theta$ are summed out, not guessed |
| Explaining away | learning one cause lowers belief in a competing cause | why marginals must come from the joint, not line by line |
| Maximum likelihood, mixtures, EM | fit weights of a mixture by alternating soft assignment and averaging | prior of $\Theta$; variant kinds |
| Laplace smoothing | $(s+1)/(t+2)$ = posterior mean under Beta(1,1) | every estimated proportion |
| Model selection, BIC | $k\ln n-2\ln L$ | number of non-repair variant kinds |
| Interventions, $do()$ | setting a variable differs from observing it | a program variant sets one statement |
| Calibration, proper scoring rules | stated 0.7 should be right about 70% of the time; Brier and log loss are minimised only by the true probability | evaluation of the probabilities |
| Bootstrap, permutation test, Holm | resampling interval; sign-flip test; step-down multiple-testing control | comparison of methods |

---

## 4. The mathematics of the model

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $n, m$ | statements, tests |
| $C_{ji}$ | test $j$ executes statement $i$ (observed coverage) |
| $y_j$ | test $j$ fails (observed) |
| $F_i$ | statement $i$ is faulty (latent) |
| $\Theta$ | exposure rate: chance an executed faulty statement breaks a test (latent); grid $\theta_g=(g-0.5)/10$, prior weights $w_g$ |
| $k_j(F)=\sum_i C_{ji}F_i$ | faulty statements executed by test $j$ |
| $\pi_n=\min(\mu/n,\,0.5)$ | prior fault probability ($\mu$ = expected faults, learned) |
| $\lambda$ | leak: failure for reasons the model has no variable for |
| $v$, $s(v)$ | a single-edit variant and the statement it edits |
| $z_{vj}$ | for a **failing** test $j$ that executes $s(v)$: the variant makes it pass |
| $b_{vj}$ | for a **passing** test $j$ that executes $s(v)$: the variant makes it fail |

### 4.2 The base model: a noisy-OR Bayesian network

Generative story. For each test $j$ and each faulty statement $i$ it executes, draw an independent exposure $X_{ji}\sim\text{Bernoulli}(\Theta)$; draw a leak $L_j\sim\text{Bernoulli}(\lambda)$. The test fails if it leaks or any exposure fires. A test passes only if **every** cause is silent, and independent causes multiply:

$$P(y_j=0\mid F,\Theta)=(1-\lambda)(1-\Theta)^{k_j(F)},\qquad P(y_j=1\mid F,\Theta)=1-(1-\lambda)(1-\Theta)^{k_j(F)}$$

Joint distribution (arrows $F_i\to y_j$ exactly where $C_{ji}=1$, and $\Theta\to y_j$ for all $j$):

$$P(F,\Theta,y)=P(\Theta)\prod_i P(F_i)\prod_j P(y_j\mid k_j(F),\Theta)$$

* The leak $\lambda$ guarantees every outcome has positive probability (no "impossible evidence").
* $\Theta<1$ because a fault is not always exposed (rare branch, easy inputs).
* Assumptions: faults independent a priori; tests independent given $(F,\Theta)$; one shared $\Theta$ per program; coverage fixed. These are the main weaknesses.

**Posterior and explaining away.** The wanted quantity

$$P(F_i=1\mid y)=\frac{\sum_{F:F_i=1}\sum_g w_g P(F)P(y\mid F,\theta_g)}{\sum_{F}\sum_g w_g P(F)P(y\mid F,\theta_g)}$$

does not factorise: given a failing test, two statements in its scope become negatively dependent.

**Hand-checked example** (test `tests/test_docs_example.py`). $\pi=0.1,\ \theta=0.8,\ \lambda=0.01$; test 1 fails (scope $\{s_1,s_2\}$), test 2 passes ($\{s_2,s_3\}$), test 3 passes ($\{s_3\}$). A test passes with probability $0.99\cdot0.2^k$ (0.99, 0.198, 0.0396 for $k=0,1,2$). Enumerating the 8 configurations gives $P(y)=0.0856$ and

| | prior | posterior |
|---|---|---|
| $s_1$ | 0.10 | **0.767** |
| $s_2$ | 0.10 | **0.169** (it also ran in a passing test: factor $1-\theta=0.2$) |
| $s_3$ | 0.10 | **0.004** (only in passing tests) |
| no fault at all | 0.73 | **0.083** (the failure was a leak) |

**Evidence that depends on one statement only** (a static-analysis flag $A_i$, a mutation-evidence level $M_i$) enters as a likelihood ratio on the prior odds, exactly:

$$\text{odds}(F_i\mid A_i,M_i)=\frac{\pi}{1-\pi}\cdot\frac{P(A_i\mid F_i{=}1)}{P(A_i\mid F_i{=}0)}\cdot\frac{P(M_i\mid F_i{=}1)}{P(M_i\mid F_i{=}0)}$$

### 4.3 Why the single-fault mutation model fails with two faults

Version 3 summarises the mutation analysis of each statement by one of five levels $M_i$ (0 no operator applies; 1 no variant helps; 2 some variant fixes some failing tests; 3 some variant fixes all failing tests but breaks a passing one; 4 some variant makes all tests pass) and multiplies the prior odds by a ratio **learned from single-fault programs**. Learned ratios: level 4: **162**; level 1: **0.023**; level 2: 0.048.

With two faults $i,g$ in the same tests, repairing $i$ removes only $i$'s exposure, so the failing tests that $g$ breaks keep failing. The true fault $i$ shows level 1 or 2 and the single-fault table reads this as innocence. The table is not wrong; it answers a question about a different population. What a variant of $i$ does depends on **which other statements are faulty**, so that dependence must be in the model. Measured damage (410 two-fault programs): both faults in the top 3 for **19.8%** with the fused model against **35.9%** without variants.

### 4.4 The key quantity: probability that a repair fixes a failing test

Condition on test $j$ failing, with $k$ faulty statements in its scope. Repairing statement $s$ removes $s$'s exposure. The test then passes iff nothing else caused the failure: no leak and none of the other $k-1$ exposures fired. Since the test failed, $s$'s exposure must have been one of the causes. Hence

$$q(k,\Theta)=\frac{P(L{=}0,\;X_s{=}1,\;X_{\text{others}}{=}0)}{P(\text{fail})}=\frac{(1-\lambda)\,\Theta\,(1-\Theta)^{k-1}}{1-(1-\lambda)(1-\Theta)^{k}}$$

| | $q(1,\Theta)$ | $q(2,\Theta)$ | $q(3,\Theta)$ |
|---|---|---|---|
| $\lambda\to0$ | 1 | $\dfrac{1-\Theta}{2-\Theta}$ | $\dfrac{(1-\Theta)^2}{3-3\Theta+\Theta^2}$ |
| $\Theta=0.5$ | 1 | 0.333 | 0.143 |
| $\Theta=0.9$ | 1 | 0.091 | 0.009 |
| $\lambda=0.01,\ \Theta=0.8$ | 0.988 | 0.165 | |

* With one fault a repair flips every failing test it reaches; with a second fault in the same test it usually does **not**, and the model now *expects* that.
* The events "$s$ is the sole cause" for different faulty $s$ in the same test are mutually exclusive, so $k\,q(k,\Theta)\le1$.
* **For $k\ge2$, $q$ has no free parameter**: it follows from noisy-OR. The multi-fault behaviour is a prediction from single-fault training data, which is what makes the experiments a real test.

### 4.5 The variant (interventional) likelihood

A variant $v$ replaces statement $s=s(v)$ by other code: an **intervention**. Only tests that execute $s$ can change (programs are deterministic).

* **Repair** (possible only if $F_s=1$ and a repair exists among the variants of $s$, probability $\alpha$): flips each failing test with probability $\gamma\,q(k_j,\Theta)$ ($\gamma\le1$ allows partial repair); breaks a passing test with probability $\beta$.
* **Non-repair kinds** $c=1..K$ ($K=5$, chosen by BIC): each has a flip rate $\psi_c$ and a break rate $\eta_c$, mixture weights $\kappa_c$.

Likelihood of the observations of $v$ when it is not a repair, and when it is:

$$N_v=\sum_c\kappa_c\prod_j\psi_c^{z}(1-\psi_c)^{1-z}\prod_j\eta_c^{b}(1-\eta_c)^{1-b},\qquad
\text{Rep}_v(F,\Theta)=\prod_{j\ \text{failing}}(\gamma q_j)^{z}(1-\gamma q_j)^{1-z}\prod_{j\ \text{passing}}\beta^{b}(1-\beta)^{1-b}$$

**Composite layer** (first design, kept as an ablation). Treating variants as independent given $(F,\Theta)$ gives one factor per *faulty* statement,

$$\Lambda_s(F,\Theta)=(1-\alpha)+\alpha\prod_{v\in V_s}\Big[1+\rho\Big(\tfrac{\text{Rep}_v(F,\Theta)}{N_v}-1\Big)\Big],\qquad \Lambda_s\ge1-\alpha,$$

so failing to find a repair can lower a statement's odds by at most $1-\alpha$, however many variants were tried (a guard for human bugs that have no single-edit repair). It double-counts: "repair of $i$ fixes test $j$" and "repair of $g$ fixes test $j$" are mutually exclusive but are multiplied as independent. On development data it gave log loss 0.143 against 0.074 for version 3 and inferred about two faults for single-fault programs.

**Exclusive layer** (the model of record). Two restrictions that are exact when they hold: (i) at most one variant of each faulty statement is its repair (choice $a_s$: none with probability $1-\alpha$, or one of $V_s$ variants with probability $\alpha/V_s$); (ii) every failing test has at most one sole cause. For configuration $F$, choices $a$ and a failing test $j$:

$$P(\text{flips on }j\mid F,a,\theta)=\sum_{c\in F,\,C_{jc}=1}q(k_j,\theta)\,g_c\prod_{s\ne c}h_s\;+\;\big(1-k_j\,q(k_j,\theta)\big)\prod_s h_s$$

with $g_c=\gamma^{z}(1-\gamma)^{1-z}$ for the repaired statement and $h_s=\delta^{z}(1-\delta)^{1-z}$ for the other repaired statements ($\delta$ = the smallest flip rate among the non-repair kinds). The variant factor of $F$ is the sum over all choices $a$ of the products of $P(a_s)$, the break and $N_v$ terms of the chosen variants, and these per-test probabilities. Cost per configuration: $\prod_s(V_s+1)$ choices, at most three statements.

### 4.6 Final posterior

$$P(F\mid\text{data})\;\propto\;\sum_g w_g\ \prod_i\pi_n^{F_i}(1-\pi_n)^{1-F_i}\ \prod_j P(y_j\mid k_j(F),\theta_g)\ \cdot\ \text{factor}(F,\theta_g),\qquad |F|\le3$$

**Consistency checks (all unit-tested):** with no variants it equals the version-2 coverage-only network (restricted to $|F|\le3$); restricted to $|F|\le1$ it is a single-fault likelihood-ratio model; with $|F|\le1$ and $\alpha=0$ it equals the single-fault closed form.

### 4.7 Parameter estimation (single-fault mutants of *other* programs only)

| Parameter | Estimator |
|---|---|
| $\pi$ (v2), $\mu$ (v3+) | $(\text{faulty}+1)/(\text{statements}+2)$ over training mutants; $\mu$ = expected faults per program |
| $\lambda$ | $(\text{failing pairs with fault outside scope}+1)/(\text{such pairs}+2)$ (order 0.001) |
| pooled $\theta$ | $q=(\text{failing pairs with fault in scope}+1)/(\text{pairs}+2)$, then $\theta=1-(1-q)/(1-\lambda)$ |
| prior of $\Theta$ | **EM** on the 10-point grid |
| $\kappa_c,\psi_c,\eta_c$ | EM on variants of fault-free statements |
| $\alpha,\rho,\gamma,\beta$ | EM on variants of faulty statements |

**EM for the exposure prior.** A training fault executed by $n_f$ tests of which $x_f$ failed has $x_f\mid\Theta{=}\theta_g\sim\text{Bin}(n_f,q_g)$, $q_g=1-(1-\lambda)(1-\theta_g)$: a mixture of binomials with unknown weights.

$$\text{E: } r_{fg}=\frac{w_g\,q_g^{x_f}(1-q_g)^{n_f-x_f}}{\sum_h w_h\,q_h^{x_f}(1-q_h)^{n_f-x_f}}\qquad\text{M: } w_g=\frac1N\sum_f r_{fg}$$

EM never decreases the likelihood; here it is concave in $w$, so EM from uniform weights reaches the global maximum. A pseudo-count per grid point stops any weight reaching zero. **BIC** for the number of non-repair kinds $K=2..7$: 58,683; 54,258; 53,522; 53,131; 53,068; 53,051; the rule "smallest $K$ within 100 of the minimum" gives $K=5$.

**No leakage:** the analysed program never contributes to its own parameters; ground truth is read only by the metrics. **Known weak point:** $\alpha$ is a property of the fault population (about 0.83 for injected faults; low for human bugs); the bound $\Lambda\ge1-\alpha$ limits but does not remove the damage.

### 4.8 Inference

* **Exact enumeration.** For each fault set $F$ and each grid value $\theta_g$ compute in log space $\log w_g+\sum_i[\ldots]+\sum_j\log P(y_j\mid k_j,\theta_g)+\log\text{factor}$, normalise by log-sum-exp, and sum configurations containing $i$. Also gives $P(\text{no fault})$, the posterior of $\Theta$ and the posterior over the **number of faults**.
* **Bounded cardinality** $|F|\le3$: $\sum_{r\le3}\binom nr$ configurations (1,351 for $n=20$; 20,876 for $n=50$). This is exact inference in the model whose prior is conditioned on $|F|\le3$, not an error bound against the unrestricted model; allowing a fourth fault changed marginals by less than 0.05 in 100 of 100 cases (maximum 0.041).
* **Version 2** (no variants) enumerates all $2^a$ configurations of the $a$ active statements ($a\le22$).
* **Why not variable elimination or a junction tree?** Every test has all the statements it executed as parents, so the moral graph is dense and the treewidth grows with the scope size; elimination gives no saving. Exactness is verified against brute force (80 random models), two closed forms and pgmpy variable elimination, to $10^{-9}$.
* **Gibbs sampling** (four chains, Gelman-Rubin diagnostic) exists as a fallback for $a>22$, validated on a small model; **no reported result uses it.**
* **Cost:** median 0.03 s per program; largest case (28 statements, 49 variants) 52 s.

### 4.9 Decisions

* **Ranking** by $P(F_i=1\mid\text{data})$; tied statements are ordered uniformly at random in expectation.
* **Abstention** as a maximum-expected-utility rule with utilities $+1$ (correct), $-c$ (wrong), $0$ (abstain): answer iff $p_{\max}>c/(1+c)$. Quality: area under the risk-coverage curve.
* **Compile errors** are certain (hard gate); **unexecuted statements** keep their prior.
* **Targeted repair:** try single edits on the top-3 statements, re-run all tests, accept only if all pass.

---

## 5. Models and baselines

### 5.1 The model family (all in `pcg/`)

| Version | Model | What it adds | Status |
|---|---|---|---|
| **v2** `PCG` | noisy-OR network, exact inference, $\Theta$ summed out under an EM-fitted prior | probabilities; ranks like Ochiai | baseline, preserved |
| v2 variants | fixed-$\theta$; + static analysis; + dependence slicing | ablations | preserved |
| **v3** `PCG fused` | + size-aware prior $\min(\mu/n,1/2)$ + five-level mutation evidence with learned tables, single fault; recalibration; selective prediction | large gain on injected single faults; **collapses at two faults** | baseline, preserved |
| **v4/v5 `IV`** (**proposed**) | interventional variant likelihood $q(k,\Theta)$, five-kind non-repair mixture, exclusive layer, exact bounded-cardinality inference ($\le3$ faults), posterior over the number of faults | multi-fault mutation evidence | proposed |
| `IV` ablations | composite layer; $\le1$ or $\le2$ faults; no variants | what each part contributes | reported |

### 5.2 Baselines (12, all receive the same coverage and variant outcomes; none sees ground truth)

| Baseline | Kind | Note |
|---|---|---|
| Ochiai, Tarantula | spectrum formulas | $\text{Ochiai}=e_f/\sqrt{F(e_f+e_p)}$; Tarantula $=\frac{e_f/F}{e_f/F+e_p/P}$ |
| Barinel | Bayesian, candidate-based (Abreu et al.) | candidates up to size 3 |
| FLITSR, FLITSR\* | multi-fault spectrum, iterative removal of explained failing tests | FLITSR\* re-implemented from the paper |
| Metallaxis, MUSE | mutation-based formulas | re-implemented from the papers |
| Hybrid average | mean of Ochiai and Metallaxis | |
| Logistic, spectrum | logistic regression on spectrum features | trained on single-fault data of other programs |
| Logistic fusion | logistic regression on spectrum + variant features | the strong learned fusion |
| PCG spectrum only | v2 coverage-only network | no variants |
| PCG v3 fused | single-fault fusion | |

Not compared: SmartFL, Prosecutor, language-model localisers.

---

## 6. Experimental design and evaluation mathematics

**Pre-registration and the confirmatory / exploratory line.** Hypotheses, data rules, metric, statistics and decision rules of the MBPP study were fixed in a protocol before the model was run on it. Every result is labelled **[C]** confirmatory, **[P]** pre-registered held-out, **[D]** descriptive, **[E]** exploratory or post hoc, **[Dev]** development only (details: [research/RESULTS.md](research/RESULTS.md) section 0). Two reproducibility deviations (a longer wall-clock limit and the seeding of the tie-breaking draws) were made *after two runs whose results had been seen*; they cannot favour a method and all ten decisions were identical in those two runs. They are disclosed.

**Primary metric: expected average precision (AP).** AP of the faults, averaged over 300 uniform random orderings of tied statements, seeded by a hash of the case id. *The flaw this fixes:* the common mid-rank AP, $\text{AP}=\text{mean}_t\,|\{u:r(u)\le r(t)\}|/r(t)$, gives $2/1.5=1.33>1$ when two faults tie at the top; it exceeded 1 in 1.0 to 5.1% of cases (maximum 1.5), and coverage-only methods tie more often (4.5% of cases) than the proposed model (1.4%), so they were favoured. We found this after the version 4 results and disclose that the version 4 headline used the old metric.

**Other metrics.** First-fault Top-1; all faults in the top 3; mean reciprocal rank (human bugs); **Brier** $=\text{mean}(p-o)^2$; **log loss** $=-\text{mean}\ln P(o)$; **Brier skill** $=1-\text{Brier}/\text{Brier}_{\text{base rate}}$ (negative = worse than always guessing the base rate); **ECE** (10 equal-width bins); risk-coverage area (AURC); AUC of the posterior expected number of faults for "multi-fault vs single-fault".

**Statistics.** The program is the unit. 95% **cluster bootstrap** (5,000 resamples of programs). **Paired sign-flip permutation test** (20,000 flips). **Holm step-down** over the 10 confirmatory comparisons (smallest p against $0.05/10$, next against $0.05/9$, and so on, stopping at the first failure).

**Confirmatory hypotheses (MBPP).** C1: the model beats MUSE, logistic fusion, v3 fused, spectrum-only and FLITSR\* at two and at three faults (Holm over 10). C2: single-fault first-fault Top-1 cost against v3 fused is at most 0.02. C3: the difference to MUSE is positive at both fault counts.

---

## 7. Results

All figures are in [`results/v5/figures/`](results/v5/figures/) (PNG and PDF); all tables in `results/v5/tables/`; every number was checked against the saved files.

### 7.1 Confirmatory study on 362 fresh MBPP programs [C]

![Expected average precision on MBPP](results/v5/figures/fig_v5_1_mbpp_ranking.png)

**Two faults, expected AP:** IV **0.613**, logistic fusion 0.590, MUSE 0.583, v3 fused 0.548, spectrum-only 0.522, FLITSR\* 0.520. First-fault Top-1: IV 0.453, MUSE 0.420, logistic fusion 0.423, v3 fused 0.349, coverage-only methods 0.07 to 0.09.

| Faults | Baseline | IV minus baseline [95% CI] | Holm p |
|---|---|---|---|
| 2 | MUSE | **+0.030** [+0.012, +0.046] | 0.004 |
| 2 | Logistic fusion | +0.023 [+0.012, +0.034] | 0.002 |
| 2 | PCG v3 fused | +0.065 [+0.054, +0.075] | 0.0005 |
| 2 | PCG spectrum only | +0.090 [+0.076, +0.104] | 0.0005 |
| 2 | FLITSR\* | +0.093 [+0.079, +0.106] | 0.0005 |
| 3 | MUSE | -0.015 [-0.035, +0.006] | 0.34 |
| 3 | Logistic fusion | +0.003 [-0.009, +0.016] | 0.56 |
| 3 | PCG v3 fused | +0.041 [+0.029, +0.053] | 0.0005 |
| 3 | PCG spectrum only | +0.015 [+0.002, +0.028] | 0.074 |
| 3 | FLITSR\* | +0.021 [+0.007, +0.034] | 0.011 |

| Hypothesis | Outcome |
|---|---|
| **C1** | **supported at two faults** against all five baselines; **not supported at three faults** against MUSE, logistic fusion, spectrum-only |
| **C2** single-fault cost at most 0.02 Top-1 | holds, narrowly: -0.006 [-0.020, +0.006] |
| **C3** positive vs MUSE at both counts | **fails** at three faults |

Single faults (M1): expected AP IV 0.904, v3 fused 0.923, logistic fusion 0.925 (IV slightly lower; Brier worse than v3 fused by +0.0046 [+0.0021, +0.0072]). Placing **all** faults in the top 3 at two faults: IV 0.262 against 0.28 to 0.30 for coverage-only and hybrid methods: **not better**.

**Pooled over QuixBugs, HumanEval and MBPP [E]:** two faults, IV minus MUSE **+0.028 [+0.014, +0.042]** (502 programs); three faults -0.006 [-0.023, +0.011] (486 programs).

### 7.2 Quality of the probabilities [D]

![Reliability of the probabilities](results/v5/figures/fig_v5_3_reliability.png)

| Brier skill vs base rate | 1 fault | 2 faults | 3 faults |
|---|---|---|---|
| **IV** | **+0.681** | **+0.062** | -0.142 |
| PCG v3 fused | +0.726 | -0.110 | -0.356 |
| Logistic fusion | +0.594 | -0.170 | -0.484 |
| PCG coverage-only | +0.057 | +0.027 | -0.105 |

IV has the lowest Brier score among the model-based methods at two and three faults (paired differences -0.029 to -0.071, intervals excluding 0), but **at three faults every method is worse than the base rate**. The posterior expected number of faults separates multi-fault from single-fault programs (AUC 0.774 on MBPP; about 0.85 on QuixBugs and HumanEval) but cannot tell two from three (mean 1.67 and 1.74).

### 7.3 Earlier held-out study (version 4): QuixBugs and HumanEvalFix, 3,622 cases [P]

| Hypothesis | Outcome |
|---|---|
| H1 two-fault MAP, Holm over 10 | **partly supported**: over v3 fused +0.083 [+0.054, +0.111] (QuixBugs), +0.085 [+0.066, +0.104] (HumanEval); over logistic fusion on HumanEval +0.036 (Holm p 0.006); **not** over spectrum-only, MUSE (+0.031 [-0.018, +0.078]; +0.028 [-0.004, +0.058]) or FLITSR\* |
| H2 single-fault cost ≤ 0.02 Top-1 | supported on QuixBugs (+0.003) and HumanEval (0.000); not on D1 (-0.017) |
| H3 human bugs, MRR above spectrum-only | supported: HumanEvalFix +0.171 [+0.120, +0.223]; QuixBugs +0.145 [+0.029, +0.264] |
| H4 unknown number of faults | supported with small margins (+0.014, +0.010 over logistic fusion) |
| H5 probabilities | **not supported**: Brier skill negative at three faults (-0.125, -0.167) |
| H6 three vs four faults allowed | supported: marginals differ by less than 0.05 in 100 of 100 cases |
| H7 inferred number of faults | supported: AUC 0.854 and 0.844; separates one from several faults, not two from three |

* **Human-written bugs (146 HumanEvalFix):** mean reciprocal rank IV **0.761**, v3 fused 0.719 (+0.042 [+0.013, +0.071]), logistic fusion 0.724 (+0.037 [+0.015, +0.062]), about 0.59 for coverage-only methods. Where a restoring variant exists (46 bugs) MRR is 0.989; where none exists (100 bugs) 0.656. On the 31 QuixBugs bugs no gain over the v3 or learned fusion.
* **Ablation (MAP on QuixBugs 2 faults / HumanEval 2 faults):** proposed 0.632 / 0.663; at most 2 faults 0.632 / 0.663 (identical); at most 1 fault 0.577 / 0.605; composite layer 0.601 / 0.639; no variants 0.578 / 0.638. Variant evidence adds +0.054 / +0.025, allowing a second fault +0.055 / +0.058, the exact layer over the composite one +0.031 / +0.024.
* **Calibration (ECE):** IV 0.032, 0.048, 0.069 against 0.080, 0.091, 0.087 for v3 fused and 0.138 to 0.150 for logistic fusion.
* **Tie-corrected re-analysis (post hoc [E]):** with expected AP, IV beats v3 fused, spectrum-only, FLITSR\* and logistic fusion on both two-fault sets (+0.046 to +0.093, Holm p at most 0.022), but not MUSE (+0.030 [-0.016, +0.075]; +0.025 [-0.003, +0.053]). This motivated the confirmatory version 5.

### 7.4 Simulated programs [E] (`results/v4_synthetic`)
250 simulated programs per cell, generating process fitted on D1. When the data follow the model, IV reaches MAP 0.937 (two faults) against MUSE 0.671, and stays first under correlated tests, coupled faults and half the variants missing. When only 20% of faulty statements have a repair among their variants (the human-bug situation) it falls to the level of the coverage-only network (0.638 against 0.608 at two faults). *Worlds are generated by the model's own mechanism, so high scores are expected; the gap to real data measures how far real faults are from the model.*

### 7.5 Version 2 and version 3 results (development benchmark D1 and QuixBugs)

| Single-fault ranking, 594 faults | Top-1 | Top-3 | MRR |
|---|---|---|---|
| Chance | 0.136 | 0.409 | 0.354 |
| Tarantula | 0.338 | 0.815 | 0.589 |
| Ochiai | 0.538 | 0.912 | 0.726 |
| Barinel | 0.538 | 0.910 | 0.724 |
| PCG v2 (Θ summed out) | 0.547 | 0.900 | 0.726 |
| PCG v2, fixed θ | 0.540 | 0.845 | 0.704 |
| Metallaxis / MUSE | 0.748 / 0.746 | 0.877 / 0.854 | 0.827 / 0.816 |
| Logistic fusion | 0.959 | 0.999 | 0.978 |
| PCG v3 fused | 0.957 | 0.990 | 0.974 |

* PCG v2 and Ochiai are **statistically indistinguishable** (Top-1 +0.0089 [-0.0002, +0.0192]); summing $\Theta$ out matters (fixed θ lowers Top-3 from 0.900 to 0.845 and gives a log loss worse than the base rate).
* The mutation results are **inflated**: the exact repair is among the variants for 492 of 594 injected faults. With the exact-repair variant removed, PCG v3 fused 0.729 Top-1, logistic fusion 0.771.
* **Human bugs (QuixBugs, 31):** Ochiai 0.376, Metallaxis 0.305, MUSE 0.395, logistic fusion 0.575, PCG v3 fused 0.511 Top-1. Mutation-only methods are *worse* than spectrum on human bugs, the reverse of injected faults. A single edit is a plausible repair for only 5 of 31.
* **Probabilities transfer better for the model-based fusion:** on injected faults in QuixBugs programs Brier 0.0200 against 0.0288 for logistic fusion (-0.0088 [-0.0135, -0.0048]); on human bugs logistic fusion's log loss (0.410) is worse than the base rate (0.380) while the fused model's (0.316) is better.
* **Selective prediction:** area under the risk-coverage curve 0.251 for the posterior against 0.312 for Ochiai-based confidence.
* **Static analysis (pylint)** attached a finding to no faulty line and to 52 fault-free lines: **no information about fault locations**.
* **Two faults (410 programs):** PCG v3 fused: both faults in the top 3 for 19.8% against 35.9% without variants: the **two-fault collapse** that motivated version 4.
* **Targeted repair:** top-3 guided plausible repair rate 0.749, exact 0.609, 4.75 candidates validated against 13.38 unguided; PCG and Ochiai guidance equivalent.
* **Simulated LLM-confidence signal (synthetic, not a language model):** a weak informative signal would lift Top-1 from 0.547 to 0.757, a strong one to 0.975.

### 7.6 Software verification and reproducibility
* **224 automated tests:** brute-force equivalence, closed forms, pgmpy agreement, EM recovery, parameter recovery from model-sampled data, the hand-checked worked example, and the consistency checks of section 4.6.
* **Checksummed results:** each `results/*/MANIFEST.json` holds the SHA-256 of every file. Two full runs of version 4 agree on 26 of 29 files (the rest differ in seconds); the final version 5 run and a repeat agree bit for bit on 13 of 17 files and on every reported number (the others differ in timings and one last-bit diagnostic value). All derived tables and figures regenerate bit for bit from the raw per-case files.
* `scripts/check_layout.py` checks paths, imports, manifests and links; `scripts/verify_manuscript.py` checks 50 numbers of the manuscript against the saved results.

---

## 8. Negative results

Kept in full, because they are part of the finding:

1. Static analysis carried no information about fault location.
2. Conditioning mutation evidence on coverage context and the size-aware prior did not help (two pre-registered version 3 hypotheses refuted).
3. The version 3 fusion **collapses at two faults**.
4. Version 4 is not significantly better than MUSE or FLITSR\*.
5. **No advantage over MUSE at three faults**; the probabilities are **worse than the base rate at three faults**.
6. The model never finds *all* faults in the top 3 more often than coverage-only methods.
7. It is slightly worse than the single-fault fusion and the learned fusion on single faults.
8. On human bugs with no restoring variant it approaches the coverage-only network.
9. A third-fault capacity was not needed (at most 2 vs 3 faults differ by at most 0.002 AP); a second repair kind, more non-repair kinds, rescaled odds, a temperature, and a hierarchical prior on the number of faults (log loss 0.2703 to 0.2697) all gave no gain and were dropped.
10. The pre-registered version 4 metric had a tie flaw (found by us, corrected post hoc).
11. The composite variant layer double-counted repairs and was replaced.
12. Two reproducibility deviations followed two runs whose results had been seen (disclosed above).

---

## 9. Limitations

* **Injected faults share their operators with the variants**, so results on injected faults are an upper bound for the variant evidence; human-bug results are weaker (restoring variant for 46 of 146 HumanEvalFix bugs, 5 of 31 QuixBugs bugs, against 0.77 to 0.83 of injected faults).
* Multi-fault programs are **constructed, not found**; naturally co-occurring faults (multi-fault Defects4J and BugsInPy versions) were not used.
* Single functions of 5 to 30 statements; cost grows with the number of variants of the faulty statements.
* Effects are small and rest on a tie-corrected metric fixed before the confirmatory data were seen but after version 4 results suggested it.
* MBPP tests are amplified by perturbing inputs; FLITSR\*, Metallaxis and MUSE are re-implementations from their papers.
* No comparison with SmartFL or Prosecutor; **no language-model evidence** (code for an LLM interface exists but no result uses it).
* Model assumptions: independent faults a priori, tests independent given faults, one shared $\Theta$, fixed coverage, no fault interaction.
* The novelty statement is search-limited; some related work was read at abstract level only. Details: [research/THREATS_TO_VALIDITY.md](research/THREATS_TO_VALIDITY.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**Readiness (our own assessment).** A careful small empirical study, suitable for a workshop or student research track; not enough for a research conference without real project bugs, a semantic competitor and a model that does not lose to coverage-only methods at three faults. No venue claim was verified.

---

## 10. Future scope

1. **Naturally occurring multi-fault data:** multi-fault versions of Defects4J and BugsInPy, to remove the shared-operator advantage.
2. **Stronger competitors:** SmartFL, Prosecutor, and language-model-based localisers.
3. **Dependence-aware test likelihood:** tests exercising the same path are correlated beyond what the faults explain; model the dependence instead of a shared $\Theta$ only; per-fault $\Theta$.
4. **Better three-fault behaviour:** recalibration conditioned on the inferred number of faults, hierarchical priors that carry information, larger enumeration with pruning.
5. **Language models:** use the posterior as a probabilistic verification layer for generated code (the motivation: type-constrained generation still fails 18 to 70% of tests, Mündler et al., PLDI 2025); fuse calibrated line-level model confidence as another likelihood factor.
6. **Value of information:** with exact inference taking milliseconds, choose the next test by the expected increase of the top posterior (VPI-based test selection).
7. **Scale:** project-level code, dynamic Bayesian networks over program versions, online parameter updating.
8. **Replication:** an independent re-implementation and a larger confirmatory sample.

---

## 11. Install, run, reproduce

Python 3.10 or newer. In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No API key and no GPU. All data sets are already in `data/external/`.

| Command | Purpose | Time |
|---|---|---|
| `python -m pytest` | 224 automated tests | 2 to 3 min |
| `python scripts/check_layout.py` | paths, imports, manifests, links | under 1 min |
| `python -m pcg benchmark` | version 2 experiments, `results/benchmark` | 6 min |
| `python -m pcg research` | version 3 experiments, `results/research` | 20 min |
| `python -m pcg v4` | version 4 held-out study, `results/v4` | about 2 h |
| `python -m pcg v5` | **version 5 confirmatory study on MBPP**, `results/v5` | about 45 min |
| `python -m pcg v4-synthetic`, `v4-report`, `v5-report`, `demo`, `demo-v4` | simulation, report rebuilds, demonstrations | seconds to 30 min |
| `python -m pcg analyze FILE --function NAME --tests TESTS.json` | analyse your own function | seconds |

Reproducibility details and the expected differences between two runs: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## 12. Repository layout

```
cognigraph/
  pcg/            the Python package
    program.py, _runner.py, execution.py     source handling, isolated test runs, coverage
    graph.py, dependence.py, static_analysis.py   constraint graph, slicing, pylint evidence
    model.py, estimation.py                  noisy-OR network, exact and Gibbs inference, EM
    mutation.py, mbfl.py                     fault injection; mutation analysis (levels, Metallaxis, MUSE)
    fusion.py, analysis.py                   version 3 fusion; analyze(): single inference entry point
    interventional.py                        version 4/5 interventional multi-fault model
    baselines.py, barinel.py, flitsr.py, discriminative.py   Ochiai, Tarantula, Barinel, FLITSR, logistic models
    metrics.py                               metrics, bootstrap, permutation test, Holm
    quixbugs.py, humanevalfix.py, mbpp.py    loaders of the external data sets (mbpp.py: test amplification)
    benchmark_programs.py                    D1, 34 programs
    experiments.py, research.py, v4.py, v5.py, synthetic.py, demo*.py   experiment drivers
    repair.py, llm.py, cli.py, reporting.py, visualize.py
  tests/          224 automated tests
  data/           external/ (QuixBugs, HumanEvalPack, MBPP: sources, commits, checksums), legacy/
  results/        one folder per experiment, each with MANIFEST.json
  figures/        index of every figure
  research/       protocols, results, audit, literature review, novelty, threats to validity
  docs/           report, model, experiments, limitations, reproducibility, review guide
  scripts/        check_layout.py, compare_manifests.py, verify_manuscript.py, make_latex_tables.py, development/
```

---

## 13. Documents

| File | Content |
|---|---|
| [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md) | version 2 and 3 model, estimation, inference, worked example |
| [research/MATHEMATICAL_DESIGN.md](research/MATHEMATICAL_DESIGN.md) | version 4/5 interventional model: design, amendments, edge cases |
| [research/RESULTS.md](research/RESULTS.md) | all version 4 and 5 results and the confirmatory / exploratory table |
| [research/RESEARCH_PROTOCOL.md](research/RESEARCH_PROTOCOL.md) | pre-registered protocol of version 4 |
| [research/THREATS_TO_VALIDITY.md](research/THREATS_TO_VALIDITY.md) | threats to validity |
| [research/LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md), [research/RELATED_WORK_AND_NOVELTY.md](research/RELATED_WORK_AND_NOVELTY.md), [research/REFERENCE_VERIFICATION.md](research/REFERENCE_VERIFICATION.md) | literature, novelty statement and its limits, what was read |
| [research/FINAL_RESEARCH_AUDIT.md](research/FINAL_RESEARCH_AUDIT.md) | audit of the research |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | version 2 and 3 experiments |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md) | limitations, reproduction, demonstration and examiner questions |
| [data/external/README.md](data/external/README.md) | data sources, licences, checksums |
| [figures/README.md](figures/README.md) | index of figures |

---

## 14. References

* Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* (noisy-OR, Bayesian networks)
* Koller, D., Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
* Abreu, R., Zoeteweij, P., van Gemund, A. J. C. (2009). Spectrum-Based Multiple Fault Localization (Barinel). ASE 2009.
* Callaghan, D., Fischer, B. (2023). Improving Spectrum-Based Localization of Multiple Faults by Iterative Test Suite Reduction (FLITSR). ISSTA 2023. arXiv:2306.09892.
* Kang, S., Choi, W., Yoo, S. (2023). A Bayesian Framework for Automated Debugging. ISSTA 2023. arXiv:2212.13773.
* Kim, Y., An, G., Feldt, R., Yoo, S. (2021). Ahead of Time Mutation Based Fault Localisation using Statistical Inference. ISSRE 2021, pp. 253-263. DOI 10.1109/ISSRE52982.2021.00036.
* Baradaran, Huang, Raghothaman, Le. Prosecutor: Bayesian Counterfactual Fault Localization. OOPSLA 2026. DOI 10.1145/3839469.
* Zeng et al. Fault Localization via Efficient Probabilistic Modeling of Program Semantics. ICSE 2022; SmartFL, IEEE TSE 2025.
* Papadakis, M., Le Traon, Y. (2015). Metallaxis-FL. STVR. Moon, S., Kim, Y., Kim, M., Yoo, S. (2014). Ask the Mutants (MUSE). ICST 2014.
* Xing, Chen, Xin, Xie (2025). Revisit the Intuition of Mutation-Based Fault Localization in Real-world Programs. Internetware 2025.
* Mündler, N. et al. (2025). Type-Constrained Code Generation with Language Models. PLDI 2025. arXiv:2504.09246.
* Gros, Devanbu. Localized Calibrated Uncertainty in Code Language Models. arXiv:2512.24560. Ferrer, Ramos (2024). Evaluating Posterior Probabilities: Decision Theory, Proper Scoring Rules, and Calibration. arXiv:2408.02841. Callaghan, Fischer. Mining Bug Repositories for Multi-Fault Programs. arXiv:2403.19171.
* Lin, D., Koppel, J., Chen, A., Solar-Lezama, A. (2017). QuixBugs. SPLASH Companion 2017.
* Muennighoff, N. et al. (2024). OctoPack: Instruction Tuning Code Large Language Models (HumanEvalPack). ICLR 2024.
* Austin, J. et al. (2021). Program Synthesis with Large Language Models (MBPP). arXiv:2108.07732.
* Ankan, A., Panda, A. (2015). pgmpy: Probabilistic Graphical Models using Python. SciPy 2015.

Verification level of each reference (what was read in full, what at abstract level): [research/REFERENCE_VERIFICATION.md](research/REFERENCE_VERIFICATION.md).
