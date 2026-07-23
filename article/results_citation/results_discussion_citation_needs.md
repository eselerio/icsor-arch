# Results and Discussion Citation Needs

The IDs below match the blue `RCxx` tags inserted in `article/wip/manuscript.tex`. Each quoted passage is the exact highlighted portion from the Results and Discussion section.

## RC01

> This distinction matters for interpreting the remaining results, because accuracy metrics evaluate closeness to the mechanistic solution whereas the conservation and non-negativity diagnostics evaluate whether a surrogate output is admissible as an activated-sludge state.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This is a methodological interpretation that separates statistical accuracy from physical/process admissibility. It needs support beyond the manuscript's own results. | A source explaining that activated sludge/process-model states are governed by mass balances and non-negative concentrations, or that prediction accuracy alone does not guarantee physical feasibility. | `("activated sludge" OR ASM) AND ("mass conservation" OR "mass balance") AND ("non-negativity" OR "nonnegative concentrations") AND ("machine learning" OR surrogate)` |

## RC02

> This ordering shows that the most accurate unconstrained approximations came from surrogates capable of representing nonlinear interactions among the operating variables and influent composition.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The ranking is internal, but the explanation attributes performance to nonlinear interaction modeling. That claim needs support from prior surrogate or wastewater modeling literature. | Evidence that wastewater/activated-sludge responses are nonlinear and that flexible learners such as neural networks or boosting models can represent nonlinear feature interactions. | `("activated sludge" OR wastewater) AND ("nonlinear interactions" OR "nonlinear relationships") AND ("neural network" OR "gradient boosting" OR surrogate)` |

## RC03

> which is consistent with gradient-boosted trees being effective in the small-to-moderate data range where local nonlinear partitions are useful and extensive representation learning is not yet well supported.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This is a general machine-learning explanation for why XGBoost and LightGBM performed well at smaller sample sizes. | A comparative or methodological source showing that gradient-boosted decision trees can perform strongly on tabular/small-to-moderate datasets and learn nonlinear local partitions efficiently. | `("gradient boosted trees" OR XGBoost OR LightGBM) AND ("small data" OR "tabular data" OR "moderate sample size") AND ("neural networks" OR "deep learning")` |

## RC04

> indicating that the 20-component response contained smooth multivariate structure that the neural surrogate could exploit once enough states were available.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This interprets the learning curve as evidence of smooth multivariate structure and neural-network suitability, which is broader than the observed numeric result. | A source on neural networks approximating smooth nonlinear multivariate functions, preferably in process modeling, surrogate modeling, or wastewater simulation. | `("neural network" OR MLP) AND ("smooth nonlinear function" OR "multivariate function approximation") AND ("surrogate model" OR "process modeling" OR wastewater)` |

## RC05

> its trajectory suggests a more data-hungry learning process than MLP or the boosting surrogates.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This describes TabNet's data requirements relative to other learners, which is an interpretation of model behavior that should be grounded in prior comparisons. | A TabNet or tabular deep-learning source discussing sample efficiency, data requirements, or comparisons with tree-based methods and MLPs. | `(TabNet OR "attentive tabular") AND ("sample efficiency" OR "data hungry" OR "data requirements") AND ("gradient boosting" OR MLP OR "tabular data")` |

## RC06

> In practical terms, TabNet's attention-based representation became competitive only after the training set was large enough to support it, whereas MLP delivered the strongest full-size accuracy with a simpler dense architecture.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This explains performance differences using attention-based representation learning and dense architecture complexity. It needs a source on TabNet/attention behavior in tabular prediction. | A source describing TabNet's sequential attention mechanism for tabular data, preferably including comparative performance or sample-size sensitivity against MLPs and tree methods. | `(TabNet OR "sequential attention") AND ("tabular data" OR "tabular learning") AND (MLP OR "multilayer perceptron" OR "dense neural network") AND ("sample size" OR performance)` |

## RC07

> Their positivity follows from averaging or selecting positive training responses, but those responses come from different influent states; a positive average therefore has no reason to satisfy the current case's affine balance $Ac_{out}=Ac_{in}$.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The first part relies on how ensemble and neighbor predictors are formed; the second part extends that to mass-balance inconsistency. This is a technical explanation, not just a result. | A source describing prediction by averaging/selecting training responses in random forests, boosting, or k-nearest neighbors, and ideally a source on mass-balance constraints not being preserved by unconstrained regression. | `("random forest" OR "k nearest neighbors" OR AdaBoost OR "extra trees") AND ("averaging predictions" OR "local averaging") AND ("mass conservation" OR "linear constraint" OR "constrained regression")` |

## RC08

> These patterns show why conservation and non-negativity must be diagnosed separately from nRMSE: as a surrogate learns the response surface more closely, it can still cross a zero boundary in a small component or miss a conserved invariant by a small but systematic amount.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This argues that accuracy metrics can improve while constraint violations persist, which is a general claim about constrained prediction. | A source showing that unconstrained machine-learning predictions can violate physical constraints even when prediction error is low, or that physical feasibility metrics should be evaluated separately from accuracy. | `("physics-informed machine learning" OR "constrained machine learning") AND ("physical constraints" OR "mass conservation" OR nonnegativity) AND ("prediction error" OR RMSE)` |

## RC09

> Symmetric regression losses and independently or jointly unconstrained output layers contain no barrier at zero; resolving these near-boundary tails can therefore lower nRMSE while allowing more small negative undershoots.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This explains negative predictions as a consequence of unconstrained losses/output layers. It needs support from optimization or machine-learning literature. | A source explaining that standard squared/symmetric regression losses and unconstrained output activations do not impose positivity, and that non-negative outputs require explicit constraints, transformations, or activations. | `("non-negative regression" OR "positivity constraint") AND ("squared loss" OR "mean squared error" OR "unconstrained output") AND ("neural network" OR regression)` |

## RC10

> In activated-sludge use, low predictive error without both mass conservation and non-negativity is not sufficient evidence that a predicted state is physically usable.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This is a strong domain-facing claim about physical usability in activated-sludge modeling. It should be supported by process-modeling or wastewater modeling references. | A source establishing that activated-sludge component states must obey mass-balance/stoichiometric constraints and non-negative concentrations for meaningful process interpretation or simulation. | `("activated sludge model" OR ASM2d OR wastewater) AND ("mass balance" OR stoichiometry) AND ("non-negative concentration" OR "state variables") AND ("model validity" OR feasibility)` |

## RC11

> This monotonic increase confirms that the exterior inputs formed a harder predictive problem than interpolation within the training envelope.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| The observed increase is internal, but the interpretation invokes the general difficulty of extrapolation outside the training domain. | A source discussing why machine-learning/surrogate models generally extrapolate less reliably outside the training distribution than they interpolate within it. | `("machine learning" OR "surrogate model") AND (extrapolation OR "out-of-distribution") AND (interpolation OR "training domain") AND ("prediction error" OR generalization)` |

## RC12

> The larger out-of-distribution benefit is consistent with the interpretation developed in-distribution: when extrapolation increases cross-component inconsistency, projecting the 20-component vector back to the feasible invariant balance can remove a larger share of the residual error.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This explains why projection helps more out of distribution by invoking constraint reconciliation of residuals. That explanation needs support from constrained projection or reconciliation literature. | A source showing that projecting predictions onto a constraint set can reduce infeasible residual components or improve consistency under conservation/linear constraints, especially for out-of-distribution or extrapolative predictions. | `("projection" OR "post-processing") AND ("linear constraints" OR "mass conservation" OR "constraint set") AND ("surrogate model" OR prediction) AND (extrapolation OR "out-of-distribution")` |

## RC13

> At the same time, the tree and neighbor exceptions show that non-negative averaging-based surrogates can require sizeable corrections when they are asked to extrapolate, and those corrections can increase aggregate nRMSE even while restoring physical admissibility.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This interprets tree/neighbor behavior under extrapolation and links it to correction size and accuracy tradeoffs. | A source on limited extrapolation behavior of tree-based and nearest-neighbor models, plus a source showing that enforcing constraints can trade off with predictive-error metrics. | `("tree based models" OR "random forest" OR "k nearest neighbors") AND extrapolation AND ("surrogate model" OR regression) AND ("constraint enforcement" OR "physical admissibility" OR "mass conservation")` |

## RC14

> Within that scope, the practical implication is strong: projection prevents accurate but physically inadmissible raw predictions from entering downstream interpretation, optimization, or control.

| Why this needs a citation | What the citation must contain | Effective Google Scholar keywords |
|---|---|---|
| This extends the result to downstream interpretation, optimization, and control, so it should be supported by literature on surrogate use in decision workflows and the risks of constraint violations. | A wastewater/process-systems source showing surrogates are used for optimization/control/decision support, and a source stating that physically constrained predictions are important for reliable downstream use. | `("wastewater" OR "activated sludge") AND ("surrogate model" OR "machine learning") AND (optimization OR control OR "decision support") AND ("physical constraints" OR "mass conservation" OR feasibility)` |
