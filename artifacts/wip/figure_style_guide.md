# Figure Style Guide for the Physical-Enforcement Benchmark

This internal authoring guide applies to the current study of post-prediction mass-conservation and non-negativity enforcement across nine conventional regressors.

The manuscript currently uses an inline graphical abstract, an evaluation-workflow diagram, an in-distribution data-scarcity figure, a projection-effect figure, and an OOD comparison figure. The supplement plans all-model learning curves, a per-component error atlas, physical-admissibility diagnostics, and OOD residual distributions.

Numerical result graphics are final only after the complete nine-model benchmark and validation checks have finished. Do not substitute estimated or partial values.

## Scientific Contract

Every final results figure must use the same benchmark contract as the tables:

- 22 inputs and 20 ASM2d-TSN effluent-component targets;
- eleven nested totals from 500 to 10,000;
- persistent seed-42 five-fold assignments shared by every model;
- one raw and one projected prediction for each scored row;
- primary aggregate metrics labeled exactly `nMSE`, `nRMSE`, and `nMAE`;
- COD, TN, TP, and TSS shown only as secondary quantities derived from the component vector;
- mild and severe OOD strata kept separate; and
- no result presented as final before the complete benchmark and validation checks finish.

## Descriptive Fold Summaries

Curves, bars, and table-linked callouts summarize the five outer-fold values as the arithmetic mean plus or minus the sample standard deviation. Compute the standard deviation with denominator four (`ddof=1`).

Use these phrases in captions and legends:

- `five-fold mean` for the central curve or marker;
- `sample SD` for an error bar or band; and
- `descriptive fold-to-fold variation` when interpretation is needed.

Do not label sample-SD bars as confidence intervals, error of the mean, or statistical evidence. Do not add p-values, stars, or pairwise-ranking annotations to this descriptive benchmark.

Minimal calculation pattern:

```python
fold_mean = fold_values.mean(axis=0)
fold_sample_sd = fold_values.std(axis=0, ddof=1)
```

## Metric Labels

The component-standardized aggregate labels are not interchangeable with unqualified physical-unit metrics.

- Use `nMSE` for mean squared standardized component error.
- Use `nRMSE` for the square root of nMSE.
- Use `nMAE` for mean absolute standardized component error.
- Use `ΔnRMSE (projected − raw)` for the paired projection effect.
- Add a component subscript, such as `nRMSE_j`, for component-specific standardized panels.
- Use unqualified `RMSE` or `MAE` only when the axis also states the physical component or composite and its unit.

Never label the primary component-standardized axis as `aggregate RMSE`, `effective RMSE`, or `test RMSE`.

## Visual Language

The visual style should remain restrained, readable in a single-column layout, and stable across main-text and supplementary figures:

- white figure and axes backgrounds;
- faint dotted grids only where they aid quantitative reading;
- vector line art and text whenever practical;
- direct axis labels with metric and unit;
- no decorative gradients, shadows, or three-dimensional effects;
- marker shapes as a second identity channel in addition to color; and
- concise titles, with scientific interpretation left to the caption and text.

Recommended plotting defaults:

```python
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```

## Palette and Semantics

Use the manuscript's muted earth-and-mineral palette:

```python
COLORS = {
    "deep_teal": "#264653",
    "mineral_green": "#2A9D8F",
    "muted_amber": "#E9C46A",
    "warm_sand": "#F4A261",
    "coral_rust": "#E76F51",
    "muted_plum": "#6D597A",
    "steel_blue": "#577590",
    "brick_red": "#BC4749",
    "slate_gray": "#8D99AE",
    "dark_gray": "#5C6770",
}
```

For paired prediction-state figures, semantics take priority over model identity:

- raw: coral rust;
- projected: mineral green;
- zero or tolerance reference: dark gray, dashed; and
- mechanistic target, if shown: deep teal or black.

For multi-model curves, keep one color and marker per model throughout the manuscript and supplement. The recommended final mapping is:

| Model | Color | Marker |
|---|---:|:---:|
| XGBoost | steel blue | circle |
| LightGBM | deep teal | circle |
| CatBoost | coral rust | square |
| AdaBoost | warm sand | square |
| Random Forest | slate gray | triangle down |
| SVR | brick red | triangle up |
| k-NN | muted amber | triangle right |
| PLS | dark gray | diamond |
| MLP | muted plum | pentagon |

No model should receive a privileged visual highlight. The scientific intervention is the shared projection, represented through raw/projected semantics.

## Main-Text Figure Recipes

### Graphical Abstract and Evaluation Workflow

The workflow should read left to right:

1. accepted mechanistic states and the 22 inputs;
2. nine conventional learners;
3. raw 20-component prediction;
4. Kircher--Votsmeier projection; and
5. paired accuracy, physical, OOD, and timing evaluation.

Use rounded boxes, one arrow direction, and no model-specific architecture details. The diagram should make clear that COD, TN, TP, and TSS are derived only after component-space prediction and correction. Do not show artificial clipping of mechanistic rates; the simulator evaluates the rate vector directly.

### In-Distribution Data-Scarcity Figure

The left panel shows `nRMSE` against total dataset size. The right panel shows model training time against the same x-axis.

- Plot five-fold means as lines with markers.
- Plot sample SD as symmetric bars or a light band.
- State `five-fold mean with sample SD` in the caption.
- Use all eleven nested sizes in final exports, even if a draft layout displays fewer ticks.
- Label the right panel `Training time (s)`.
- Use a logarithmic time axis only if the observed range warrants it.
- Keep inference latency out of the setup-time panel; report it separately.

### Projection-Effect Figure

Plot

```text
ΔnRMSE = nRMSE_projected − nRMSE_raw
```

against total dataset size. A horizontal zero line is mandatory. Negative values denote lower nRMSE after projection and positive values denote higher nRMSE after projection. Use the same model colors as the learning curves. Final curves must show five-fold means with sample-SD bars or bands; the caption must also state that feasibility and predictive error are distinct outcomes.

### OOD Raw/Projected Figure

Use grouped bars, paired points, or slope markers for raw versus projected nRMSE within each model. Keep mild and severe strata in separate panels if both are shown. The legend must map raw to coral rust and projected to mineral green. Avoid implying that projected means accurate: a note or caption should state that projection restores the declared physical contract but need not reduce extrapolation error.

If model names make vertical bars crowded, use horizontal paired points instead. Preserve the same model order in all OOD panels.

## Supplementary Figure Recipes

### All-Model Learning Curves

Show all nine models over all eleven data sizes. If one panel is too dense, split the learners into aligned panels with identical axes. Do not rank models by eye through line thickness; use equal widths and the fixed color/marker mapping.

### Per-Component Error Atlas

Use the fixed 20-component order on the x-axis and models on the y-axis. Separate raw nRMSE$_j$, projected nRMSE$_j$, and projection displacement into aligned panels. A shared color scale is appropriate only when the metric and range are identical. Physical-unit component panels need their own units or separate scales.

For annotated heatmaps, change text color according to cell luminance and omit cell numbers when the matrix becomes illegible at journal size.

### Physical-Admissibility Diagnostics

Keep mass-conservation residual, non-negativity frequency/magnitude, backtracking activation, and projection displacement in separate aligned panels. Do not merge differently scaled quantities on dual y-axes. A log scale may be used for positive residual magnitudes, but zero values must be handled and described explicitly.

### OOD Distributions

Show sample-level residual or displacement distributions separately for mild and severe OOD inputs. Paired raw/projected points are preferable when row identity matters. Use identical limits across severity panels when direct comparison is intended.

## Placeholder Treatment

Until the complete benchmark is available:

- retain `Illustrative placeholder—not experimental data` at the start of every numerical figure caption;
- keep draft-only footer text if a figure is exported externally;
- do not remove placeholder panels simply because final values are unavailable;
- do not interpolate missing folds or data sizes; and
- do not mix completed and provisional model outputs in a nominally final figure.

Recommended draft footer:

```text
Illustrative layout only; replace after the complete final benchmark.
```

Remove the footer only when every plotted value has passed the final cross-check.

## Accessibility and Layout Checks

- Verify that every series remains identifiable in grayscale through marker or line style.
- Avoid red/green as the only raw/projected distinction; use fill, marker, or hatching as well.
- Keep model labels horizontal when possible and rotate them no more than 45 degrees.
- Use at least 7-point text at final rendered size.
- Define every abbreviation in the caption or main text.
- Keep legends outside dense plotting regions.
- Use consistent decimal precision within a panel.

## Export Rules

- Prefer vector PDF for line art and plots.
- Embed fonts and inspect the final rendered size.
- Use tight bounding boxes without clipping labels or sample-SD bars.
- Keep final figure filenames flat and stable for submission packaging.
- Do not display local filesystem paths, repository locations, user names, run-history notes, or code-cell identifiers in reader-facing figures.
- Preserve a raster version only when the submission system explicitly requires it.

## Final Figure Checklist

Before any result figure is treated as final, confirm that:

1. the figure comes from the completed benchmark rather than illustrative arrays;
2. all five folds use the persistent shared assignment;
3. sample SD uses denominator four;
4. metric labels are exactly nMSE, nRMSE, nMAE, or a clearly unit-bearing physical metric;
5. raw and projected outputs refer to identical rows;
6. every model series uses its frozen accepted configuration;
7. OOD severity and in-distribution results are not pooled;
8. captions describe fold summaries as five-fold mean with sample SD;
9. placeholder wording is removed only after every value is verified; and
10. no reader-facing internal path or development record remains.
