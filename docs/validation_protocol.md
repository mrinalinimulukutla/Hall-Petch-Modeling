# Validation protocol

The central methodological claim of the manuscript is that the validation
split must match the decision the model is meant to support. This note states
what each protocol does and does not test.

## Pooled Q², not mean fold R²

For every protocol, predictions from all held-out records are pooled and
scored once:

    Q² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²

This is reported as the cross-validated R². It is **not** the mean of
fold-specific R² values, which is a different and generally more optimistic
quantity. Where an archived number is a mean of fold-wise R², the manuscript
says so and does not compare it directly with the pooled values.

## The three internal protocols

**Shuffled 5-fold (seed 42).** Interpolation within the pooled dataset. The
weakest test here; reported for continuity with common practice.

**Leave-one-out.** Record-level interpolation. Critically, 21 of the 94
records share an exact composition with another record, so for those a
chemically identical alloy can remain in the training set. LOO is therefore
*not* a test on unseen compositions and is interpreted alongside the
repeated-composition audit.

**Leave-one-batch-out.** Each of the six Bayesian-optimisation batches is held
out in turn. This is the primary grouped transfer test: it asks whether a
model carries to a new experimental iteration. One composition appears in both
CBB and CBC, so LOBO reduces but does not eliminate exact-composition overlap.

Batch-clustered percentile intervals come from resampling the six batches with
replacement 1000 times, keeping each batch intact. With only six clusters
these show sensitivity to which batches were sampled; they are not precise
inferential confidence intervals.

## Beyond the internal protocols

**Literature stress test.** Three evidence classes are kept separate: 54
direct strength measurements, 3 values reconstructed from published Hall–Petch
fits, and 25 proxies converted from hardness. The compilation mixes sources
and test modes and was filtered using outcomes, so it exposes domain
sensitivity and unsafe equations but cannot give an unbiased external
estimate. Every evaluated model is worse than predicting its mean.

**Singularity audit.** Each reported closed form is checked for small or
transformed denominators, sign changes, non-finite values and extreme
predictions across the observed and literature ranges. Two findings matter:
one SISSO form places a shear-modulus difference in a denominator and
destabilises when the constituent moduli are similar; and one PySR form hides
a pole inside a protected square root, so it stays real but diverges just
outside the sampled envelope.

## What a reader should take from the numbers

A score is only interpretable next to the question its split asks. The same
fitted model supports different conclusions at different validation scales,
and the honest summary of a model is the *change* across protocols rather than
its best single column.
