# Paired Boarding Strategy Report

Generated: 2026-04-19 22:24 UTC

## Study Design
- Independent variable: boarding strategy
- Primary dependent variable: total boarding time
- Strategies: back-to-front zonal vs modified reverse pyramid
- Fixed assumptions: aircraft layout and cabin topology, seat map and class structure, load factor, luggage probability, active boarding doors, behavioral parameter settings, simulation logic and completion condition

## Run Summary
| required_replications | replications_attempted | completed_pairs | master_seed | load_factor | luggage_probability | cross_zone_violation_rate | paired_runs | failed_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 274 | 274 | 274 | 2.026e+07 | 0.85 | 0.75 | 0.05 | 274 | 0 |

## Descriptive Statistics
| strategy | n_completed | mean | std | median | min | q10 | q25 | q75 | q90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyramid | 274 | 504.091 | 39.954 | 499.25 | 406.5 | 457.15 | 476 | 530.875 | 560.05 | 616.5 |
| std | 274 | 732.443 | 62.689 | 728.25 | 593.5 | 655.3 | 689.75 | 773.875 | 816.85 | 940 |

## Paired Inference
| n_pairs | mean_paired_difference | mean_relative_improvement | normality_p_value | selected_test | test_statistic | p_value | effect_size_vargha_delaney_A | effect_size_paired_d | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 274 | 228.352 | 0.308 |  | one-sided paired t-test (zonal - pyramid > 0) | 60.104 | 9.441e-160 | 1 | 3.631 | 220.873 | 235.832 |

## Paired Metrics
| replication_id | boarding_time_zonal | boarding_time_pyramid | difference | ratio | relative_improvement |
| --- | --- | --- | --- | --- | --- |
| 1 | 731 | 471 | 260 | 0.644 | 0.356 |
| 2 | 673 | 487.5 | 185.5 | 0.724 | 0.276 |
| 3 | 848 | 478 | 370 | 0.564 | 0.436 |
| 4 | 786 | 465.5 | 320.5 | 0.592 | 0.408 |
| 5 | 817 | 593.5 | 223.5 | 0.726 | 0.274 |
| 6 | 853.5 | 541.5 | 312 | 0.634 | 0.366 |
| 7 | 783.5 | 457.5 | 326 | 0.584 | 0.416 |
| 8 | 747 | 516 | 231 | 0.691 | 0.309 |
| 9 | 724.5 | 526 | 198.5 | 0.726 | 0.274 |
| 10 | 699 | 506 | 193 | 0.724 | 0.276 |
| 11 | 622 | 518.5 | 103.5 | 0.834 | 0.166 |
| 12 | 796 | 521 | 275 | 0.655 | 0.345 |

## Figures
![Boarding time by strategy](fig_boxplot_boarding_time.png)

![Histogram of paired differences with fitted normal density](fig_hist_paired_differences.png)

![Q-Q plot of paired differences](fig_qq_paired_differences.png)

### Boarding time by strategy
- What it shows: distribution of total boarding times per strategy (median, spread, and outliers).
- How to read it: lower boxes/medians mean faster boarding; narrower spread means more consistency.
- Conclusion: compare central tendency and spread to assess speed and reliability tradeoffs.

### Histogram of paired differences with fitted normal density
- What it shows: frequency distribution of paired differences (zonal - pyramid) with a fitted normal curve overlay.
- How to read it: values above 0 indicate pyramid is faster; the red curve is a visual normal-reference guide.
- Conclusion: center and spread indicate average gain and variability; compare bars vs curve for rough normality fit.

### Q-Q plot of paired differences
- What it shows: observed quantiles of paired differences against theoretical normal quantiles.
- How to read it: points close to a straight line suggest approximate normality; systematic bends indicate departures.
- Conclusion: supports whether the paired t-test normality assumption is reasonable.

## Notes
- Replication summary: required=274, attempted=274, completed_pairs=274.
- Selected test: one-sided paired t-test (zonal - pyramid > 0), p-value: 9.441e-160, mean paired difference: 228.352 s.
- Best mean boarding time: pyramid at 504.091 s.
- Completed pairs: 274.
- Failed runs: 0.
