# Bayesian-Conformal Uncertainty Quantification for Hierarchical Data

Code for our paper in *Scientific Reports* (Nature, 2026).

## What's this about?

Predicting hospital length of stay is useful, but predictions alone aren't enough, clinicians need to know *how confident* those predictions are. The problem: existing methods either give you reliable coverage guarantees (conformal prediction) or adaptive uncertainty (Bayesian), but not both.

We built a hybrid framework that combines both:
- **Conformal prediction** guarantees overall 95% coverage
- **Bayesian uncertainty** lets intervals adapt — narrower when confident, wider when uncertain

**Key result:** 21% narrower intervals for low-uncertainty cases, 6% wider for high-uncertainty cases, while achieving 94.3% overall coverage (target: 95%).

## Why This Matters

Bayesian models give well-calibrated uncertainty estimates, but those alone severely under-cover (only 14.1% coverage!). We use Bayesian uncertainty to *weight* conformal scores, getting the best of both worlds.

## Hierarchical Calibration Methods

Healthcare data violates the exchangeability assumption , patients within the same hospital are correlated. We implement three methods to handle this:

| Method | Description | Use When |
|--------|-------------|----------|
| `cdf_pooling` | Use all calibration data | Quick baseline |
| `subsampling_once` | One patient per hospital | Fast, breaks dependence |
| `repeated_subsampling` | B=50 bootstrap iterations | Most robust (recommended) |

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from src.models import HybridConformalHRF

# Train
model = HybridConformalHRF(method='repeated_subsampling')
model.fit(X_train, y_train, groups_train)
model.calibrate(X_calib, y_calib, groups_calib, alpha=0.05)

# Predict with adaptive intervals
predictions, lower, upper, uncertainties = model.predict(X_test, groups_test)

# Check coverage
coverage = ((y_test >= lower) & (y_test <= upper)).mean()
print(f"Coverage: {coverage:.1%}")  # Should be ~95%

# Check adaptation
low_unc = uncertainties < np.percentile(uncertainties, 25)
high_unc = uncertainties > np.percentile(uncertainties, 75)
print(f"Width (low uncertainty):  {(upper[low_unc] - lower[low_unc]).mean():.2f}")
print(f"Width (high uncertainty): {(upper[high_unc] - lower[high_unc]).mean():.2f}")
```

## Models

| Model | Coverage Guarantee | Adaptive Intervals |
|-------|-------------------|-------------------|
| `HierarchicalRandomForest` | ❌ | ❌ |
| `BayesianHRF` | ❌ (14.1% actual) | ✅ |
| `ConformalHRF` | ✅ | ❌ (uniform) |
| `HybridConformalHRF` | ✅ (94.3%) | ✅ |

## Files

```
├── src/
│   └── models.py           # All model implementations
├── notebooks/
│   └── CBHRF.ipynb         # Full analysis
├── requirements.txt
└── README.md
```

## Data

- **NIS 2019**: 61,538 patients from 3,793 hospitals across 4 US regions
- Available from [HCUP](https://www.hcup-us.ahrq.gov/nisoverview.jsp) (requires data use agreement)

## Citation

```
Shahbazi, M.A., Baheri, A., & Azadeh-Fard, N. (2026). A hierarchical conformal 
framework for uncertainty-aware length of stay prediction in multi-hospital 
settings. Scientific Reports. https://doi.org/10.1038/s41598-026-37450-w
```

## Contact

Marzieh Amiri Shahbazi — ma7684@g.rit.edu
