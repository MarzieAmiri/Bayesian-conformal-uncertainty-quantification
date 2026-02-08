# Hierarchical Conformal Prediction for Hospital Length of Stay

Code for our paper in *Scientific Reports* (Nature, 2026).

## What's this about?

Predicting hospital length of stay is useful, but predictions alone aren't enough, clinicians need to know *how confident* those predictions are. The problem: existing methods either give you reliable coverage guarantees (conformal prediction) or adaptive uncertainty (Bayesian), but not both.

We built a hybrid framework that combines both:
- **Conformal prediction** gives you guaranteed 95% coverage (your true value falls in the interval 95% of the time)
- **Bayesian uncertainty** lets intervals adapt — narrower when confident, wider when uncertain

The result: 21% narrower intervals for easy cases, while still maintaining coverage for hard cases.

## The Key Insight

Bayesian models give well-calibrated uncertainty estimates, but those alone severely under-cover (only 14%!). We use Bayesian uncertainty to *weight* conformal scores, getting the best of both worlds.

## Methods

We implemented four approaches:

| Model | Coverage Guarantee | Adaptive Intervals |
|-------|-------------------|-------------------|
| Standard HRF | ❌ | ❌ |
| Bayesian HRF | ❌ | ✅ |
| Conformal HRF | ✅ | ❌ |
| **Hybrid (ours)** | ✅ | ✅ |

## Results

Tested on 61,538 patients from 3,793 hospitals:

- **Coverage:** 94.3% (target: 95%)
- **Low-uncertainty cases:** 21% narrower intervals
- **High-uncertainty cases:** 6% wider intervals (appropriate conservatism)
- **Bayesian alone:** Only 14.1% coverage (not usable clinically)

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from src.models import HybridConformalHRF

# Train
model = HybridConformalHRF()
model.fit(X_train, y_train, groups_train)
model.calibrate(X_calib, y_calib, groups_calib, alpha=0.05)

# Predict with intervals
predictions, lower, upper = model.predict(X_test, groups_test)

# Check coverage
coverage = ((y_test >= lower) & (y_test <= upper)).mean()
print(f"Coverage: {coverage:.1%}")  # Should be ~95%
```

## Data

- **NIS 2019**: Available from [HCUP](https://www.hcup-us.ahrq.gov/nisoverview.jsp) (requires data use agreement)

## Citation

```
Shahbazi, M.A., Baheri, A., & Azadeh-Fard, N. (2026). A hierarchical conformal 
framework for uncertainty-aware length of stay prediction in multi-hospital 
settings. Scientific Reports. https://doi.org/10.1038/s41598-026-37450-w
```

## Contact

Marzieh Amiri Shahbazi — ma7684@g.rit.edu
