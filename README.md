# bog-nu-analysis
Bayesian Beta Binomial regression analysis of Bog.nu ratings.

## Inference

Install requirements:

```bash
pip install -r requirements.txt
```

Then run the scipt, which will save a parameter summary in the `results/` directory and figures in the `figures/` directory.
The data has to be in tab separated format under `dat/data_w_themes.csv`.

```bash
python3 beta_binomial_model.py
```

## Model description

We model book ratings using a beta-binomial distribution, that I reparameterized
for easier interpretation.

### Data
- `a` indicates whether the author is male (0 or 1).
- `r` indicates whether the reviewer is male (0 or 1).
- `y` is the `rating - 1` (integers between 0-5).

### Parameterization
Instead of using α and β, we use the mean of the distribution and a shape parameter:
- $\mu = n \cdot \frac{\alpha}{\alpha + \beta}$
- $p = log(\frac{alpha+beta}{2})$
- $n=5$

When $p = 0$, that means that the distribution is flat (uniform), <br>
when $p > 0$, the distribution is concave,<br>
when $p < 0$, the distribution is convex.

### Parameters
We fit an intercept and then effects for:
- The author being male;
- The reviewer being male;
- Their interaction;
in both parameters.

The priors are set as such:
- Intercepts:
    - $\mu_0 \sim N(2.5, 0.2)$
    - $p_0 \sim N(0.0, 0.2)$
- All Effects:
    - $\lambda \sim N(0.0, 0.2)$


The likelihood is then a Beta Binomial:
- $\mu = \mu_0 + \lambda_{\mu, a} \cdot a + \lambda_{\mu, r} \cdot r + \lambda_{\mu, a:r} \cdot (a\cdotr)$
- $p = p_0 + \lambda_{p, a} \cdot a + \lambda_{p, r} \cdot r + \lambda_{p, a:r} \cdot (a\cdotr)$
- $\alpha = \frac{\mu \cdot e^{2 \cdot p}}{n}$
- $\beta = e^{2 \cdot p} - \alpha$
- $y \sim BetaBinomial(n, \alpha, \beta)$
