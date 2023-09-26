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

Instead of using α and β, we use the mean of the distribution and a shape parameter:
    - $\mu = n \cdot \frac{\alpha}{\alpha + \beta}$
    - $p = log(\frac{alpha+beta}/2)$
    - $n=5$
When $p = 0$, that means that the distribution is flat (uniform),
when $p > 0$, the distribution is concave,
when $p < 0$, the distribution is convex.

We fit an intercept and then effects for:
    - The author being male;
    - The reviewer being male;
    - Their interaction;
in both parameters.

The priors are set as such:
    - Intercepts:
        - $\mu_0 ~ N(2.5, 0.2)$
        - $p_0 ~ N(0.0, 0.2)$
    - All Effects:
        - $effect ~ N(0.0, 0.2)$

The likelihood is then a Beta Binomial:
    - $\alpha = \frac{\mu \cdot e^{2 \cdot p}}{n}$
    - $\beta = e^{2 \cdot p} - \alpha$
    - $y ~ BetaBinomial(n, \alpha, \beta)$
