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
