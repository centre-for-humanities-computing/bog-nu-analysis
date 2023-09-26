"""Script for sampling the Beta Binomial model,
producing figures and parameter summary"""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

print("Preprocessing data...")
data = pd.read_csv("dat/data_w_themes.csv", sep="\t")
# Drop incomplete rows
data = data.dropna(subset=["rev_gender_updated", "author_gender"])
# I renamed so that it's easier to access
data = data.rename(
    columns={"rev_gender_updated": "reviewer", "author_gender": "author"}
)
data["grades"] = data["grades"].astype(int)

with pm.Model() as model:
    """
    This is a Beta Binomial model of ratings.
    I decided to reparameterize, so things are easier to interpret.
    One of the parameters is the mean of the distribution:
        mu = n * (alpha/(alpha+beta))
    The other one is a shape parameter:
        p = log((alpha+beta)/2)
    If p = 0 then the distribution is uniform.
    Higher values indicate more pointy distributions.
    Lower values mean convex distributions.
    I fitted effects in both parameters for the:
        1. Author's gender.
        2. Reviewer's gender.
        3. Their interaction.
    Intercept is female, female, any deviation from this is modelled as an effect.
    """
    # Data
    reviewer = pm.MutableData(
        "reviewer", data["reviewer"].map({"male": 1, "female": 0})
    )
    author = pm.MutableData(
        "author", data["author"].map({"male": 1, "female": 0})
    )
    # Priors
    # Mean centered originally at 2.5, cause that's the middle
    mu_0 = pm.Normal("mu_0", 2.5, 0.2)
    # Uniform is the default prior
    p_0 = pm.Normal("p_0", 0, 0.2)
    author_mu = pm.Normal("author_mu", 0, 0.2)
    author_p = pm.Normal("author_p", 0, 0.2)
    reviewer_mu = pm.Normal("reviewer_mu", 0, 0.2)
    reviewer_p = pm.Normal("reviewer_p", 0, 0.2)
    interaction_mu = pm.Normal("interaction_mu", 0, 0.2)
    interaction_p = pm.Normal("interaction_p", 0, 0.2)
    # Outcome
    n = 5
    mu = (
        mu_0
        + author_mu * author
        + reviewer_mu * reviewer
        + interaction_mu * author * reviewer
    )
    p = (
        p_0
        + author_p * author
        + reviewer_p * reviewer
        + interaction_p * author * reviewer
    )
    # Variable transformation back to original parameter space
    alpha = mu * pt.exp(2 * p) / n
    beta = pt.exp(2 * p) - alpha
    pm.BetaBinomial(
        "outcome",
        n=n,
        alpha=alpha,
        beta=beta,
        observed=data["grades"]
        - 1,  # We substract one so that ratings are from 0 to 5
    )


# We sample the posterior and the posterior predictive
with model:
    print("Sampling posterior...")
    idata = pm.sample()
    print("Sampling posterior predictive...")
    idata.extend(pm.sample_posterior_predictive(idata))


print("\n-----------------")
print("Parameter Summary")
print("-----------------\n")
summary = az.summary(idata)
print(summary)
print("\n")

print("Creating output directories...")
figures_path = Path("figures")
figures_path.mkdir(exist_ok=True)
results_path = Path("results")
results_path.mkdir(exist_ok=True)
# Creating directory if it doesn't exist
summary.to_csv(results_path.joinpath("summary.csv"))

print("Producing figures...")
az.plot_forest(idata, var_names={"_p"}, filter_vars="like")
plt.tight_layout()
plt.savefig(figures_path.joinpath("p_effects_forest.png"), dpi=300)

az.plot_forest(idata, var_names={"_mu"}, filter_vars="like")
plt.tight_layout()
plt.savefig(figures_path.joinpath("mu_effects_forest.png"), dpi=300)

az.plot_ppc(idata)
plt.tight_layout()
plt.savefig(figures_path.joinpath("posterior_predictive.png"), dpi=300)

print("DONE")
