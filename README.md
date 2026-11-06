# What makes a book highly rated?

**Exploring book ratings through interpretable machine learning in Weka.**

[![Weka](https://img.shields.io/badge/tool-Weka-2563eb)](https://www.cs.waikato.ac.nz/ml/weka/)
[![Task](https://img.shields.io/badge/task-binary_classification-0f766e)](#the-question)
[![Status](https://img.shields.io/badge/status-experiments_pending-d97706)](#project-status)

*A Machine Learning project by **[Borja Barahona Gómez](https://github.com/bbarahon)**, developed for Practical Applications 1 + 2 of the MSc Data Science at UPM.*

The project investigates whether bibliographic metadata can distinguish books rated **4 stars or above** from those rated below 4. It compares model families and feature-selection strategies, with an emphasis on honest evaluation and understanding what the models learn.

## The question

Can genre, publication history, format and other book metadata help predict a book's rating category?

| Target | Definition |
| --- | --- |
| `high_rating` | Observed average rating ≥ 4 |
| `low_rating` | Observed average rating ≥ 1 and < 4 |

Books with no ratings are excluded. The threshold is fixed before model comparison. This is a **retrospective classification task** on an existing book catalogue; it does not establish causes of high ratings or demonstrate prediction before publication.

## Dataset at a glance

The following counts describe the corrected local dataset prepared for this project. They are **data-audit statistics, not model performance**.

| Item | Value |
| --- | ---: |
| Original records | 52,478 |
| Records after excluding books without ratings | 52,407 |
| Predictors | 33 |
| High-rating books | 28,695 · 54.75% |
| Low-rating books | 23,712 · 45.25% |

The source is a preprocessed *Best Books Ever* dataset. Its exact upstream release, collection date and redistribution terms still need to be documented. Data files are therefore not bundled in this repository. See [data provenance and local setup](data/README.md).

## Experimental design

The planned comparison crosses model families with four feature-selection settings:

| Setting | Implementation |
| --- | --- |
| All admissible predictors | No feature selection |
| Univariate filter | `InfoGainAttributeEval` + `Ranker` |
| Multivariate filter | `CfsSubsetEval` + `BestFirst` |
| Wrapper | `WrapperSubsetEval` + a documented search strategy |

**Candidate classifiers:** IBk, JRip, SMO, MultilayerPerceptron, J48, Logistic, NaiveBayes and BayesNet with TAN search. Ensembles include Bagging, AdaBoostM1, RandomForest and Vote. The final list must cover the algorithms taught in the current course.

A reserved test set is used only after configurations are fixed. Imputation, scaling and feature selection are learned within training folds. Accuracy is reported alongside balanced accuracy, F1, per-class recall, ROC-AUC and computational cost.

The full-sample majority proportion is **54.75%**. This is a descriptive reference, not a measured test score: the final ZeroR baseline must be trained and evaluated on the same split as the other models.


## Getting started

1. Obtain the corrected ARFF and place it at `data/local/goodreads_weka_preprocessed.arff` as described in [data setup](data/).
2. Validate its schema and basic invariants with Python 3:

   ```bash
   python scripts/validate_dataset.py data/local/goodreads_weka_preprocessed.arff
   ```

3. Open the ARFF in **Weka Explorer** and select `rating_class` as the class.
4. Follow the [step-by-step Weka guide](docs/weka-guide.es.md) to reserve the test set and build training-only preprocessing pipelines.
5. Record configurations, outputs and metrics using the [results template](results/metrics.csv).

The validation script uses only Python's standard library. Weka and Java are separate requirements; record their exact versions with each experiment.

## Repository map

| Path | Purpose |
| --- | --- |
| [`results/`](results/) | Reporting conventions and metrics, as well as more results of the different classifiers |
| [`results/`](data/) | The preprocessed data, as an .arff file |
| [`figures/`](figures/) | Existing visual material of the memory |

## Author and reuse

**Borja Barahona Gómez** · Physics graduate and MSc Data Science student · [GitHub](https://github.com/bbarahon)

This repository retains its existing [CC0 1.0 dedication](LICENSE). It does not establish redistribution rights for third-party Goodreads or upstream dataset content. Source attribution and dataset terms must be checked separately before distributing data.
