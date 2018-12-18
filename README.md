# Apollonius

A library that creates an end-to-end splitting, SMOTE oversampling, and model-applying pipeline.
Used to diagnose highly imbalanced diseases.
Original built early 2017 -- my first foray into teaser neural networks.

**Core RAIL team:** Logan Graham, Ilan Price, Jonas Bovijn, and many others over the course of the initiative.

***Apollonius:** (of memphis) the [Greek physician credited with](https://en.wikipedia.org/wiki/History_of_diabetes) first naming diabetes.*

## Overview

1. `load_and_clean_data`: a file containing utilities to load and clean the large datafiles.
1. `sampling.py`: a file containing sampling functions that generate splits with custom oversampling & SMOTE ratios.
1. `SMOTE.py`: a custom implementation of [the SMOTE algorithm (Chawla et al., 2002)](https://arxiv.org/pdf/1106.1813.pdf) that specifically deals with the structure of the data. Optimized with vectorization.
1. `models.py`: Two model classes; one for a CatBoost model, and another for a Tensorflow-based neural net.
    1. To add a model, simply write a new class wrapper for the model, with the custom `predict_proba()` function.
1. `pipeline.py`: A pipeline class the combines the sampling splits with models, and trains and evaluates them.
1. `main.py`: an example of how a pipeline is run.

## Modelling Challenges
This was a very interesting project for two modelling challenges, beyond the expected missing data challenge:

1. **Highly imbalanced and sensitive classes:** Annual incidence of high-cost sensitive diseases is low. Loss functions / class weightings that don't capture this lead models to fail.
2. **Heterogenous data:** Patients are described with all sorts of datatypes. Models should account for this.

## TODO

- [ ] Probabilistic implementation of a SMOTE-like generative algorithm.
