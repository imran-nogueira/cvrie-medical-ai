<picture><source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png"><img alt="CVRIE — Unsupervised clustering of patient testimonials and supervised skin-lesion classification." src="assets/banner-light.png"></picture>

**Epitech Bachelor · Year 2 · Machine Learning (B-SPE-311) · Sep 2026**
Source code is private under Epitech's academic policy. Happy to walk you through the notebooks or demo it live → **imran.nogueira@gmail.com**

## Overview

CVRIE V1 is a first version of a medical AI built on two classic ML tasks with scikit-learn: grouping free-text patient testimonials into themes without any labels, and classifying dermatoscope images into 7 diagnoses.

## Stack

| Layer | Tech |
|---|---|
| Language | Python |
| NLP | NLTK (lemmatization, stopwords), TF-IDF |
| ML | scikit-learn — TruncatedSVD, KMeans, DBSCAN, PCA, SVC |
| Analysis | pandas, NumPy, matplotlib, Jupyter |

## How it works — testimonial clustering

1. **Normalize** — NLTK lemmatization and stopword removal on 1,011 English testimonials.
2. **Vectorize** — TF-IDF (1,127 terms, `min_df=2`).
3. **Reduce** — TruncatedSVD to 100 dimensions (44.6% of variance kept).
4. **Cluster** — KMeans, with *k* chosen by elbow + silhouette → **k = 17**.
5. **Verify** — DBSCAN tested independently (k-distance plot, 9 `eps × min_samples` combinations): no stable density structure, which supports KMeans.
6. **Interpret** — qualitative reading of each cluster, 2D projection, per-cluster silhouette, wordclouds, live cluster prediction for a new testimonial.

**Supervised part:** HAM10000 (10,015 images, 7 classes, 58× imbalance) — split by lesion to prevent data leakage, StandardScaler + PCA + SVC, macro-F1 0.40.

## Key decisions

- **Split by lesion, not by image** — the same lesion photographed twice must never sit in both train and test.
- **A second algorithm as a referee** — DBSCAN was run to challenge the KMeans result, not to replace it.
- **Honest limits** — the silhouette score is low (0.062): the testimonials form a continuous cloud rather than sharply separated groups, and the report says so.

## Results

- Graded **36/37**.

## Contact

Notebook walkthrough or live demo on request → **imran.nogueira@gmail.com** · [Portfolio](https://imran-nogueira.github.io)
