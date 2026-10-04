# Satellite Orbit Classification

## Machine Learning Classification of Satellite Orbital Classes Using CelesTrak Active Satellites

[![Dataset](https://img.shields.io/badge/Dataset-CelesTrak%20Active%20Satellites-0ea5e9)](https://celestrak.org/norad/elements/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-f59e0b)](https://colab.research.google.com/)
[![Language](https://img.shields.io/badge/Python-3.x-3776ab)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-scikit--learn-f7931e)](https://scikit-learn.org/)

> A reproducible machine-learning experiment that classifies active satellites into **LEO, MEO, GEO, and HEO** orbital classes using CelesTrak General Perturbation (GP) orbital parameters.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Research Question](#research-question)
* [Why This Dataset](#why-this-dataset)
* [Dataset](#dataset)
* [CelesTrak GP Data](#celestrak-gp-data)
* [Raw Features](#raw-features)
* [Data Snapshot Strategy](#data-snapshot-strategy)
* [Orbit-Class Construction](#orbit-class-construction)
* [ESA Orbital Taxonomy](#esa-orbital-taxonomy)
* [Four-Class Aggregation](#four-class-aggregation)
* [Target Leakage and Feature Selection](#target-leakage-and-feature-selection)
* [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
* [Train Validation Test Split](#train-validation-test-split)
* [Machine Learning Models](#machine-learning-models)
* [Evaluation](#evaluation)
* [Results](#results)
* [Why the Best Model Won](#why-the-best-model-won)
* [Ablation Study](#ablation-study)
* [Reproducibility](#reproducibility)
* [Google Drive and GitHub Workflow](#google-drive-and-github-workflow)
* [Repository Structure](#repository-structure)
* [Running the Project](#running-the-project)
* [Limitations](#limitations)
* [Scientific and Methodological Notes](#scientific-and-methodological-notes)
* [References](#references)

---

# Project Overview

This project investigates whether machine-learning models can classify the orbital regime of active satellites using General Perturbation (GP) orbital parameters.

The project uses the **CelesTrak Active Satellites** dataset as the sole primary satellite dataset. CelesTrak provides current GP data in CSV and JSON formats using the same orbital-element definitions associated with the OMM/GP data framework. The Active Satellites group is available under CelesTrak's Supplemental GP Data.

The machine-learning task is a four-class classification problem:

* **LEO** — Low Earth Orbit
* **MEO** — Medium Earth Orbit
* **GEO** — Geostationary Orbit
* **HEO** — Assignment-level High-Earth umbrella class

The project deliberately separates:

1. **Physics-based target construction**
2. **Machine-learning feature selection**
3. **Model training**
4. **Model evaluation**
5. **Ablation analysis**

This separation is important because the orbital class is not supplied as a label in the CelesTrak Active Satellites CSV. Instead, the label is derived from orbital mechanics and a published ESA orbital-regime taxonomy.

---

# Research Question

The central research question is:

> **How accurately can machine-learning models classify active satellites into LEO, MEO, GEO and HEO using GP orbital parameters after excluding the orbital variables directly used to construct the target labels?**

A secondary question is:

> **Which properties of the remaining orbital data explain differences in model performance?**

The project also performs an ablation experiment to test whether a specific remaining orbital parameter contributes meaningful predictive information.

---

# Why This Dataset

The dataset was selected for three reasons.

### 1. Direct relevance to Space Science

The project uses real orbital-element data for active Earth-orbiting satellites, making it directly connected to satellite-orbit analysis and Space Science rather than being a generic machine-learning benchmark.

### 2. Open and machine-readable

CelesTrak provides GP data in CSV and JSON formats, allowing the experiment to be downloaded programmatically and reproduced without manual extraction. CelesTrak documents its GP query system and supports `GROUP` queries with CSV output.

### 3. Continuously updated

The Active Satellites dataset is a live catalog rather than a permanently frozen benchmark. CelesTrak publishes the current data timestamp and updates its GP data continuously. A dated snapshot is therefore recorded for every experiment run.

This makes the project both:

* **realistic**, because it uses current satellite data;
* **reproducible**, because the exact snapshot used for each run is archived.

---

# Dataset

## Primary Dataset

**Source:** CelesTrak NORAD GP Element Sets → Supplemental GP Data → Special-Interest Satellites → Active Satellites.

**Format:** CSV

The dataset contains current GP orbital-element information for active satellites.

The current CelesTrak GP query system supports CSV output and uses the GP/OMM keyword definitions. CelesTrak also notes that CSV and JSON formats avoid the five-digit catalog-number limitation of legacy TLE formats.

### Why CSV?

CSV was selected because it is:

* directly supported by CelesTrak;
* easy to parse with pandas;
* easy to inspect;
* compact;
* suitable for reproducible programmatic processing.

---

# CelesTrak GP Data

CelesTrak's GP query structure allows a group to be requested directly using a URL of the form:

`gp.php?GROUP=ACTIVE&FORMAT=CSV`

The notebook uses the Active Satellites group as the primary data source.

CelesTrak currently recommends using its newer non-TLE formats for current catalogs because six-digit catalog numbers now exist. As of July 11, 2026, CelesTrak reported that the catalog had exceeded the five-digit limit supported by legacy TLE formats.

For this reason, this project intentionally uses the **CSV GP representation rather than legacy TLE text files**.

---

# Raw Features

The Active Satellites GP CSV contains fields such as:

| Feature               | Description / role                   |
| --------------------- | ------------------------------------ |
| `OBJECT_NAME`         | Satellite/object name                |
| `OBJECT_ID`           | International designator             |
| `EPOCH`               | Epoch of the GP element set          |
| `MEAN_MOTION`         | Mean orbital motion, revolutions/day |
| `ECCENTRICITY`        | Orbital eccentricity                 |
| `INCLINATION`         | Orbital inclination, degrees         |
| `RA_OF_ASC_NODE`      | Right ascension of ascending node    |
| `ARG_OF_PERICENTER`   | Argument of pericenter               |
| `MEAN_ANOMALY`        | Mean anomaly                         |
| `EPHEMERIS_TYPE`      | Ephemeris type                       |
| `CLASSIFICATION_TYPE` | Classification/provenance field      |
| `NORAD_CAT_ID`        | Catalog identifier                   |
| `ELEMENT_SET_NO`      | Element-set number                   |
| `REV_AT_EPOCH`        | Revolution number at epoch           |
| `BSTAR`               | Drag-related GP parameter            |
| `MEAN_MOTION_DOT`     | First mean-motion derivative         |
| `MEAN_MOTION_DDOT`    | Second mean-motion derivative        |

The GP data are orbital-element representations suitable for SGP4-based propagation and analysis. CelesTrak documents the definitions and data formats in its GP documentation.

---

# Data Snapshot Strategy

Because CelesTrak is a live dataset, this project uses a daily snapshot strategy.

At the beginning of a run:

1. Google Colab checks Google Drive for today's dated Active Satellites CSV.
2. If today's file exists, the notebook uses it without contacting CelesTrak.
3. If today's file does not exist, the notebook requests the current Active Satellites CSV from CelesTrak.
4. The downloaded file is validated.
5. The new file is saved to Google Drive with the date in its filename.
6. Older daily raw CelesTrak snapshots in the Google Drive cache are removed.
7. The exact daily snapshot is also archived permanently in GitHub.

Example:

```text
active_satellites_2026-10-04.csv
```

Google Drive therefore acts as the current working cache, while GitHub acts as the historical public archive.

This design also avoids repeatedly downloading the same CelesTrak GP group unnecessarily. CelesTrak's usage policy asks users to download GP data only once per update.

---

# Orbit-Class Construction

The Active Satellites CSV does **not** contain a ready-made `ORBIT_CLASS` column.

Therefore, the target is constructed using orbital mechanics.

From mean motion:

$$
n = \frac{2\pi M}{86400}
$$

where \(M\) is the mean motion in revolutions/day.

The semi-major axis is then derived from Kepler's third law:

$$
a = \left(\frac{\mu}{n^2}\right)^{1/3}
$$

where:

$$
\mu = 398600.4418\;km^3/s^2
$$

Perigee and apogee radii are:

$$
r_p = a(1-e)
$$

$$
r_a = a(1+e)
$$

and altitude is calculated relative to the adopted Earth radius.

The resulting quantities include:

* semi-major axis;
* perigee altitude;
* apogee altitude;
* orbital period.

These quantities are then compared against the orbital-regime definitions described by ESA.

---

# ESA Orbital Taxonomy

Rather than inventing arbitrary boundaries, the project uses **ESA's Annual Space Environment Report, Table 1.2** as the scientific basis for the orbital-regime classification.

ESA defines orbital classes using combinations of:

* semi-major axis \(a\);
* eccentricity \(e\);
* inclination \(i\);
* perigee height \(h_p\);
* apogee height \(h_a\).

ESA's detailed taxonomy includes:

* GEO — Geostationary Orbit
* IGO — Inclined Geosynchronous Orbit
* EGO — Extended Geostationary Orbit
* NSO — Navigation Satellites Orbit
* GTO — GEO Transfer Orbit
* MEO — Medium Earth Orbit
* GHO — GEO-superGEO Crossing Orbit
* LEO — Low Earth Orbit
* HAO — High Altitude Earth Orbit
* MGO — MEO-GEO Crossing Orbit
* HEO — Highly Eccentric Earth Orbit
* LMO — LEO-MEO Crossing Orbit
* plus undefined/escape cases in the full ESA taxonomy.

ESA's current report gives, for example:

* LEO: perigee and apogee within 0–2,000 km;
* MEO: perigee and apogee within 2,000–31,570 km;
* GEO: inclination 0–25° with perigee and apogee between 35,586 and 35,986 km;
* LMO: perigee 0–2,000 km and apogee 2,000–31,570 km;
* MGO: perigee 2,000–31,570 km and apogee 31,570–40,002 km;
* GTO: perigee 0–2,000 km and apogee 31,570–40,002 km;
* HEO: perigee 0–31,570 km and apogee above 40,002 km;
* HAO: both perigee and apogee above 40,002 km.

---

# Four-Class Aggregation

The assignment requires only four classes, so the detailed ESA taxonomy is explicitly aggregated into:

```text
LEO
MEO
GEO
HEO
```

The mapping used by this project is:

| ESA regime | Assignment class |
| ---------- | ---------------- |
| LEO        | LEO              |
| LMO        | MEO              |
| MEO        | MEO              |
| NSO        | MEO              |
| GEO        | GEO              |
| GTO        | HEO              |
| MGO        | HEO              |
| IGO        | HEO              |
| EGO        | HEO              |
| GHO        | HEO              |
| HEO        | HEO              |
| HAO        | HEO              |

This aggregation is **specific to this assignment**. It should not be interpreted as ESA officially defining these twelve detailed regimes as four classes.

In particular, the project uses `HEO` as an **assignment-level high-Earth umbrella** containing several ESA high-altitude and crossing regimes.

This distinction is retained explicitly in the processed dataset through two separate fields:

```text
ESA_ORBIT_REGIME
ORBIT_CLASS
```

For example:

```text
ESA_ORBIT_REGIME = GTO
ORBIT_CLASS      = HEO
```

This preserves the original detailed physical interpretation while providing the four-class target required by the assignment.

---

# Target Leakage and Feature Selection

A major methodological issue is that the orbit-class label is derived from orbital variables.

The target-construction process explicitly uses:

```text
MEAN_MOTION
ECCENTRICITY
INCLINATION
```

and quantities derived from them:

```text
SEMI_MAJOR_AXIS
PERIGEE_ALTITUDE
APOGEE_ALTITUDE
ORBITAL_PERIOD
```

To create a strict leakage-controlled experiment, these variables are **not provided to the machine-learning models**.

The final ML feature set is therefore:

```text
RA_OF_ASC_NODE
ARG_OF_PERICENTER
MEAN_ANOMALY
BSTAR
MEAN_MOTION_DOT
MEAN_MOTION_DDOT
```

The following are also excluded because they identify the satellite or describe metadata rather than being intended orbital predictors:

```text
OBJECT_NAME
OBJECT_ID
NORAD_CAT_ID
EPOCH
ELEMENT_SET_NO
REV_AT_EPOCH
EPHEMERIS_TYPE
CLASSIFICATION_TYPE
```

### Why exclude inclination?

Inclination is explicitly part of the GEO definition used in the target construction. ESA's GEO regime includes an inclination range of 0–25° in combination with the GEO altitude window.

Therefore, including inclination as an ML predictor after using it to construct the target would make the leakage-control argument inconsistent.

The project consequently excludes:

```text
MEAN_MOTION
ECCENTRICITY
INCLINATION
```

from model input.

---

# Data Cleaning and Preprocessing

The raw dataset is checked for:

* missing values;
* duplicate complete rows;
* duplicate NORAD catalog IDs;
* invalid mean-motion values;
* invalid eccentricities;
* invalid inclinations;
* unusable orbital records;
* class imbalance.

### Numeric conversion

The following numerical columns are explicitly converted to numeric types:

```text
MEAN_MOTION
ECCENTRICITY
INCLINATION
RA_OF_ASC_NODE
ARG_OF_PERICENTER
MEAN_ANOMALY
BSTAR
MEAN_MOTION_DOT
MEAN_MOTION_DDOT
```

### Duplicate handling

Duplicate NORAD catalog entries are reduced to one current record per satellite in the downloaded snapshot.

### Missing values

The machine-learning pipelines use:

```text
Median imputation
```

rather than deleting every record containing a missing predictor.

### Scaling

A `StandardScaler` is applied after imputation.

The same preprocessing design is used for every model.

Importantly, preprocessing is included inside each scikit-learn `Pipeline`, preventing the scaler from learning from validation or test data.

---

# Train Validation Test Split

The dataset is divided into:

| Split      | Proportion |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Test       |        15% |

The split is:

* stratified by orbit class;
* reproducible;
* generated once and then used consistently across the models.

The experiment uses:

```text
random_state = 42
```

The validation set is used for model selection.

The test set is retained as a final held-out evaluation set.

---

# Machine Learning Models

Seven algorithms are evaluated.

## 1. Dummy Baseline

The `DummyClassifier` provides a deliberately simple baseline based on the most frequent class.

Purpose:

> Determine whether the actual machine-learning models learn useful orbital structure beyond class-frequency guessing.

---

## 2. Logistic Regression

Logistic Regression provides a linear baseline.

Question tested:

> Can the remaining six orbital features separate the four classes using approximately linear decision boundaries?

---

## 3. Random Forest

Random Forest provides a nonlinear tree-based ensemble.

Question tested:

> Do nonlinear thresholds and interactions among the remaining orbital parameters improve classification?

---

## 4. HistGradientBoosting

HistGradientBoosting provides a second nonlinear tree-based approach using sequential boosting.

Question tested:

> Can sequential refinement of tree-based predictions extract additional information from overlapping orbital-feature distributions?

---

## 5. Extra Trees

Extra Trees provides an alternative randomized tree ensemble.

It introduces additional randomness in the construction of tree splits compared with conventional Random Forests.

Question tested:

> Does a more randomized partitioning of the six-dimensional orbital feature space improve generalization?

---

## 6. RBF Support Vector Machine

An RBF-kernel SVM provides a nonlinear kernel-based approach.

Question tested:

> Are the orbital classes better represented by nonlinear boundaries in the standardized feature space?

The model receives the same imputed and standardized inputs as the other models.

---

## 7. K-Nearest Neighbors

KNN provides an instance-based perspective.

Question tested:

> Do satellites with similar remaining orbital-feature values tend to share the same orbital class?

Because KNN uses distances, standardization is particularly important.

---

# Experimental Controls

Every algorithm uses:

* the same dataset;
* the same target;
* the same six ML features;
* the same imputation;
* the same standardization;
* the same train/validation/test split;
* the same random seed where the algorithm supports one;
* the same evaluation procedure.

This makes the comparison focus on the models rather than differences in the data preparation.

---

# Evaluation

## Primary Metric: Macro-F1

The primary metric is:

**Macro-F1**

Macro-F1 is the arithmetic mean of the F1 score for each class.

This is preferred over accuracy because the orbit classes may be imbalanced.

A model that performs extremely well on LEO but poorly on GEO/HEO could still have high overall accuracy if LEO dominates the data.

Macro-F1 gives each class equal influence on the main comparison.

## Additional Metrics

The experiment also reports:

* Accuracy
* Balanced Accuracy
* Macro Precision
* Macro Recall
* Training Time
* Saved Pipeline Size

Per-class precision, recall and F1 are also examined through classification reports and confusion matrices.

---

# Results

> **Populate this section automatically from the latest `model_comparison.csv` generated by the notebook. Do not manually enter predicted or outdated values.**

## Final Model Comparison

| Model                | Validation Macro-F1 |  Test Macro-F1 |  Test Accuracy | Test Balanced Accuracy |  Training Time |     Model Size |
| -------------------- | ------------------: | -------------: | -------------: | ---------------------: | -------------: | -------------: |
| Dummy Baseline       |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| Logistic Regression  |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| Random Forest        |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| HistGradientBoosting |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| Extra Trees          |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| RBF SVM              |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |
| KNN                  |      `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |         `[RUN RESULT]` | `[RUN RESULT]` | `[RUN RESULT]` |

### Best Model

**Model:** `[RUN RESULT]`

**Validation Macro-F1:** `[RUN RESULT]`

**Test Macro-F1:** `[RUN RESULT]`

**Training time:** `[RUN RESULT] seconds`

**Model size:** `[RUN RESULT] KB`

---

# Why the Best Model Won

The explanation should focus on the **properties of the data**, not on generic claims that one algorithm is more powerful.

The target-defining orbital variables:

```text
MEAN_MOTION
ECCENTRICITY
INCLINATION
```

are deliberately withheld.

Therefore the models must infer orbit class from:

```text
RA_OF_ASC_NODE
ARG_OF_PERICENTER
MEAN_ANOMALY
BSTAR
MEAN_MOTION_DOT
MEAN_MOTION_DDOT
```

This produces a more difficult classification problem involving weaker and potentially overlapping orbital signatures.

The project therefore evaluates feature distributions and permutation importance to determine whether the winning model is benefiting from identifiable structure in the data.

The final explanation should reference the actual experiment output:

> **Best model:** `[MODEL]`

> **Top predictive features:** `[FEATURE 1]`, `[FEATURE 2]`, `[FEATURE 3]`

> **Observed data property:** `[DATA-BASED OBSERVATION FROM FEATURE DISTRIBUTIONS / PERMUTATION IMPORTANCE]`

The explanation should not simply say:

> "`[MODEL]` won because it is more powerful."

Instead, it should connect the result to measurable properties of the satellite dataset, such as nonlinear class overlap, differing feature distributions, or localized structure.

---

# Ablation Study

## Hypothesis

`BSTAR` contains information associated with orbital regime because it represents a drag-related parameter in the GP/SGP4 formulation.

Lower-altitude satellite populations are more strongly influenced by atmospheric drag than high-altitude populations, making BSTAR a plausible candidate for additional predictive information.

## Experiment

Two feature sets are compared:

### Full model

```text
RA_OF_ASC_NODE
ARG_OF_PERICENTER
MEAN_ANOMALY
BSTAR
MEAN_MOTION_DOT
MEAN_MOTION_DDOT
```

### Ablated model

```text
RA_OF_ASC_NODE
ARG_OF_PERICENTER
MEAN_ANOMALY
MEAN_MOTION_DOT
MEAN_MOTION_DDOT
```

The exact same:

* train/validation/test split;
* preprocessing;
* random seeds;
* seven algorithms;
* evaluation metric

are used for both experiments.

## Ablation Results

| Experiment    | Best Validation Macro-F1 | Best Model     |
| ------------- | -----------------------: | -------------- |
| Full features |           `[RUN RESULT]` | `[RUN RESULT]` |
| Without BSTAR |           `[RUN RESULT]` | `[RUN RESULT]` |

### Ranking change

`[STATE WHETHER THE MODEL RANKING CHANGED]`

### Interpretation

If the performance decreases substantially after removing BSTAR, this supports the hypothesis that BSTAR contains useful information related to orbital regime.

If the ranking changes, that is particularly informative because it indicates that the removed feature affected the relative usefulness of the model representations.

The result should be interpreted from the observed data rather than assumed beforehand.

---

# Reproducibility

The experiment is designed to be reproducible despite using a live satellite catalog.

Each run records:

* run date;
* unique run identifier;
* CelesTrak source;
* exact daily dataset filename;
* dataset SHA-256 hash;
* number of records;
* latest GP epoch represented in the dataset;
* feature set;
* excluded variables;
* split proportions;
* random seed;
* model list;
* orbit definitions;
* ESA-to-four-class mapping;
* ablation definition.

The exact daily CelesTrak snapshot used for a run is archived in GitHub.

This allows a future reader to reproduce the analysis even if CelesTrak's current Active Satellites population has changed.

---

# Google Drive and GitHub Workflow

The project uses two types of persistent storage for different purposes.

## Google Drive

Google Drive acts as the working cache.

Only the latest daily raw CelesTrak snapshot is retained:

```text
CelesTrak_Active_Satellites/
└── active_satellites_YYYY-MM-DD.csv
```

If today's file already exists, the notebook does not contact CelesTrak.

If today's file does not exist:

1. the latest Active Satellites CSV is downloaded;
2. the file is validated;
3. today's dated snapshot is saved;
4. older daily raw snapshots in the cache are deleted.

## GitHub

GitHub acts as the permanent public archive.

Historical daily datasets are retained:

```text
datasets/
├── active_satellites_2026-10-01.csv
├── active_satellites_2026-10-02.csv
└── active_satellites_2026-10-04.csv
```

Each experiment run is also archived:

```text
runs/
└── YYYY-MM-DD/
    └── run_TIMESTAMP/
```

The GitHub repository therefore preserves both:

1. the exact dataset snapshot;
2. the corresponding experiment outputs.

---

# Repository Structure

The repository is organized approximately as follows:

```text
satellite-orbit-classification/
│
├── Satellite_Orbit_Classification.ipynb
│
├── README.md
│
├── datasets/
│   ├── active_satellites_YYYY-MM-DD.csv
│   └── ...
│
└── runs/
    ├── YYYY-MM-DD/
    │   └── run_TIMESTAMP/
    │       ├── models/
    │       │   ├── dummy_baseline.joblib
    │       │   ├── logistic_regression.joblib
    │       │   ├── random_forest.joblib
    │       │   ├── histgradientboosting.joblib
    │       │   ├── extra_trees.joblib
    │       │   ├── rbf_svm.joblib
    │       │   └── k_nearest_neighbors.joblib
    │       │
    │       ├── results/
    │       │   ├── model_comparison.csv
    │       │   ├── ablation_results.csv
    │       │   └── experiment_metadata.json
    │       │
    │       ├── figures/
    │       │
    │       ├── active_satellites_with_orbit_class.csv
    │       │
    │       └── experiment_bundle.zip
    │
    └── ...
```

---

# Running the Project

## Recommended Environment

The experiment is designed to run directly in **Google Colab**.

No local Python installation or Anaconda environment is required.

The notebook uses:

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* joblib
* requests

The notebook downloads the current/day-specific CelesTrak Active Satellites dataset automatically when required.

---

## GitHub Authentication

GitHub uploads are authenticated using a GitHub fine-grained personal access token stored as a **Google Colab Secret**.

The secret name used by the notebook is:

```text
GITHUB_TOKEN
```

The token should be restricted to the project repository and granted only the repository permissions required for writing repository contents.

The token itself must never be committed to this repository.

---

## Basic Run Sequence

Open the notebook and run the cells sequentially.

The high-level workflow is:

```text
Google Drive check
        ↓
Daily CelesTrak snapshot
        ↓
Data quality analysis
        ↓
Orbital mechanics
        ↓
ESA detailed regime
        ↓
Four-class target
        ↓
Feature selection
        ↓
Train / validation / test split
        ↓
Seven ML models
        ↓
Model comparison
        ↓
Feature analysis
        ↓
Ablation
        ↓
Run metadata
        ↓
GitHub archive
```

---

# Limitations

## 1. The four-class target is an aggregation

ESA provides a much more detailed orbital taxonomy than the four categories required by this assignment.

Therefore:

```text
ORBIT_CLASS
```

is a deliberately coarse assignment-level label.

The original detailed regime is preserved as:

```text
ESA_ORBIT_REGIME
```

to maintain scientific transparency.

## 2. GP elements are modeled orbital elements

The dataset consists of GP orbital-element representations rather than direct instantaneous Cartesian positions.

The analysis therefore classifies the orbital regime represented by the GP elements at the supplied epoch.

## 3. Live data change

The Active Satellites catalog changes over time.

Consequently, model results can change between daily snapshots even when the code remains unchanged.

For this reason, each run records its exact input snapshot and SHA-256 hash.

## 4. Target construction determines the scientific question

Because orbit classes are physics-derived rather than provided as a native dataset label, the exact definition of the target is a methodological choice.

This project minimizes ambiguity by documenting the ESA basis and the four-class aggregation explicitly.

## 5. Leakage-controlled feature set

The strict experiment intentionally excludes the variables directly used to construct the target.

This makes the ML problem harder and means performance should not be interpreted as the maximum possible accuracy achievable from all GP parameters.

---

# Scientific and Methodological Notes

## Why exclude identifiers?

Variables such as:

```text
OBJECT_NAME
NORAD_CAT_ID
OBJECT_ID
```

may encode satellite identity rather than general orbital structure.

Using them could allow a model to learn object-specific patterns rather than orbital relationships.

The purpose of this project is to study orbital-feature classification, so identifiers are excluded.

## Why use the same preprocessing?

The assignment requires a controlled comparison between algorithms.

All models therefore receive the same imputation and standardization pipeline.

## Why use Macro-F1?

The four orbit classes are not necessarily equally represented.

Macro-F1 prevents a dominant class from completely determining the headline performance.

## Why keep the test set separate?

The validation set is used to select the best model.

The test set is held out until final evaluation.

This prevents the test score from becoming part of the model-selection process.

---

# References

## CelesTrak

**CelesTrak — NORAD GP Element Sets**

Use this link for the main dataset/source documentation.

**Active Satellites**

Use this link for the actual Active Satellites GP dataset/table.

**CelesTrak GP Data Format Documentation**

Use this link for the explanation of the CSV/JSON GP formats and query system.

## European Space Agency

**ESA Annual Space Environment Report — Table 1.2**

This is the scientific source for the detailed orbital-regime definitions used to construct the target.

The ESA table defines orbital regimes using semi-major axis, eccentricity, inclination, perigee height and apogee height.

## Scikit-learn

The machine-learning models are implemented using scikit-learn.

The project uses scikit-learn implementations of:

* DummyClassifier
* LogisticRegression
* RandomForestClassifier
* HistGradientBoostingClassifier
* ExtraTreesClassifier
* SVC
* KNeighborsClassifier

---

# Project Outputs

The primary outputs generated by each run are:

### Dataset

```text
active_satellites_YYYY-MM-DD.csv
```

### Processed dataset

```text
active_satellites_with_orbit_class.csv
```

### Model comparison

```text
model_comparison.csv
```

### Ablation results

```text
ablation_results.csv
```

### Experiment metadata

```text
experiment_metadata.json
```

### Trained models

```text
*.joblib
```

### Experiment bundle

```text
experiment_bundle.zip
```

---

# Conclusion

This project develops a reproducible four-class satellite orbital classification experiment using real, continuously updated CelesTrak Active Satellites GP data.

The core methodology is:

```text
CelesTrak Active Satellites
            ↓
      Data Cleaning
            ↓
     Orbital Mechanics
            ↓
    ESA Detailed Taxonomy
            ↓
   Four-Class Aggregation
            ↓
Leakage-Controlled Features
            ↓
  70/15/15 Stratified Split
            ↓
       Seven Models
            ↓
    Macro-F1 Comparison
            ↓
     Data-Based Analysis
            ↓
       BSTAR Ablation
```

The experiment is designed not only to identify the best-performing classifier, but also to determine **what properties of the orbital data make classification easier or harder**, while preserving a complete record of the dataset snapshot and experiment configuration used for each run.
