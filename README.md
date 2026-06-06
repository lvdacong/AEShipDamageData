# Supporting Data for Autoencoder-Based Gradual Damage Warning for Ship Structures

This repository contains supporting data and code for the manuscript:

**A Gradual Damage Suspicion Warning Method for Ship Structures Based on Autoencoder Health Manifold Modeling**

Submitted to *Ships and Offshore Structures*.

## Contents

### `data/`

Preprocessed strain-response matrices used in the main source-domain validation experiments.
Each `.npz` file contains one NumPy array named `V`, where rows are samples and columns are measurement channels.

| File | Shape | Description |
| --- | ---: | --- |
| `health_original_2000_preprocessed_data_raw.npz` | `(2000, 252)` | Healthy strain-response samples used for autoencoder training and validation. |
| `crack_first_damage_original_100_preprocessed_data_raw.npz` | `(100, 252)` | Crack damage test samples. |
| `corrosion_second_damage_12_original_100_preprocessed_data_raw.npz` | `(100, 252)` | Corrosion damage test samples. |
| `multi_damage_two_circle_original_100_preprocessed_data_raw.npz` | `(100, 252)` | Combined multi-damage test samples. |

### `metadata/`

Measurement-point and model-index mapping files used to interpret the 252-channel strain-response vectors.

### `model/`

The pretrained autoencoder model and training-loss record used for source-domain validation.

### `results/`

Baseline detection metrics and representative validation figures.

### `code/`

Python scripts used for preprocessing, autoencoder training, source-domain validation, and damage detection.

## Notes on the finite element model

The complete Abaqus finite element model files and full simulation input decks are not included in this public package. They contain detailed ship structural model information and unpublished institutional research materials. The shared matrices and metadata provide the data directly supporting the reported autoencoder validation and damage-detection analyses.

## Suggested data availability statement

The data supporting the findings of this study are openly available in the public GitHub repository "AEShipDamageData" at https://github.com/lvdacong/AEShipDamageData, release v1.0.0: https://github.com/lvdacong/AEShipDamageData/releases/tag/v1.0.0. The repository contains the preprocessed strain-response matrices, measurement metadata, baseline validation results, pretrained model, and analysis scripts used in the study. The complete Abaqus finite element model files and simulation input decks are not publicly shared because they contain institution-controlled ship structural model information and unpublished research materials.

## License

Unless otherwise stated, the data in this repository are provided for academic research and review purposes. Please cite the manuscript if you use the data or code.
