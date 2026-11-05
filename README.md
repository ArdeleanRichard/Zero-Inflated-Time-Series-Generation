# Zero Inflated Time Series Generation
 
This repository contains the code and configurations for a study submitted as a scientific research article including two new proposed approach for zero inflated time series generation.


[![DOI](https://img.shields.io/badge/DOI-10.3390/make8110365-blue)](https://doi.org/10.3390/make8110365) 

This repository contains the code and configurations used in the study titled:
**"Zero-Inflated Time Series Generation"**
by Eugen-Richard Ardelean, and Raluca Portase, published in *MAKE*, 2026.


## 📄 Overview

Zero-Inflated Time Series (ZITS), characterized by excessive zeros relative to standard distributions, pose fundamental challenges for synthetic data generation due to the need to simultaneously preserve the temporal dependencies, the excess zero structure, and the heavy-tailed non-zero distributions. Current generative models, including variational autoencoders (VAE) and generative adversarial networks (GAN), lack the mechanisms for handling zero-inflated data and fail to generate data with similar characteristics. 

This work proposes two architectures, ZITS-GAN and ZITS-VAE, designed for zero-inflated time series generation through a two-head decoder structure that separately models the zero pattern and the non-zero magnitude. The proposed architectures use dilated convolutional layers to capture long sequence dependencies and stabilize training through $\beta$-weighted ELBO for the ZITS-VAE and Wasserstein GAN with gradient penalty for ZITS-GAN. The proposed models were evaluated on two real world time series datasets with zero ratios of 59\% and 74\% and have demonstrated superior performance compared to state-of-the-art models including 

* **TimeGAN**,
* **TimeVAE**,
* **ChronoGAN**,
* **TransFusion**,
* **FIDE**.


---

## 📊 Datasets

### 1. M5 forecasting Dataset

* **Name**: M5
* **Description**: provided on Kaggle, which contains daily unit sales collected by Walmart and distributed as several CSV files. The raw release comprises a wide table of bottom-level series, each row corresponds to a single stock keeping unit (SKU) at a single store, together with a separate calendar mapping that associates each recorded day index with its calendar date and auxiliary fields such as weekday, month, and special events. This publicly available data comprises 30,490 items over 1,941 days. The M5 data explicitly focuses on realistic retail demand and displays intermittency (lots of zero daily sales).
* **Access**: Available at 
  🔗 [M5 Dataset Page](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)

### 2. Household IoT Devices Dataset

* **Name**: Household IoT Devices
* **Description**: consists of time series data capturing the daily operational duration of household IoT devices characterized by discrete operating cycles over one year. To ensure statistical relevance and data quality, we retained from the raw dataset - comprising individual running cycle durations in seconds per device - only those devices exhibiting a sufficient number of running cycles. 
* **Access**: unavailable. 

### 3. Car Parts Dataset

* **Name**: Car Parts
* **Description**: 2,674 intermittent monthly car-parts sales series, covering January 1998 through March 2002 (51 observations per series). Missing values in the original series are replaced with zeros in this release.
* **Access**: [Zenodo Car Parts Dataset](https://zenodo.org/records/4656021) (CC BY 4.0).
* **Download and preprocessing**: run `python download_data_carparts.py`. This saves the source archive and TSF file, a CSV with month columns, and arrays under `data/carparts/`. The model-ready arrays are `carparts_X_51.npy` with shape `(2674, 51)` and `carparts_X_51.npz` with key `data` and shape `(2674, 51, 1)`.

To evaluate the saved Car Parts model runs, run `python main_metrics_carparts.py` from the repository root. The script writes a checkpointed `results_cp/carparts_metrics.csv`; completed runs are skipped on restart, while failed runs are reported and retried.

---

## 🧠 Models Evaluated

* TimeGAN
* TimeVAE
* ChronoGAN
* TransFusion
* FIDE
* Proposed models: ZITS-GAN and ZITS-VAE


## 📈 Results Summary

We show here a summary of the evaluations made showing the difference in 0 ratios (Δ 0 ratio), the Long Sequence Predictive Score (LPS) and the Long Sequence Discriminative Score (LDS) obtained.

| Model     | Dataset         			| Δ 0 ratio | LPS        | LDS        |
| --------- | ------------------------- | --------- | ---------  | ---------  |
| TimeVAE   | M5    					| 0.144     | **1.344**  | 0.389      |
| ZITS-GAN  | M5    					| **0.012** | 1.448      | 0.213      |
| ZITS-VAE	 | M5    					| 0.014     | 1.408      | **0.126**  |
| TimeVAE   | Household IoT Devices     | 0.259     | 2847       | 0.5        |
| ZITS-GAN  | Household IoT Devices     | 0.038     | **2153**   | **0**      |
| ZITS-VAE  | Household IoT Devices     | **0.033** | **2153**   | 0.248      |

---

## 📜 Citation

If you use this code or reference the models/datasets in your work, please cite:

```bibtex
@article{10.3390/make8110365,
	title = {Zero-{Inflated} {Time} {Series} {Generation}},
	volume = {8},
	copyright = {All rights reserved},
	issn = {2504-4990},
	url = {https://www.mdpi.com/2504-4990/8/11/365},
	doi = {10.3390/make8110365},
	language = {en},
	number = {11},
	urldate = {2026-11-05},
	journal = {Machine Learning and Knowledge Extraction},
	publisher = {Multidisciplinary Digital Publishing Institute},
	author = {Ardelean, Eugen-Richard and Portase, Raluca},
	month = nov,
	year = {2026},
	keywords = {GAN, synthetic data, time series generation, VAE, zero-inflated data, zero-inflated time series},
	pages = {365},
	file = {Full Text PDF:C\:\\Users\\Richard\\Zotero\\storage\\ABQ4KX3S\\Ardelean and Portase - 2026 - Zero-Inflated Time Series Generation.pdf:application/pdf},
}
```

---

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---
