# Spatial-omics Data Analysis in Cancer and Core Sampling Strategies for TMA

## Spatial Analysis & Visualization Toolkit

This repository contains Python tools for quantitative spatial analysis of tumor tissues using data from **t-CycIF** (tissue cyclic immunofluorescence). The toolkit enables comprehensive spatial characterization of tumor microenvironments and supports core sampling strategies for **Tissue Microarrays (TMA)**.

---

## Key Features

- **Spatial diversity metrics**: Shannon Index, Simpson Index  
- **Spatial distribution analysis**: Ripley's K Function  
- **Phenotype characterization**: Phenotypic Proportions, Spatial Co-occurrence  
- **High-performance computing**: Parallel processing with `Numba` and `multiprocessing`  
- **Visualization**: TIFF spatial maps, heatmaps, and interactive plots  
- **Memory optimization**: Chunked processing for large datasets  

---

## Repository Structure

### Core Analysis Scripts

| File                     | Description                                             |
|--------------------------|---------------------------------------------------------|
| `ROI_sampling.py`        | Main script for spatial metrics calculation (parallelized) |
| `shannon_tif.py`         | Generates TIFF maps of Shannon Index distribution       |
| `simpson_tif.py`         | Generates TIFF maps of Simpson Index distribution       |
| `ripley_k_tif.py`        | Generates TIFF maps of Ripley's K distribution          |


### Visualization Scripts

| File                          | Description                                         |
|-------------------------------|-----------------------------------------------------|
| `ripley_k_index_heatmap.py`   | Creates smooth heatmaps of Ripley's K values        |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/spatial-omics-analysis.git
cd spatial-omics-analysis
```

Dependencies:
- Python 3.9+
- numpy
- pandas
- tifffile
- scipy
- shapely
- scikit-image
- matplotlib
- geopandas
- numba
- seaborn


# Usage

## 1. Compute Spatial Metrics

```bash
python ROI_sampling.py
```

Sigue las instrucciones para seleccionar una métrica:
- Shannon Index
- Ripley K Function
- Phenotypic Proportion
- Spatial Co-occurrence
- Simpson Index

  
## 2. Generate Spatial Maps 

# Shannon Index
python shannon_tif.py

# Simpson Index
python simpson_tif.py

# Ripley's K
python ripley_k_tif.py


Key Configuration Parameters in the Scripts:

| Parameter       | Description                          | Default Value       |
|-----------------|--------------------------------------|---------------------|
| RADIUS_STEP     | Radial step size                     | 50 μm               |
| NUM_POINTS      | Number of sampling points            | 100,000             |
| STEP_SIZE       | Radial increment                     | 10 μm               |
| MAX_STEPS       | Maximum number of radial steps       | 100                 |
| X_RES, Y_RES    | Spatial resolution                   | 1 μm/pixel          |


