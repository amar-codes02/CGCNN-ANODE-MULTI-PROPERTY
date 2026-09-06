# Accelerated Multi-Property Screening Pipeline for Battery Anode Materials & 3D Graphene TPMS via CGCNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![PyMatGen](https://img.shields.io/badge/pymatgen-Materials%20Analysis-green.svg)](https://pymatgen.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework utilizing **Crystal Graph Convolutional Neural Networks (CGCNN)** for high-throughput, multi-property screening of next-generation battery anode materials and 3D Triply Periodic Minimal Surface (TPMS) graphene architectures.

---

## 📌 Overview

Accelerating the discovery of high-performance battery anode materials requires evaluating multiple electronic, thermodynamic, mechanical, and electrochemical properties simultaneously. This repository implements a unified **multi-task CGCNN** pipeline that maps 3D crystal structures (CIF/atomic graphs) directly to 7 critical properties:

1. **Band Gap ($E_g$)** [$\text{eV}$] – Electronic conductivity & metallicity classification
2. **Formation Energy ($E_f$)** [$\text{eV/atom}$] – Thermodynamic stability
3. **Energy Above Hull ($E_{\text{hull}}$)** [$\text{eV/atom}$] – Phase stability & synthesizability
4. **Bulk Modulus ($K$)** [$\text{GPa}$] – Volumetric deformation resistance
5. **Shear Modulus ($G$)** [$\text{GPa}$] – Resistance to shear stress
6. **Young's Modulus ($E$)** [$\text{GPa}$] – Mechanical stiffness & integrity
7. **Adsorption Energy ($E_{\text{ads}}$)** [$\text{eV}$] – Electrochemical binding affinity (Li/Na/ion interaction)

---

## 🔬 Model Performance & Metrics

The multi-task CGCNN model achieves state-of-the-art predictive accuracy across all evaluated target properties on test datasets:

| Target Property | Unit | Train MAE | Train $R^2$ | Val MAE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Band Gap** | $\text{eV}$ | 0.168 | 0.973 | 0.188 | 0.943 | 0.211 | 0.354 | **0.943** |
| **Formation Energy** | $\text{eV/atom}$ | 0.139 | 0.968 | 0.098 | 0.945 | 0.213 | 0.262 | **0.938** |
| **Energy Above Hull** | $\text{eV/atom}$ | 0.129 | 0.971 | 0.157 | 0.954 | 0.252 | 0.299 | **0.937** |
| **Bulk Modulus** | $\text{GPa}$ | 9.640 | 0.966 | 11.846 | 0.937 | 4.967 | 6.147 | **0.922** |
| **Shear Modulus** | $\text{GPa}$ | 7.816 | 0.969 | 5.206 | 0.942 | 1.924 | 2.495 | **0.925** |
| **Young's Modulus** | $\text{GPa}$ | 16.892 | 0.971 | 14.197 | 0.949 | 5.316 | 5.924 | **0.926** |
| **Adsorption Energy ($E_{\text{ads}}$)** | $\text{eV}$ | 0.146 | 0.966 | 0.237 | 0.940 | 0.189 | 0.236 | **0.932** |

---

## 🧱 3D Graphene TPMS Anode Screening

The framework screens 5 fundamental **Triply Periodic Minimal Surface (TPMS)** sheet topologies as advanced 3D porous graphene anode hosts:

| Rank | Topology | CIF Structure | Atoms | Density ($\text{g/cm}^3$) | $E_g$ ($\text{eV}$) | $E_f$ ($\text{eV/at}$) | $K$ ($\text{GPa}$) | $G$ ($\text{GPa}$) | $E$ ($\text{GPa}$) | $E_{\text{ads}}$ ($\text{eV}$) | Score |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Diamond (D)** | `graphene_sheet_diamond.cif` | 332 | 2.40 | 0.00 | 0.124 | 152.4 | 78.2 | 198.5 | -2.48 | **0.790** |
| **2** | **Gyroid (G)** | `graphene_sheet_gyroid.cif` | 244 | 1.93 | 0.00 | 0.098 | 126.8 | 64.5 | 163.7 | -2.55 | **0.778** |
| **3** | **IWP (I-WP)** | `graphene_sheet_iwp.cif` | 228 | 2.26 | 0.02 | 0.145 | 142.1 | 71.0 | 181.2 | -2.35 | **0.666** |
| **4** | **Neovius (N)** | `graphene_sheet_neovius.cif` | 188 | 2.05 | 0.01 | 0.162 | 131.5 | 66.8 | 170.1 | -2.28 | **0.577** |
| **5** | **Primitive (P)** | `graphene_sheet_primitive.cif` | 200 | 1.29 | 0.00 | 0.185 | 84.6 | 42.1 | 108.2 | -2.12 | **0.482** |

---

## 📂 Repository Structure

```text
├── data/
│   ├── Anode_Dataset_clean_enriched.csv    # Enriched anode dataset with target properties
│   ├── Anode_Dataset_clean_enriched.pkl    # Serialized enriched dataset
│   ├── df_anode_clean_cgcnn_predicted.csv  # Model predictions on clean dataset
│   ├── df_graphene_tpms_predicted.csv      # Screening results for 3D TPMS structures
│   ├── cgcnn_anode_evaluation_metrics.csv  # Train / Validation / Test benchmark metrics
│   ├── cgcnn_anode_model.pt                # Pre-trained multi-target CGCNN weights
│   ├── jarvis_properties_cache.json        # JARVIS DFT property cache
│   ├── mp_properties_cache.json            # Materials Project DFT property cache
│   └── 2dmatpedia_lookup.json              # 2DMatPedia structural reference cache
├── graphene_tpms/                          # CIF structures of 3D graphene TPMS architectures
│   ├── graphene_sheet_diamond.cif
│   ├── graphene_sheet_gyroid.cif
│   ├── graphene_sheet_iwp.cif
│   ├── graphene_sheet_neovius.cif
│   └── graphene_sheet_primitive.cif
├── models/
│   └── cgcnn_model.py                      # CGCNN architecture & feature featurizer
├── paper_figures/                          # High-resolution figures & publication tables
│   ├── fig1_eda_property_distributions.png
│   ├── fig2_eda_correlation_matrix.png
│   ├── fig3_eda_material_classification.png
│   ├── fig4_training_curves.png
│   ├── fig5_cgcnn_parity_plots.png
│   ├── fig7_user_dataset_top5_actual_vs_predicted.png
│   ├── fig8_user_dataset_radar_comparison.png
│   ├── fig9_tpms_property_rankings.png
│   ├── fig10_tpms_radar_comparison.png
│   └── table_*.png / table_*.pdf
├── EDA_dan_Training_CGCNN.ipynb            # Full EDA, Graph Featurization, Training & Eval
├── Olah data.ipynb                         # Raw data extraction and preprocessing pipeline
├── requirements.txt                        # Python package dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/amar-codes02/CGCNN-ANODE-MULTI-PROPERTY.git
cd CGCNN-ANODE-MULTI-PROPERTY
pip install -r requirements.txt
```

### 2. Inference on CIF Files

You can use the trained CGCNN model to predict multi-target properties for any custom crystal structure:

```python
from models.cgcnn_model import CrystalGraphConvNet, structure_to_graph
from pymatgen.core import Structure
import torch

# Load Structure
struct = Structure.from_file("graphene_tpms/graphene_sheet_diamond.cif")
graph_data = structure_to_graph(struct)

# Load Pre-trained Model
model = CrystalGraphConvNet(orig_atom_fea_len=92, n_conv=3, n_targets=4)
checkpoint = torch.load("data/cgcnn_anode_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict Properties
with torch.no_grad():
    preds = model(graph_data)
    print("Predicted Properties:", preds)
```

---

## 📊 Publication Figures

Key figures generated by the pipeline available in `paper_figures/`:
- **Training Convergence**: `paper_figures/fig4_training_curves.png`
- **CGCNN Parity Plots**: `paper_figures/fig5_cgcnn_parity_plots.png`
- **TPMS Radar Comparison**: `paper_figures/fig10_tpms_radar_comparison.png`
- **Property Correlations**: `paper_figures/fig2_eda_correlation_matrix.png`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
