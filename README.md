# BT4221 Group Project: HDB Resale Price Prediction

An end-to-end pipeline for predicting HDB resale flat prices in Singapore using PySpark and LLM-guided agents.

**Pipeline Overview:**

1. **Dataset Extraction** — Auxiliary amenity datasets (hawker centres, shopping malls, supermarkets, MRT/LRT stations, schools)
2. **HDB Data Loading & Geocoding** — Downloads resale price data from Data.gov.sg and geocodes addresses via OneMap
3. **Data Cleaning** — LLM-guided agent profiles the data and applies structured PySpark cleaning operations
4. **Exploratory Data Analysis** — Price trends, distributions, and correlations
5. **ML Pipeline** — LangGraph-orchestrated pipeline with feature engineering, model training, and evaluation agents

---

## Running on Google Colab (Recommended)

### Step 1: Copy the project to Google Drive

Clone or download this repository and upload the entire project folder to your Google Drive. The notebook expects the folder at:

```
My Drive/
└── bt4221-group-project/
    ├── bt4221_grp_project.ipynb
    └── dataset/
        ├── hawker_centre/
        ├── shopping_mall/
        ├── supermarket/
        ├── transport/
        ├── school/
        └── demographics/
```

> The `dataset/` subdirectories contain pre-extracted CSVs. If they are empty or missing, Section 1 of the notebook will regenerate them via API calls (requires internet access and may take several minutes).

---

### Step 2: Set up API keys in Colab Secrets

The notebook reads credentials from **Colab Secrets** (recommended) or a `.env` file.

**To add Colab Secrets:**

1. Open the notebook in Google Colab
2. Click the **key icon** in the left sidebar (Secrets)
3. Add each secret below with the toggle set to **enabled for this notebook**


| Secret Name       | Where to get it                                                                    |
| ----------------- | ---------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`  | [platform.openai.com](https://platform.openai.com/api-keys)                        |
| `GOV_DATA`        | [data.gov.sg](https://data.gov.sg) — create a free account and generate an API key |
| `ONEMAP_EMAIL`    | Email used to register at [onemap.gov.sg](https://www.onemap.gov.sg/apidocs/)      |
| `ONEMAP_PASSWORD` | Password for your OneMap account                                                   |


> **Alternative (local only):** Copy `.env.example` to `.env` and fill in your keys. Do **not** commit `.env` to version control.

---

### Step 3: Open the notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → Google Drive**
3. Navigate to `bt4221-group-project/bt4221_grp_project.ipynb` and open it

---

### Step 4: Run the notebook

Run cells sequentially from top to bottom (**Runtime → Run all**, or Shift+Enter cell by cell).

**Cell 4** mounts Google Drive and sets the working directory:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/bt4221-group-project')
```

> If your project folder is in a subfolder of Drive (e.g. `My Drive/NUS/bt4221-group-project`), update the `os.chdir(...)` path in Cell 4 before running.

---

### Section notes


| Section                    | Runtime                    | Notes                                                                                                                        |
| -------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **1 — Dataset Extraction** | 5–15 min                   | Skipped automatically if CSV files already exist in `dataset/`                                                               |
| **2 — HDB Data Loading**   | 2–5 min                    | Downloads ~1M rows from Data.gov.sg                                                                                          |
| **2e — Geocoding**         | 20–40 min (first run only) | Geocodes ~10k unique addresses via OneMap; results cached to `dataset/geocoded_addresses.csv` and skipped on subsequent runs |
| **3 — Data Cleaning**      | 5–10 min                   | Requires `OPENAI_API_KEY` for LLM-guided cleaning                                                                            |
| **4 — ML Pipeline**        | 5–10 min                   | Feature engineering stubs — see TODO items                                                                                   |


---

## Running Locally

**Prerequisites:**

- Python 3.10+
- Java 17+ (`brew install openjdk@17` on macOS)
- `JAVA_HOME` set to your JDK installation

**Install dependencies:**

```bash
pip install pyspark langgraph langchain langchain-openai openai python-dotenv beautifulsoup4 matplotlib seaborn
```

**Set up secrets:**

```bash
cp .env.example .env
# Fill in your API keys in .env
```

**Run the notebook:**

```bash
jupyter notebook bt4221_grp_project.ipynb
```

> On macOS, Cell 2 runs `apt-get install openjdk-17-jdk-headless` which will fail silently — this is expected. Java must be installed separately.

---

## Dataset Structure

After a full run, the `dataset/` directory will contain:

```
dataset/
├── hawker_centre/hawker_centres.csv        # ~100 rows
├── shopping_mall/shopping_malls.csv        # ~170 rows
├── supermarket/supermarkets.csv            # pre-extracted
├── transport/mrt_lrt_stations.csv          # ~200 rows
├── school/                                 # school locations
├── demographics/                           # demographic data
└── geocoded_addresses.csv                  # ~10k HDB address → lat/lng (generated by section 2e)
```

---

## Troubleshooting

**Geocoding cell hangs with no output**
The geocoding cell (Section 2e) uses `toPandas()` to extract unique addresses before making API calls. If it appears to hang, wait for the first print output — the Spark job may take a few minutes to collect results. Progress is printed every 100 addresses.

`**Missing secret 'X'` error**
A required API key is not set. Check that all four secrets are added in Colab Secrets with the notebook access toggle enabled.

`**DateTimeException: CANNOT_PARSE_TIMESTAMP`**
This was a known issue in earlier versions of the notebook caused by Spark 3.4+ columnar evaluation. It has been fixed in `_transaction_month_to_date` (Section 3b). Re-run from Cell 1 to pick up the fix.

**Drive path not found**
Update the `os.chdir(...)` path in Cell 4 to match where you placed the project folder in Google Drive.