# 🌍 Climate Risk Intelligence Engine  
### A Streamlit-Based Climate Forecasting, Economic Risk, and Impact Modeling Platform

The **Climate Risk Intelligence Engine** is an interactive, data-driven analytics dashboard that models global climate trajectories, economic impacts, and human-centered risks using real-world datasets and scientific methodologies.

Built with **Python**, **Streamlit**, **Facebook Prophet**, **Monte Carlo uncertainty modeling**, and **IPCC-aligned climate physics**, this tool enables policymakers, researchers, and analysts to explore how emissions trends translate into future warming, GDP loss, sea-level rise, mortality risks, and population displacement.

---

## 🚀 Key Features

### ✔ Real-Time Climate Forecasting
- Uses **Facebook Prophet** to model CO₂ emissions, greenhouse gases, and energy trends  
- Produces forward-looking projections up to **100 years**

### ✔ Scenario Analysis
- Adjustable **annual emission reduction targets**
- Compare **two countries side-by-side**

### ✔ Uncertainty Quantification
- Monte Carlo simulation (500 iterations)
- Generates **10th–90th percentile uncertainty bands**

### ✔ IPCC Temperature Modeling
- Converts cumulative emissions → atmospheric CO₂ ppm → predicted warming
- Based on **IPCC AR6 WGI climate sensitivity** (≈3°C per doubling of CO₂)

### ✔ Economic Damage Modeling
- Estimates GDP loss using the **DICE Integrated Assessment Model** (Nordhaus 2017)
- Visualizes **climate-adjusted vs. baseline GDP** over time

### ✔ Human Impact Estimation
- Climate-related mortality (Lancet 2020–based model)
- Displacement risk from sea-level rise (PNAS 2017 scaling)

### ✔ Geospatial Risk Visualization
- Choropleth map showing **country-level coastal flood risk indices**

### ✔ Professional Report Export
- Generates a downloadable **HTML climate intelligence report**
- Includes charts, metrics, maps, and full methodology documentation

---

## 🧩 How It Works — System Architecture

### 1️⃣ Data Ingestion

The application loads two primary data sources:

- **OWID CO₂ Dataset**  
  https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv

- **World Bank GDP API**  
  Used to retrieve economic baselines (`NY.GDP.MKTP.CD`)

All data is cleaned, filtered, and normalized prior to modeling.

---

### 2️⃣ Country-Level Time Series Preparation

```python
df_country = df[df["country"] == country][["year", column]]
df_country.rename(columns={"year": "ds", column: "y"})
```

### 3️⃣ Emissions Forecasting with Prophet
```python
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=years, freq="Y")
forecast = model.predict(future)
```

Optionally applies exponential emission reduction scenarios.

### 4️⃣ Monte Carlo Uncertainty Simulation
```python
noise = np.random.normal(0, np.std(base) * 0.05, len(base))
```

500 simulations introduce stochastic variability, producing uncertainty ranges.

### 5️⃣ Temperature Modeling

Follows IPCC AR6 radiative forcing physics:
```
CO₂ ppm = preindustrial_ppm + cumulative_gt_co2 / 2.12  
ΔT = climate_sensitivity × log2(CO₂_ppm / preindustrial_ppm)
```

Where climate sensitivity ≈ 3°C.

### 6️⃣ GDP Damage Modeling (DICE)
```
Damage % = 0.01 × (ΔT²)  
Damage $ = GDP × Damage %
```

GDP baseline is retrieved from the World Bank API.

### 7️⃣ Sea Level & Human Impact Models

- Sea-level rise = cumulative emissions × sensitivity factor

- Mortality and displacement scaled from peer-reviewed research

### 8️⃣ Dashboard Visualization

### Built using:

- Plotly (interactive charts and maps)

- Streamlit (UI, controls, layout)

- Dynamic metrics for:

- Temperature

- GDP

- Sea level

- Mortality

- Displacement

### 9️⃣ Professional Report Generator

- Exports a fully formatted HTML report including:

- Executive summary

- Forecast charts

- Economic projections

- Flood risk map

- Academic references

- Full methodology

## 📦 Installation
### 1. Clone the Repository
```
git clone https://github.com/YOUR_USERNAME/climate-risk-intelligence-engine.git
cd climate-risk-intelligence-engine
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
Required libraries include:

- streamlit

- prophet

- pandas

- numpy

- requests

- pycountry

- plotly

### ▶️ Running the Dashboard

Start the Streamlit app:
```
streamlit run app.py
```
The dashboard will open at:
```
http://localhost:8501
```
## 🎛 User Controls

The sidebar provides adjustable parameters:

| Setting	| Description |
| :--- | :--- |
| Primary Country |	Country for base analysis |
| Comparison Country |	Optional second country |
| Indicator	| CO₂, GHG, energy, per capita emissions |
| Forecast Years |	10–100 year horizon |
| Annual Reduction (%)	| Emission reduction policy scenario |

## 📊 Outputs & Analytics Provided
### Climate Metrics

- Emissions projections

- Temperature increase

- Sea-level rise

### Economic Metrics

- GDP damage percentage

- Dollar-value economic loss

- Climate-adjusted vs. baseline GDP

### Human Metrics

- Climate-related mortality

- Climate displacement

### Geospatial Metrics

- Country-level flood risk index map

## 🔬 Scientific Methodologies Used
Component	Method / Citation
Emissions Forecasting	Facebook Prophet
Temperature Modeling	IPCC AR6 WGI climate sensitivity
Economic Damage	Nordhaus DICE IAM (NBER 2017)
Displacement	Hauer 2017 – PNAS
Mortality	Watts et al. 2020 – Lancet
Sea-Level Rise	IPCC AR6 Table 9.7
Uncertainty Modeling	Monte Carlo (500 runs)
## 📄 Report Exporting

Click the “Export Professional Climate Intelligence Report” button to download a styled, multi-section HTML report containing:

- Executive summary

- Scientific projections

- Interactive charts

- Economic modeling

- Risk maps

- Full methodology

- Citations

## 📚 Project Structure
```
├── app.py               # Main Streamlit application
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

## 🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss the proposed update.
