# Week 3: Advanced Data Analysis & Visualization in Logistics

Exploratory data analysis (EDA) and visualization of a simulated logistics dataset — examining delivery time, transportation cost, and shipment volume to surface operational bottlenecks and cost drivers, built with `pandas`, `numpy`, `matplotlib`, and `seaborn`.

## Overview

This project simulates a 2,000-shipment logistics dataset spanning four transport modes (Road, Air, Rail, Sea) and five regions, then applies statistical EDA and seven purpose-chosen visualizations to answer concrete operational questions: Where are delays concentrated? What drives transportation cost? Which regions and modes underperform on SLA?

## Contents

| File | Description |
|---|---|
| `Week3_Logistics_Data_Analysis_Visualization.docx` | Full report: methodology, EDA results, all visualizations with interpretation, and recommendations |
| `logistics_eda_visualization.py` | Standalone script — simulates the dataset, runs EDA, and generates all charts |
| `logistics_dataset_simulated.csv` | The simulated dataset used for analysis |
| `charts/` | All exported visualization PNGs |

## Analysis Highlights

- **Bimodal delivery-time distribution** — a fast cluster (Road/Rail) and a slow cluster (Sea + delay-affected shipments), invisible in a single average
- **Speed-cost trade-off** — Air is fastest but ~5-6x more expensive than Road/Rail
- **Correlation analysis** — cost correlates moderately with both delivery time (-0.34) and volume (+0.44); volume has negligible effect on delivery speed
- **Regional gap** — the East region trails the network's 89.8% on-time average
- **Bottleneck candidates** — Road and Rail show the widest delivery-time variability and most outliers

## Visualizations

1. Delivery time distribution (histogram + KDE)
2. Delivery time & cost by transport mode (grouped bar)
3. On-time delivery rate by region (bar + reference line)
4. Correlation heatmap
5. Cost vs. shipment volume (scatter, colored by mode)
6. Delivery time spread & outliers by mode (boxplot)
7. Shipment volume share by product category (pie)

## Usage

```bash
pip install pandas numpy matplotlib seaborn
python logistics_eda_visualization.py
```

Outputs a simulated dataset CSV, summary statistic CSVs, and all chart PNGs to `charts/`.

## Tools

`pandas` · `numpy` · `matplotlib` · `seaborn`

## Author

Anam Khan
