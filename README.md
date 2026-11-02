# Week 4 - Predictive Modeling and Optimization in Logistics

Predicts delivery time and late-delivery risk on the Week 3 dataset, then optimizes carrier allocation (MILP) and delivery-date promises.

- `Predictive_Modeling_Optimization_Report.docx` - full report
- `w4_model.py` - preparation, models, tuning, validation, optimization, figures (seed 42)
- `logistics_clean.csv` - input data (Week 3 cleaned dataset)
- `carrier_recommendations.csv` - recommended carrier per shipment
- `results4.json` - all computed metrics
- `figures/` - charts used in the report

Run: `pip install pandas numpy matplotlib seaborn scipy scikit-learn` then `mkdir figs4 && python w4_model.py`
