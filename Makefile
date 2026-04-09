PYTHON=python
RAW_DATA?=data/raw/Telco-Customer-Churn.csv

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m src.pipeline --input $(RAW_DATA) --output-dir reports

dashboard:
	streamlit run dashboard/app.py
