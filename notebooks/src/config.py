from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

DADOS_ORIGINAIS = PASTA_DADOS / "stroke_risk_dataset.csv"
DADOS_LIMPOS = PASTA_DADOS / "stroke_risk_clean.parquet"

PASTA_MODELOS = PASTA_PROJETO / "modelos"
MODELO_FINAL_CLF = PASTA_MODELOS / "stroke_risk_clf_model.joblib"
MODELO_FINAL_REG = PASTA_MODELOS / "stroke_risk_reg_model.joblib"
MODELO_FINAL_REG_DL = PASTA_MODELOS / "stroke_risk_reg_dl_model.joblib"
