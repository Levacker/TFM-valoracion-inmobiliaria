# Valoración Inmobiliaria Masiva: SLX vs Machine Learning

Trabajo Fin de Máster — Comparativa entre modelos econométricos espaciales (SLX) y modelos de Machine Learning (Random Forest, XGBoost) para la valoración masiva de inmuebles residenciales en el condado de King (Washington, EE.UU.).

---

## Descripción del proyecto

Este repositorio contiene el **Módulo 3 (Machine Learning)** del TFM. El objetivo es construir modelos predictivos del precio de la vivienda utilizando el dataset King County House Prices (21.613 observaciones, 2014–2015) e incorporar **features de autocorrelación espacial mediante lags KNN** para capturar la dimensión geográfica del precio.

Los modelos desarrollados se comparan con el modelo SLX (Spatial Lag of X) del Módulo 2 en términos de RMSE, MAE, R² e Índice de Moran de los residuos.

---

## Estructura del repositorio

```
tfm-valoracion-inmobiliaria/
├── data/
│   ├── raw/                    # Dataset original (kc_house_data.csv)
│   ├── spatial/                # Shapefiles King County
│   └── processed/              # Generado al ejecutar los notebooks
├── notebooks/
│   ├── 01_EDA.ipynb            # Análisis exploratorio
│   ├── 02_preprocessing.ipynb  # Preprocesamiento y train/test split
│   ├── 03_spatial_features.ipynb  # Feature engineering espacial (KNN lags)
│   ├── 04_random_forest.ipynb  # Modelo Random Forest
│   ├── 05_xgboost.ipynb        # Modelo XGBoost
│   ├── 06_shap_explicabilidad.ipynb  # Análisis SHAP
│   └── 07_comparativa_final.ipynb   # Comparativa SLX vs ML
├── src/
│   └── utils.py                # Funciones reutilizables
├── outputs/
│   ├── models/                 # Modelos serializados (.joblib)
│   ├── figures/                # Gráficos exportados
│   └── metrics/                # Métricas en JSON
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd tfm-valoracion-inmobiliaria

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Datos

| Archivo | Descripción |
|---|---|
| `data/raw/kc_house_data.csv` | Dataset King County House Prices (21.613 viviendas) |
| `data/spatial/King_county_zip.*` | Shapefile de códigos postales del condado de King |
| `data/spatial/kc_house.*` | Shapefile de puntos de viviendas |

Los archivos en `data/processed/` son generados automáticamente al ejecutar los notebooks y no se versionan en git.

---

## Orden de ejecución

Los notebooks deben ejecutarse en orden secuencial, ya que cada uno genera archivos que el siguiente necesita:

| Orden | Notebook | Genera |
|---|---|---|
| 1 | `01_EDA.ipynb` | Figuras exploratorias |
| 2 | `02_preprocessing.ipynb` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, coordenadas |
| 3 | `03_spatial_features.ipynb` | `X_train_spatial.csv`, `X_test_spatial.csv` |
| 4 | `04_random_forest.ipynb` | `rf_model.joblib`, `rf_metrics.json`, residuos RF |
| 5 | `05_xgboost.ipynb` | `xgb_model.joblib`, `xgb_metrics.json`, residuos XGB |
| 6 | `06_shap_explicabilidad.ipynb` | Figuras SHAP |
| 7 | `07_comparativa_final.ipynb` | Tabla comparativa RF / XGBoost / SLX |

Para ejecutar un notebook de forma no interactiva:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb \
    --output notebooks/01_EDA_ejecutado.ipynb
```

---

## Nota sobre el modelo SLX (Módulo 2)

El notebook `07_comparativa_final.ipynb` incluye un **placeholder** para las métricas del modelo SLX desarrollado en el Módulo 2. Para completar la comparativa se necesita un archivo `data/slx_metrics.json` con la siguiente estructura:

```json
{
  "modelo": "SLX",
  "RMSE": null,
  "MAE": null,
  "R2": null,
  "moran_i_residuos": null
}
```

---

## Reproducibilidad

Todos los modelos utilizan `random_state=42`. La variable objetivo es `log(price)` (logaritmo natural). Las métricas se reportan tanto en escala logarítmica como en escala original (dólares) para facilitar la interpretación.
