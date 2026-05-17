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

Recomendado: usar **Conda con Python 3.11** para evitar problemas de compilación en macOS ARM (por ejemplo con `matplotlib` en Python 3.13).

### Opción recomendada (Conda)

```bash
# Clonar el repositorio
git clone https://github.com/Levacker/TFM-valoracion-inmobiliaria.git
cd tfm-valoracion-inmobiliaria

# Crear entorno con Python 3.11
conda create -n tfm311 python=3.11 -y
conda activate tfm311

# Verificar versión de Python (debe ser 3.11.x)
python --version

# Instalar dependencias del proyecto
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Opción alternativa (venv)

Si prefieres `venv`, asegúrate igualmente de usar Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Solución rápida de errores comunes

- Si ves errores al instalar `matplotlib` o `shap` con mensajes de compilación (`meson`, `metadata-generation-failed`, etc.), normalmente estás en Python 3.13 o en un entorno distinto al esperado.
- Comprueba:
  - `which python` apunta al entorno activo (`.../envs/tfm311/bin/python` en conda).
  - `python --version` muestra `3.11.x`.
- Si estabas en otro entorno, desactívalo (`deactivate`) y vuelve a activar conda (`conda activate tfm311`).

---

## Preparación del entorno (Adriana) para ejecutar notebooks

Pasos recomendados de principio a fin:

```bash
# 1) Clonar repo
git clone https://github.com/Levacker/TFM-valoracion-inmobiliaria.git
cd tfm-valoracion-inmobiliaria

# 2) Crear y activar entorno conda
conda create -n tfm311 python=3.11 -y
conda activate tfm311

# 3) Instalar dependencias
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 4) Instalar kernel de Jupyter para este entorno
python -m ipykernel install --user --name tfm311 --display-name "Python (tfm311)"

# 5) Abrir Jupyter
jupyter notebook
```

Dentro de Jupyter:
- Abrir el notebook que toque (por ejemplo `notebooks/01_EDA.ipynb`).
- Seleccionar kernel: `Python (tfm311)`.
- Ejecutar celdas en orden.

Comprobación rápida antes de ejecutar:

```bash
which python
python --version
```

Debe mostrar:
- Un path tipo `.../anaconda3/envs/tfm311/bin/python`.
- Versión `Python 3.11.x`.

Nota importante:
- Clonar el repo **no** copia el entorno. Cada persona debe crear su propio entorno local con los pasos anteriores.

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
