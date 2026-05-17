"""
Funciones de utilidad compartidas por todos los notebooks del TFM.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    nulos = df.isnull().sum()
    if nulos.any():
        print(f"\nValores nulos:\n{nulos[nulos > 0]}")
    else:
        print("\nSin valores nulos.")
    return df


def compute_spatial_lags(df: pd.DataFrame, features: list, coords: np.ndarray,
                         k: int = 15, fit_coords: np.ndarray = None) -> pd.DataFrame:
    """
    Calcula spatial lags KNN para las features indicadas.

    Parámetros
    ----------
    df          : DataFrame con los datos (train o test)
    features    : lista de columnas a lagear
    coords      : coordenadas (lat, long) del conjunto actual (train o test)
    k           : número de vecinos
    fit_coords  : si None, se fittea con 'coords' (caso train);
                  si se pasa array, se usa como conjunto de entrenamiento del KNN (caso test).
                  Esto evita data leakage: el KNN del test se fittea con las coords de train.

    Retorna
    -------
    df con columnas nuevas lag_{feature}_k{k}
    """
    df = df.copy()

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="haversine")

    if fit_coords is None:
        nn.fit(np.radians(coords))
        _, indices = nn.kneighbors(np.radians(coords))
        indices = indices[:, 1:]  # excluir el propio punto
        source_df = df
    else:
        nn.fit(np.radians(fit_coords))
        _, indices = nn.kneighbors(np.radians(coords))
        source_df = None  # los valores del vecindario vienen del train externo

    for feat in features:
        col_name = f"lag_{feat}_k{k}"
        if source_df is not None:
            values = source_df[feat].values
            df[col_name] = np.mean(values[indices], axis=1)
        else:
            raise ValueError(
                "Cuando se pasa fit_coords, también debes pasar el DataFrame de train "
                "para extraer los valores de los vecinos. Usa compute_spatial_lags_test()."
            )

    return df


def compute_spatial_lags_train(df_train: pd.DataFrame, features: list,
                                coords_train: np.ndarray, k: int = 15):
    """
    Calcula spatial lags para el conjunto de entrenamiento.
    Fittea el KNN con coords_train y transforma train.
    Devuelve (df_train_con_lags, nn_fitted).
    """
    df = df_train.copy()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="haversine")
    nn.fit(np.radians(coords_train))
    _, indices = nn.kneighbors(np.radians(coords_train))
    indices = indices[:, 1:]  # excluir el propio punto

    for feat in features:
        values = df[feat].values
        df[f"lag_{feat}_k{k}"] = np.mean(values[indices], axis=1)

    return df, nn


def compute_spatial_lags_test(df_test: pd.DataFrame, df_train: pd.DataFrame,
                               features: list, coords_test: np.ndarray,
                               nn_fitted: NearestNeighbors, k: int = 15) -> pd.DataFrame:
    """
    Calcula spatial lags para el conjunto de test usando el KNN fitteado en train.
    Los valores del lag provienen del DataFrame de train (sin leakage).
    """
    df = df_test.copy()
    _, indices = nn_fitted.kneighbors(np.radians(coords_test))

    for feat in features:
        train_values = df_train[feat].values
        df[f"lag_{feat}_k{k}"] = np.mean(train_values[indices], axis=1)

    return df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Evalúa un modelo en escala logarítmica y original.
    y_true y y_pred deben estar en escala log.
    """
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log  = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)

    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig  = mean_absolute_error(y_true_orig, y_pred_orig)

    metrics = {
        "modelo":    model_name,
        "RMSE_log":  round(rmse_log, 6),
        "MAE_log":   round(mae_log, 6),
        "R2":        round(r2, 6),
        "RMSE_orig": round(rmse_orig, 2),
        "MAE_orig":  round(mae_orig, 2),
    }

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  RMSE (log):    {metrics['RMSE_log']:.4f}")
    print(f"  MAE  (log):    {metrics['MAE_log']:.4f}")
    print(f"  R²:            {metrics['R2']:.4f}")
    print(f"  RMSE (€/$):    {metrics['RMSE_orig']:,.0f}")
    print(f"  MAE  (€/$):    {metrics['MAE_orig']:,.0f}")
    print(f"{'='*45}\n")

    return metrics


def compute_moran_i(residuals: np.ndarray, coords: np.ndarray, k: int = 8):
    """
    Calcula el Índice de Moran I de los residuos usando pesos KNN.
    Requiere esda y libpysal.
    """
    try:
        from libpysal.weights import KNN as KNNWeights
        import esda

        w = KNNWeights.from_array(coords, k=k)
        w.transform = "r"
        mi = esda.Moran(residuals, w)
        print(f"  Moran's I: {mi.I:.4f}  (p-valor: {mi.p_sim:.4f})")
        return {"moran_i": round(float(mi.I), 6), "p_value": round(float(mi.p_sim), 4)}
    except ImportError:
        print("  [AVISO] esda/libpysal no disponible. Instala con: pip install esda libpysal")
        return {"moran_i": None, "p_value": None}


def save_model(model, name: str):
    models_dir = os.path.join(OUTPUTS_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def save_metrics(metrics: dict, filename: str):
    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas guardadas en: {path}")


def set_plot_style():
    plt.rcParams.update({
        "figure.figsize":    (10, 6),
        "figure.dpi":        120,
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "legend.fontsize":   11,
        "lines.linewidth":   1.8,
        "savefig.bbox":      "tight",
        "savefig.dpi":       150,
    })
