import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.ensemble import RandomForestRegressor

class ReplenishmentModel:

    def __init__(
        self
    ):
        self._model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
        
        self.feature_cols = [
            'gtin', 'año', 'mes_sin', 'mes_cos',
        ]
        
        # Una lista para la validación de la API en producción
        self.trained_gtins = set()      

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos crudos para entrenamiento o predicción.

        Args:
            data (pd.DataFrame): datos crudos.
            target_column (str, opcional): si se establece, se retorna el target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features y target.
            o
            pd.DataFrame: features.
        """
        df_out = data.copy()

        # Parseo de fechas e ingeniería de variables temporales básicas
        df_out['fecha_dt'] = pd.to_datetime(df_out['fecha'])
        df_out['año'] = df_out['fecha_dt'].dt.year
        df_out['mes'] = df_out['fecha_dt'].dt.month

        # Componentes cíclicos del mes
        df_out['mes_sin'] = np.sin(2 * np.pi * df_out['mes'] / 12.0)
        df_out['mes_cos'] = np.cos(2 * np.pi * df_out['mes'] / 12.0)

        # CASO A: Pipeline de entrenamiento
        if target_column is not None:
            if 'tipo_movimiento' in df_out.columns:
                df_out = df_out[df_out['tipo_movimiento'] == 'S'].copy()
            
            # Guardamos los GTINs de entrenamiento para la validación de la API
            self.trained_gtins = set(df_out['gtin'].unique())

            # Añade fecha a la salida por requerimiento
            cols_to_return = self.feature_cols + ['fecha']
            features = df_out[cols_to_return].copy()
            target = df_out[[target_column]].copy()
            return features, target         
        
        # CASO B: Pipeline de inferencia / predicción (consumo desde API)
        else:
            cols_to_return = self.feature_cols + ['fecha']
            features = df_out[cols_to_return].copy()
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Entrena el modelo con los datos preprocesados.

        Args:
            features (pd.DataFrame): datos preprocesados.
            target (pd.DataFrame): variable objetivo.
        """
        X_train = features[self.feature_cols]
        self._model.fit(X_train, target.values.ravel())
        self.is_trained = True

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[dict]:
        """
        Predice el consumo para una lista de productos.

        Args:
            features (pd.DataFrame): datos preprocesados.

        Returns:
            (List[dict]): predicciones con keys 'fecha' y 'cantidad'.
        """
        date_original = features['fecha'].values
        X_inference = features[self.feature_cols]

        # Manejo en caso de predict sin ajuste
        if not self.is_trained:
            raw_preds = np.zeros(len(X_inference))
        else:
            raw_preds = self._model.predict(X_inference)

        # Chequeo que cantidad cosnumida no puede ser negativa
        preds_clipped = np.clip(raw_preds, a_min=0, a_max=None)

        results = []
        for date, pred in zip(date_original, preds_clipped):
            results.append({
                "fecha": str(date),
                "cantidad": float(pred)
            })
        return results

    def save(
        self,
        path: str
    ) -> None:
        """
        Guarda el modelo entrenado en disco.

        Args:
            path (str): ruta donde guardar el modelo.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self._model,
                "is_trained": self.is_trained,
                "trained_gtins": self.trained_gtins
            }, f)

    def load(
        self,
        path: str
    ) -> None:
        """
        Carga un modelo entrenado desde disco.

        Args:
            path (str): ruta desde donde cargar el modelo.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el modelo en {path}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._model = data["model"]
            self.is_trained = data["is_trained"]
            self.trained_gtins = data["trained_gtins"]