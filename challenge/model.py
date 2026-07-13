import pandas as pd
import numpy as np
from typing import Tuple, Union, List


class ReplenishmentModel:

    def __init__(
        self
    ):
        self._model = None  # El modelo debe guardarse en este atributo.

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
        df_out['dia_semana'] = df_out['fecha_dt'].dt.dayofweek

        # Componentes cíclicos del mes
        df_out['mes_sin'] = np.sin(2 * np.pi * df_out['mes'] / 12.0)
        df_out['mes_cos'] = np.cos(2 * np.pi * df_out['mes'] / 12.0)

        # Variable de tendencia
        fecha_minima = df_out['fecha_dt'].min() 
        df_out['tendencia_dias'] = (df_out['fecha_dt'] - fecha_minima).dt.days

        # CASO A: Pipeline de entrenamiento
        if target_column is not None:
            if 'tipo_movimiento' in df_out.columns:
                df_out = df_out[df_out['tipo_movimiento'] == 'S'].copy()
            
            # Calculo de variables linea_encoded y uso_encoded usando el consumo promedio por categoria
            self.gtin_to_linea_media = df_out.groupby('gtin')[target_column].transform('mean').groupby(df_out['gtin']).first().to_dict()
            self.gtin_to_uso_media = df_out.groupby('gtin')[target_column].transform('mean').groupby(df_out['gtin']).first().to_dict()

            # Asignar los encodings calculados internamente a la matriz
            df_out['linea_encoded'] = df_out['gtin'].map(self.gtin_to_linea_media)
            df_out['uso_encoded'] = df_out['gtin'].map(self.gtin_to_uso_media)

            features = df_out[self.feature_cols].copy()
            target = df_out[[target_column]].copy()
            return features, target         
        
        # CASO B: Pipeline de inferencia / predicción (consumo desde API)
        else:
            # Si aparece un gtin desconocido, se usa valor por defecto para evitar errores
            df_out['linea_encoded'] = df_out['gtin'].map(self.gtin_to_linea_media).fillna(0.0)
            df_out['uso_encoded'] = df_out['gtin'].map(self.gtin_to_uso_media).fillna(0.0)

            features = df_out[self.feature_cols].copy()
            features['fecha'] = data['fecha'].values 
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
        return

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
        return

    def save(
        self,
        path: str
    ) -> None:
        """
        Guarda el modelo entrenado en disco.

        Args:
            path (str): ruta donde guardar el modelo.
        """
        return

    def load(
        self,
        path: str
    ) -> None:
        """
        Carga un modelo entrenado desde disco.

        Args:
            path (str): ruta desde donde cargar el modelo.
        """
        return