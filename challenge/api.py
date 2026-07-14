import os
import pandas as pd
import fastapi
from typing import List, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from challenge.model import ReplenishmentModel

app = fastapi.FastAPI()

model = ReplenishmentModel()

# Cargar el modelo si existe
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
else:
    print(f"Modelo no encontrado en {MODEL_PATH}. Ejecuta el entrenamiento primero.")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Captura los fallos de estructura o de formato de fecha de Pydantic,
    extrayendo los mensajes de texto plano para evitar problemas de serialización JSON.
    """
    error_messages = []
    for error in exc.errors():
        # Extraemos una descripción legible del fallo (ej: 'Field required' o el mensaje del ValueError)
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Error de validación")
        error_messages.append(f"[{loc}]: {msg}")

    return JSONResponse(
        status_code=400,
        content={
            "detail": "Error 400: Estructura del payload incorrecta o valores inválidos.",
            "errors": error_messages
        }
    )

# Validación y control de errores
class PredictionInputItem(BaseModel):
    gtin: Union[int, str]
    fecha: str

    @field_validator('fecha')
    @classmethod
    def validar_formato_fecha(cls, value: str) -> str:
        try:
            pd.to_datetime(value, format="%Y-%m-%d", errors='raise')
            return value
        except Exception:
            raise ValueError("Formato de fecha no válido")

class PredictionRequest(BaseModel):
    instances: Optional[List[PredictionInputItem]] = None
    products: Optional[List[PredictionInputItem]] = None

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(payload: PredictionRequest) -> dict:
    raw_items = payload.instances or payload.products

    if not raw_items:
        raise HTTPException(
            status_code=400, 
            detail="Error 400: Debe proporcionar una lista válida en 'instances' o 'products'."
        )

    # Si el modelo no está cargado o no está entrenado, respondemos 503
    if model is None or not getattr(model, "is_trained", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no ha sido entrenado en el servidor."
        )

    # Convertir el payload validado a un DataFrame para el preprocesamiento
    try:
        items = [item.model_dump() for item in raw_items]
        df_raw = pd.DataFrame(items)
     
        if 'gtin' not in df_raw.columns:
            raise KeyError("La columna 'gtin' es requerida.")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Estructura de datos inválida: {str(e)}")

    # Validación de productos desconocidos.
    if model.is_trained and len(model.trained_gtins) > 0:
        for gtin_solicitado in df_raw['gtin']:
            if int(gtin_solicitado) not in model.trained_gtins:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Error 400: El producto con GTIN {gtin_solicitado} es desconocido para el modelo."
                )
    try:
        # Preprocesamiento
        features = model.preprocess(data=df_raw)
        # Obtener las predicciones
        predictions = model.predict(features)
        return {"predict": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))