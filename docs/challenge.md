# Análisis Exploratorio de Datos (EDA)

## 1. Calidad e integridad de datos
*   Se confirmó mediante operaciones de conjuntos que los 36 códigos GTIN coinciden simétricamente en las tres fuentes de datos, asegurando un cruce limpio sin registros huérfanos.
*   El análisis de frecuencias demostró que el dataset ya se encuentra consolidado con un único registro por día para cada combinación de (GTIN + Tipo de Movimiento). La coexistencia de filas para un mismo día corresponde a flujos opuestos de entrada 'E'(reposición) y salidas 'S' (consumo).

## 2. Descriptivos y distribución del consumo
 A partir de los descriptivos de consumo, se puede concluir que existen medicamentos de alta rotación (consumo constante diario) con promedio diario alredor de 6 unidades. Mientras que otros tienen un patron diario de consumo bajo o medio pero con mayor cantidad consumida promedio. Lo anterior sugiere que existen comportamientos de consumo diferentes dentro de la muestra, resultando relevante para una posterior planificación de abastecimiento. 

 ![alt text](Grafico-tendencia-estacionalidad.png)

 La gráfica de tendencia y estacionalidad por linea terapéutica arroja un claro patrón de consumo por línea. Se distingue tres agrupaciones o clusters según consumo:
 - Grupo de medicamentos de tendencia al alza sin estacionalidad (cardiovasculares)
 - Grupo de medicamentos con estacionalidad invernal (emergencias, dermatológicos y respiratorios)
 - Grupo de medicamentos con inelasticidad de consumo temporal (enfermedades crónicas y quirúrgicos)
 Sin embargo al revisar los mismos comportamientos dentro de cada línea terapéutica, los resultados son mixtos, es decir que el comportamiento de consumo no depende de la linea terapéutica. A modo de ejemplo se muestran la linea de enfermedades cronicas. Donde se observa que tres medicamentos presentan una tendencia positiva, mientras que el resto se presenta invariante en el tiempo. 
 
 ![alt text](Grafico-tendencia-enf-cronicas.png)


En cuanto a efecto intrasemanal o efecto dia de semana existiría una regularidad de consumo promedio intrasemanal para casi todas las lineas terapéuticas, es decir que no existirían shocks de consumo para estos datos, a diferencia de una farmacia donde el mayor consumo podría situarse los fines de semana. Sin embargo, a nivel individual si podrian existir patrones como el caso de medicamento dermatológico, que tiene una caída los dias sábados. 

![alt text](Grafico-efecto-dia-semana.png)

El Analisis exploratorio, indica que existen claros clusters de consumo con respecto a los medicamentos, por lo que el modelo de ajuste debe contemplar las tres variantes tendencia alcista, estacionalidad invernal e invariabilidad temporal de manera de mejorar las métricas de desempeño y la precisión de las predicciones.

# Modelamiento (`challenge/modelo.py`)

## 1. Selección del Algoritmo
Se seleccionó un `RandomForestRegressor`. Esta decisión se fundamenta en los hallazgos del EDA: 
* Al existir una demanda inelástica pero con tendencias crecientes a largo plazo (Cardiovascular) y picos invernales (Respiratorio), este modelo aísla de forma óptima las interacciones de variables categóricas sin necesidad de transformaciones de estacionariedad econométrica clásica.

## 2. Preprocesamiento e ingeniería de características (Features)
La matriz de diseño se construyó a partir de variables de orden temporal:
* Se utilizó componentes armónicos (`mes_sin`, `mes_cos`) para encapsular la estacionalidad del consumo.
* Una variable de conteo lineal continuo (`tendencia_dias`) para guiar la proyección frente al crecimiento interanual.
* Dos variables de categorías operativas (línea terapéutica  y uso) reemplazando por la media histórica de su demanda mediante target encoding, agrupando los registros nuevos o faltantes bajo la etiqueta 'OTRO'.
* También se incluyo variable de `dia_semana` para abordar variaciones en consumo según dia de la semana. 

El modelo ajustado, arrojó los siguientes resultados: 
*   MAE Baseline (Media por Producto): 4.7828
*   MAE Modelo (Random Forest):        3.4401
*   Mejora respecto al benchmark:      28.08%

El modelo arroja una mejora del 28.08% con respecto al modelo de referencia. Pese a lo anterior, según lo determinado en el EDA, se recomienda generar un modelo distinto por cada tipo de demanda de medicamentos para un mejor desempeño y precisión en las predicciones. Por simplicidad en este caso se prosigue con el modelo único dado que cumple las especificaciones solicitadas. 

Optimizar el modelo y reducir su complejidad es vital para un mejor desempeño y menor uso de recursos por lo que se realizó una reducción de features obteniendo la siguiente mejora: 

*   MAE Baseline (Media por Producto): 4.7828
*   MAE Modelo (Random Forest):        3.3560
*   Mejora respecto al benchmark:      29.83%

Al realizar ajuste de features quitando las variables de dia de semana y las categoricas el modelo mejora con una reducción del 29.83% con respecto al caso base. Un modelo más simple que no requiera de datos adicionales categóricos es vital para evitar problemas de sobrecarga y pueda superar pruebas de stress.

## 3. Decisiones de Diseño y Arquitectura

Tratamiento de GTINs (Productos):

*   Durante el preprocesamiento, los códigos de barra (GTIN) se transforman de manera estricta a tipos numéricos enteros (int64).

*    Seguridad contra Productos Desconocidos: El modelo guarda en un set  (trained_gtins) los identificadores únicos con los que fue entrenado. Si la API recibe un producto fuera de este set durante la inferencia, se mitiga el riesgo de una predicción errónea rechazando la petición con un error HTTP 400.

Tratamiento Temporal: La columna de fecha se procesa mediante técnicas de ingeniería de características (Feature Engineering) extrayendo componentes temporales (mes o estacionalidad) requeridos por el pipeline de entrenamiento.

# Documentación de la API de Inferencia (`challenge/api.py`)

## Arquitectura y Decisiones de Diseño

### 1. Robustez contra datos inválidos
Para garantizar que la API responda con los códigos de error correctos exigidos por el challenge, se implementaron las siguientes estrategias:
* Manejo global de excepciones (`RequestValidationError`): Por defecto, FastAPI responde con códigos `422 Unprocessable Entity` cuando falla la validación de Pydantic. Se configuró un interceptor global (`@app.exception_handler`) que captura estos fallos y los transforma en un código **`HTTP 400 Bad Request`**, extrayendo descripciones legibles en texto plano de cada error para evitar fallos de serialización JSON.
* Validación de fechas: Se utilizó el validador personalizado `@field_validator('fecha')`. Esto asegura que fechas inexistentes en el calendario (como el 30 de febrero o el 32 de diciembre) o strings corruptos sean rechazados inmediatamente en la puerta de entrada.

### 2. Flexibilidad de Contratos
Diferentes pruebas del challenge consumen la API usando estructuras de datos variadas. Para absorber esto de forma transparente, el esquema `PredictionRequest` se configuró con soporte dinámico para:
* El campo **`instances`** (usado comúnmente en servicios de cloud).
* El campo **`products`** (usado en tests específicos de este challenge).

El endpoint resuelve internamente cuál llave contiene la lista de elementos, unificándola en un DataFrame estructurado.

---

## Especificación de Endpoints

### 1. Control de Salud (Health Check)
Verifica que el servicio esté arriba y responda solicitudes básicas.

* **Método:** `GET`
* **Ruta:** `/health`
* **Código de Respuesta de Éxito:** `200 OK`


### 2. Predicción de Reabastecimiento

Recibe un listado de productos y fechas para calcular la estimación del algoritmo.

* **Método:** `POST`
* **Ruta:** `/predict`
* **Código de Respuesta de Éxito:** `200 OK`


### 3. Respuestas de Error Comunes (400 Bad Request)

* Error de Formato/Fecha Inexistente: Ocurre si la fecha no tiene formato YYYY-MM-DD o no pertenece al calendario real.
* Producto Desconocido: Ocurre si el GTIN enviado no estuvo presente en el dataset con el que se entrenó el modelo.

