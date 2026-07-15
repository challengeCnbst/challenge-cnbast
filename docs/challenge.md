## Análisis Exploratorio de Datos (EDA)

### 1. Calidad e integridad de datos
*   Se confirmó mediante operaciones de conjuntos que los 36 códigos GTIN coinciden simétricamente en las tres fuentes de datos, asegurando un cruce limpio sin registros huérfanos.
*   El análisis de frecuencias demostró que el dataset ya se encuentra consolidado con un único registro por día para cada combinación de (GTIN + Tipo de Movimiento). La coexistencia de filas para un mismo día corresponde a flujos opuestos de entrada 'E'(reposición) y salidas 'S' (consumo).

### 2. Descriptivos y distribución del consumo
 A partir de los descriptivos de consumo, se puede concluir que existen medicamentos de alta rotación (consumo constante diario) con promedio diario alredor de 6 unidades. Mientras que otros tienen un patron diario de consumo bajo o medio pero con mayor cantidad consumida promedio. Lo anterior sugiere que existen comportamientos de consumo diferentes dentro de la muestra, resultando relevante para una posterior planificación de abastecimiento. 

 ![alt text](Grafico-tendencia-estacionalidad.png)

 La gráfica de tendencia y estacionalidad por linea terapéutica arroja un claro patrón de consumo por línea. Se distingue tres agrupaciones o clusters según consumo:
 - Grupo de medicamentos de tendencia al alza sin estacionalidad (cardiovasculares)
 - Grupo de medicamentos con estacionalidad invernal (emergencias, dermatológicos y respiratorios)
 - Grupo de medicamentos con inelasticidad de consumo temporal (enfermedades crónicas y quirúrgicos).

 Sin embargo al revisar los mismos comportamientos dentro de cada línea terapéutica, los resultados son mixtos, es decir que el comportamiento de consumo no depende de la linea terapéutica. A modo de ejemplo se muestran la linea de enfermedades crónicas. Se observa que tres medicamentos presentan una tendencia positiva, mientras que el resto serían invariantes en el tiempo. 
 
 ![alt text](Grafico-tendencia-enf-cronicas.png)


En cuanto a efecto intrasemanal o efecto dia de semana existiría una regularidad de consumo promedio intrasemanal para casi todas las lineas terapéuticas, es decir que no existirían shocks de consumo para estos datos, a diferencia de una farmacia donde el mayor consumo podría situarse los fines de semana. Sin embargo, a nivel individual si podrian existir patrones como el caso de medicamento dermatológico, que tiene una caída los dias sábados. 

![alt text](Grafico-efecto-dia-semana.png)

El Analisis exploratorio, indica que existen claros clusters de consumo con respecto a los medicamentos, por lo que el modelo de ajuste debe contemplar las tres variantes tendencia alcista, estacionalidad invernal e invariabilidad temporal de manera de mejorar las métricas de desempeño y la precisión de las predicciones.

## Parte I: Modelamiento (`challenge/modelo.py`)

### 1. Selección del Algoritmo
Se seleccionó un `RandomForestRegressor`. Esta decisión se fundamenta en los hallazgos del EDA: 
* Al existir una demanda inelástica pero con tendencias crecientes a largo plazo (Cardiovascular) y picos invernales (Respiratorio), este modelo aísla de forma óptima las interacciones de variables categóricas sin necesidad de transformaciones de estacionariedad econométrica clásica.

### 2. Preprocesamiento e ingeniería de características (Features)
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

Al realizar ajuste de features quitando las variables de dia de semana y las categoricas, el modelo mejora con una reducción del 29.83% con respecto al caso base. Un modelo más simple que no requiera de datos adicionales categóricos es vital para evitar problemas de sobrecarga y pueda superar pruebas de stress.

### 3. Decisiones de Diseño y Arquitectura

Tratamiento de GTINs (Productos):

*   Durante el preprocesamiento, los códigos de barra (GTIN) se transforman de manera estricta a tipos numéricos enteros (int64).

*    Seguridad contra Productos Desconocidos: El modelo guarda en un set  (trained_gtins) los identificadores únicos con los que fue entrenado. Si la API recibe un producto fuera de este set durante la inferencia, se mitiga el riesgo de una predicción errónea rechazando la petición con un error HTTP 400.

Tratamiento Temporal: La columna de fecha se procesa mediante técnicas de ingeniería de características (Feature Engineering) extrayendo componentes temporales (mes o estacionalidad) requeridos por el pipeline de entrenamiento.

## Parte II: Documentación de la API de inferencia (`challenge/api.py`)

### Arquitectura y Decisiones de Diseño

### 1. Robustez contra datos inválidos
Para garantizar que la API responda con los códigos de error correctos exigidos por el challenge, se implementaron las siguientes estrategias:
* Manejo global de excepciones (`RequestValidationError`): Por defecto, FastAPI responde con códigos `422 Unprocessable Entity` cuando falla la validación de Pydantic. Se configuró un interceptor global (`@app.exception_handler`) que captura estos fallos y los transforma en un código **`HTTP 400 Bad Request`**, extrayendo descripciones legibles en texto plano de cada error para evitar fallos de serialización JSON.
* Validación de fechas: Se utilizó el validador personalizado `@field_validator('fecha')`. Esto asegura que fechas inexistentes en el calendario (como el 30 de febrero o el 32 de diciembre) o strings corruptos sean rechazados.

### 2. Flexibilidad de Contratos
Se gestiono que la API pueda soportar estructuras de datos variadas. Se configuró el esquema `PredictionRequest` con soporte para:
* El campo **`instances`** (usado comúnmente en servicios de cloud).
* El campo **`products`** (usado en tests específicos de este challenge).

El endpoint resuelve internamente cuál llave contiene la lista de elementos, unificándola en un DataFrame estructurado.


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

## Parte III: Tests 

### 1. Pruebas de la API
Valida la disponibilidad del servicio (`/health`), el flujo de predicción (`/predict`), el bloqueo seguro ante formatos de fecha inválidos (`HTTP 400`) y la restricción para productos (GTINs) fuera del set de entrenamiento (`HTTP 400`).
```bash
make api-test
```
### 2. Pruebas del Modelo

Valida la integridad del pipeline de entrenamiento, los pasos del preprocesamiento y la consistencia dimensional de la salida del estimador.
```bash
make model-test
```
Nota: Las pruebas de la API fueron validadas en un entorno limpio utilizando versiones alineadas de fastapi, starlette y httpx para asegurar la estabilidad del cliente de pruebas, las modificaciones fueron añadidas en los archivos de dependencias correspondientes.


## Parte IV: Deploy y resultados del test de estrés (Locust)

### Arquitectura de Contenedores y Recursos

Para asegurar un rendimiento óptimo del modelo matemático bajo escenarios de alto estrés sin incurrir en sobrecostos redundantes, se definió la siguiente topología de recursos asignados por instancia:

*   **Imagen Base:** `python:3.11-slim` (Minimiza el peso del contenedor y reduce la superficie de vulnerabilidades en producción).
*   **Procesador (CPU):** `1 vCPU` (Balanceado para ejecutar operaciones con matrices de decisión del modelo Random Forest).
*   **Memoria RAM:** `1 GiB` (Espacio para la carga estática de `model.pkl` en memoria y la manipulación de DataFrames).
*   **Estrategia de Escalado:** Mínimo `0` instancias en reposo (costo cero sin tráfico) y un techo estricto de `--max-instances 3` gestionado por el balanceador de carga de GCP ante picos de concurrencia.

### Test de estrés

El endpoint /predict fue sometido a una prueba de carga simulando 100 usuarios concurrentes con una tasa de aceleración (spawn rate) de 1 usuario/segundo durante un ciclo continuo.

* URL de Producción Activa: https://replenishment-api-823361295040.southamerica-west1.run.app

* Peticiones Totales Procesadas: 3,279 solicitudes exitosas.

* Tasa de Errores (Failure %): 0.00% (Ninguna degradación ni caídas por falta de memoria OOM).

* Rendimiento Promedio: 54.85 req/s sostenidos.

* Percentil 50 (Mediana de respuesta): 530 ms.

* Latencia Mínima: 32 ms.

## Parte V: Integración y Despliegue Continuo (CI/CD)

### 1. Integración Continua (CI) — `ci.yml`

El flujo de CI se dispara de manera automática ante cada `push` en cualquier rama de desarrollo (`feature/*`), `main`, o al abrir un `Pull Request` hacia la rama `main`. 

Su objetivo principal es ejecutar pruebas sobre un entorno limpio e idéntico al de producción.

*   **Entorno:** Instancia aislada sobre `ubuntu-latest` configurada con **Python 3.11** para mantener consistencia exacta con la imagen de Docker.
*   **Optimización de Dependencias:** Implementa el mecanismo de caché nativo de GitHub para `pip`. Esto reduce el tiempo promedio de ejecución al evitar la descarga redundante de paquetes pesados (`pandas`, `scikit-learn`, `fastapi`).
*   **Entrenamiento del Modelo:** Si el archivo no se encuentra en el espacio de trabajo, el pipeline compila una versión ligera del modelo ejecutando el script `challenge/train.py` para levantar la API de forma íntegra.
*   **Separación de Pruebas:** Para facilitar el diagnóstico de errores ante fallos las pruebas se dividen en dos fases independientes con reportes en consola:
    1.  **Run Model Tests:** Validación de la lógica y aserciones del modelo predictivo.
    2.  **Run API Tests:** Validación del comportamiento del servidor HTTP y los contratos de datos de FastAPI.

### 2. Despliegue Continuo (CD) — `cd.yml`

El flujo de CD se activa de manera exclusiva al consolidar cambios de forma exitosa en la rama principal (`push` a `main` o tras aprobar un `Pull Request`).

Este pipeline automatiza por completo el flujo de despliegue manual que realizamos inicialmente en Google Cloud Platform.

#### Flujo de Ejecución del pipeline de CD:

1.  **Autenticación Segura (IAM):** Se conecta a la nube mediante la acción oficial `google-github-actions/auth`, consumiendo de forma encriptada las credenciales JSON de la Service Account desde los secretos del repositorio (`secrets.GCP_SA_KEY`).
2.  **Alineación Geográfica de Docker:** Configura el daemon de Docker local en la máquina virtual de GitHub para interactuar con el repositorio privado de **Artifact Registry** localizado en la región chilena (`southamerica-west1-docker.pkg.dev`).
3.  **Compilación y Despliegue de Imagen:**
*   Construye la imagen a partir del `Dockerfile`.
*   Sube la imagen etiquetada como `latest` al registro de artefactos en la nube.
*   Actualiza el servicio de **Cloud Run** inyectando de manera forzada la identidad de la Service Account asignada (`--service-account`) con el fin de evadir bloqueos y asegurar la correcta delegación de permisos de invocación.


## Parte VI: Predicción de Reabastecimiento

Pasar de predecir el **consumo de stock** (demanda) a anticipar el **próximo pedido de reabastecimiento** por producto se requiere transformar un modelo puramente matemático en un modelo de toma de decisiones operativas. 

Para resolver este problema en un entorno logístico real de distribución de fármacos e insumos, el objetivo central debe ser: **minimizar el costo total de la cadena de suministro garantizando un nivel de servicio óptimo (cero quiebres de stock)**, sujeto a restricciones físicas, normativas y burocráticas del sector salud.

---

### 1. El Algoritmo de Decisión Logística (ROP Dinámico)

Para cada gtin, se debe obtener el punto de reorden, para ello se debe estimar el punto en donde la suma del consumo esperado menos el stock actual sea igual al inventario critico. 

Dado que no podemos conocer con certeza el lead time para cada fármaco se tendrá que parametrizar dentro de un rango aceptable de 10 a 15 dias. A partir de los datos podriamos:  
 * Calcular el inventario critico promedio, tomando el stock historico promedio para cada gtin justo antes de una nueva entrada de stock.
 * Cálcular el inventario máximo permitido, tomando el promedio de los maximos de stock.  

Entonces para calcular el punto de reabastecimiento, utilizariamos el modelo de estimacion de consumo de este challenge, parametrizando un lead time logistico dentro del rango de 10 a 15 dias y en base a las predicciones y el stock histórico, se puede estimar el dia de reabastecimiento cuando cumpla esta condición:

$$\text{Inventario critico} = \text{Inventario}_{t} - \sum_{i=1}^{15} (\text{Consumo predicho}_{t+i}) $$

Ahora bien, dado que se detectaron tres comportamientos de demanda, el tratamiento de reabastecimiento debe ser distinto para cada tipo, por lo que resultaría util tener modelos distintos para cada cluster de demanda y evitar quiebres de stock por no detectar correctamente tendencias alcistas o estacionalidades donde el consumo aumenta en el tiempo y el reabastecimiento se hace más crítico. 

---

### 2. Caso Variables de Entrada y Restricciones de Negocio Complementarias

Para que el modelo predictivo sea viable y seguro en un escenario real, debe integrar las siguientes dimensiones operativas:

#### A. Restricciones Físicas y de Producto (Sanitarias)
*   **Vida Útil de los Medicamentos / Insumos:** 
    *   Para medicamentos o insumos con corta vida útil, resulta vital el analizar la rotacion e inventario de los mismos y evitar mermas por sobrestock. También es útil esta variable en caso de almacenar grandes cantidades para luego distribuir a lo largo del año.  
*   **Capacidad de Almacenamiento Físico (Bodega y Farmacia):**
    *   Las farmacias de los centros de salud y los depósitos intermedios tienen un espacio físico limitado (en metros cúbicos o unidades máximas equivalentes). El modelo debe incluir una restricción de capacidad máxima, dado que el algoritmo de reorden no puede sugerir un volumen de compra que supere la capacidad de almacenamiento disponible.

#### B. Modelado de Lead Time Administrativo
*   **Tiempos de Adquisición (Licitación, Convenio Marco o Trato Directo):**
    *   En el sector público de salud, el Lead Time está fuertemente impactado por los tiempos administrativos de compra. El modelo de reabastecimiento debe segmentar y modelar dinámicamente estos tiempos de espera según el tipo de compra:
        *   **Trato Directo / Convenio Marco:** Lead Times cortos.
        *   **Licitaciones Públicas:** Lead Times extensos y variables (meses) debido a la burocracia de postulación, adjudicación y firmas de contratos. El punto de reabastecimiento debe adaptase para productos que operan bajo licitaciones largas para evitar desabastecimiento.

#### C. Estrategia de Abastecimiento y Análisis de Costos
Según sea el producto podría ser más conveniente optar por otra estrategia de optimización logística.

*   **Abastecimiento Reactivo (Cantidad Fija, Periodo Variable):** Se coloca una orden de compra por un lote óptimo (EOQ) fijo cada vez que el inventario toca el ROP. Es el más económico para productos de bajo costo unitario y alta rotación.
*   **Abastecimiento Regulado (Cantidad Variable, Periodo Fijo):** Se revisa el inventario en intervalos de tiempo constantes y se pide la cantidad necesaria para volver a un nivel máximo predefinido. Es ideal para optimizar costos de envío consolidados con laboratorios específicos que ganaron licitaciones de múltiples productos.