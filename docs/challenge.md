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

El Analisis exploratorio, indica que existen claros clusters de consumo con respecto a los medicamentos, por lo que el modelo de ajuste debe contemplar las tres variantes tendencia alcista, estacionalidad invernal e invariabilidad temporal de manera de mejorar las métricas de desempeño y la precisión de las predicciones.

