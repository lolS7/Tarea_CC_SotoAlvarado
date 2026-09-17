# Clasificación de éxito de canciones

Predice `EsExito = (popularity > 50)` con scikit-learn y sirve inferencias mediante FastAPI. Popularidad igual a 50 pertenece a la clase 0. La API recibe ocho variables; nunca recibe `popularity`.

## Archivos

| Archivo | Función |
| --- | --- |
| `Model_01.py` | Carga, preprocesamiento, entrenamiento, comparación y evaluación de los modelos sin guardar |
| `Model_02.py` | Ejecuta el entrenamiento de Model_01 y serializa el pipeline completo y los metadatos |
| `Model_03_read_pkl.py` | Lectura del PKL e inferencia interactiva o desde JSON |
| `Model_04_FASTAPI.py` | Punto de entrada compatible con la estructura de referencia |
| `model/model.pkl` | Pipeline completo: imputación, escalamiento, codificación y clasificador |
| `model/metadata.json` | Versiones, variables ordenadas, huellas SHA-256, partición y métricas |
| `app/main.py` | API FastAPI con validación de entrada |
| `requirements.txt`, `runtime.txt`, `Procfile` | Dependencias fijadas, Python y arranque cloud |
| `tests/verify_local.py` | Pruebas reales de HTTP y paridad con el PKL |
| `docs/` | Evidencia de entrenamiento, reproducibilidad y pruebas locales |

La referencia externa `Ref` y los documentos originales no se modificaron. Los scripts 01–04 conservan su función conceptual, adaptada al dataset de canciones; comparten código para evitar divergencias.

## Datos y fuente

Fuente pública identificada: [Top Hits Spotify from 2000–2019, Paradise Joy, Kaggle](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019). El archivo local proporcionado es `songs_normalize.csv`; su nombre y esquema corresponden a ese dataset. No se verificó identidad binaria contra la descarga de Kaggle. La huella del archivo utilizado queda en `model/metadata.json`.

Se dejó una copia local en `data/songs_normalize.csv`, excluida de Git. Para reproducir desde un clon, obtener el CSV de la fuente y colocarlo allí, o proporcionar `--data RUTA`. El servicio de inferencia no necesita el CSV.

Hay 2.000 filas originales y 1.941 después de eliminar 59 duplicados exactos. Clases: 1.610 éxitos y 331 no éxitos. No hay nulos en las ocho variables utilizadas. Se supera el mínimo de 500 filas y cuatro predictores, con variables categóricas.

## Instalación y ejecución

Usar Python **3.12.14** y las versiones fijadas para cargar el PKL, especialmente scikit-learn **1.5.2**.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Model_02.py --data data/songs_normalize.csv
.\.venv\Scripts\python.exe Model_03_read_pkl.py --json examples/cancion.json
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Para comparar sin guardar, ejecutar `Model_01.py`. También se puede arrancar con `uvicorn Model_04_FASTAPI:app`. En Linux/macOS, usar `.venv/bin/python` en lugar de `.venv\Scripts\python.exe`.

Abrir [Swagger UI](http://127.0.0.1:8000/docs). Salud: `GET /health`. Inferencia: `POST /prediccion/`.

```json
{
  "artist": "Britney Spears",
  "duration_ms": 211160,
  "danceability": 0.751,
  "loudness": -5.444,
  "key": 1,
  "genre": "pop",
  "energy": 0.834,
  "tempo": 95.053
}
```

`duration_ms` es un entero positivo en milisegundos; `danceability` y `energy` están entre 0 y 1; `tempo` es un número positivo en BPM; `loudness` es un número finito en dB; `key` es un entero de 0 a 11. `artist` y `genre` son textos no vacíos. `energy` y `tempo` son obligatorios: los JSON de la versión anterior deben incluirlos. Entradas inválidas o campos extra devuelven HTTP 422.

Los géneros se separan por comas y se codifican como indicadores binarios: `rock, pop` activa las categorías `rock` y `pop`. Se ignoran mayúsculas y espacios en los extremos, conservando categorías como `hip hop`, `R&B` y `Dance/Electronic`. La etiqueta original `set()` se conserva como categoría de género no especificado. Artista y tonalidad usan OneHotEncoder; géneros usan CountVectorizer. Las categorías nuevas se ignoran sin provocar errores. El vocabulario se aprende solo dentro del entrenamiento de cada pliegue.

Después de reemplazar el PKL, reiniciar Uvicorn con Ctrl+C y ejecutar nuevamente el comando de arranque para cargar el modelo nuevo; la recarga automática de código no garantiza recargar cambios de un archivo `.pkl`.

Ejemplo desde PowerShell, con el servidor activo:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/prediccion/ -Method Post -ContentType 'application/json' -InFile examples/cancion.json
```

Respuesta observada para ese ejemplo:

```json
{"EsExito": true, "ProbabilidadExito": 1.0}
```

Esta es una predicción del modelo, no una consulta a la etiqueta real. El ejemplo pertenece al dataset y no sirve como prueba independiente de generalización. La probabilidad es la estimación del bosque y no se ha calibrado: 1,0 no implica certeza sobre el éxito futuro.

## Método y resultados

Se elimina la duplicación exacta antes de separar los datos. `StratifiedGroupKFold` con cinco pliegues y semilla 42 reserva el primer pliegue para prueba; los grupos son artista+título, evitando que registros de una misma canción crucen la partición. Quedan 1.552 filas de entrenamiento y 389 de prueba. Los artistas sí pueden repetirse: esta evaluación no mide exclusivamente artistas nuevos.

Dentro de entrenamiento, validación cruzada agrupada de cinco pliegues genera predicciones fuera de muestra para 22 configuraciones:

- Regresión logística: `C` en `[0.1, 1, 10]` y `class_weight` en `[None, 'balanced']`.
- Bosque aleatorio: 100 árboles, `max_depth` en `[None, 12]`, `min_samples_leaf` en `[1, 4]`, `max_features` en `['sqrt', 0.2]` y los mismos dos pesos de clases.

Para cada configuración se evalúan 25 umbrales de probabilidad entre 0,20 y 0,80, en pasos de 0,025. Se maximiza accuracy sobre las predicciones fuera de muestra; los empates se resuelven por F1 macro, recall de éxitos y cercanía a 0,5. La cifra de validación cruzada se usa para seleccionar y puede ser optimista por la búsqueda; el conjunto de prueba no interviene en esa selección.

Todos los transformadores se ajustan dentro de cada pliegue. Se seleccionó un bosque aleatorio con `class_weight=None`, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1` y umbral **0,475**. Accuracy de selección: **0,8331**. El modelo final se ajustó solo en entrenamiento y se evaluó en prueba. `FixedThresholdClassifier` guarda el umbral dentro del pipeline: la API y la consola usan el mismo `predict()`.

El umbral de popularidad **50** define la etiqueta real. El umbral de probabilidad **0,475** define cuándo el clasificador responde `true`. Son valores con funciones distintas.

| Métrica de prueba | Modelo | Baseline de clase mayoritaria |
| --- | ---: | ---: |
| F1 macro | 0,4977 | 0,4559 |
| Accuracy | 0,8355 | 0,8380 |
| ROC AUC | 0,6150 | — |
| Balanced accuracy | 0,5177 | 0,5000 |
| Recall de éxitos | 0,9877 | 1,0000 |
| Precisión de éxitos | 0,8429 | 0,8380 |

Accuracy es ahora el criterio solicitado de selección. F1 macro da el mismo peso a ambas clases; ROC AUC mide discriminación sin depender del umbral; recall de éxitos permite medir los falsos negativos que motivaron el cambio. La matriz de confusión, con filas reales y columnas predichas en orden `[0, 1]`, es `[[3, 60], [4, 322]]`.

Se detectan 322 de 326 éxitos, pero solo 3 de 63 no éxitos. El modelo responde `true` a 382 de las 389 canciones de prueba. Su accuracy es ligeramente inferior a responder siempre `true`, por lo que no se demuestra una mejora de accuracy frente a esa regla. El umbral 0,475 ganó en validación cruzada, pero en este conjunto de prueba produce las mismas etiquetas que 0,5; se reporta la comparación en `test_metrics_at_0_5` de los metadatos.

El rendimiento discriminativo sigue siendo limitado. El dataset ya contiene canciones seleccionadas como top hits, por lo que no representa todo el catálogo ni permite afirmar que predice el éxito futuro de cualquier lanzamiento. El objetivo clasifica popularidad registrada; no hay evaluación temporal ni causal. La nueva accuracy no es comparable directamente con la versión anterior: cambió la etiqueta de `popularity > 60` a `popularity > 50`, aumentó la proporción de éxitos y cambió la partición estratificada. Tampoco se realizó un estudio de ablación que atribuya mejoras a cada variable incorporada. Para medir mejoras futuras con mayor rigor conviene disponer de un conjunto externo nuevo, dado que este dataset ya se ha inspeccionado durante el desarrollo.

## Verificación

```powershell
.\.venv\Scripts\python.exe tests/verify_local.py
```

La prueba inicia Uvicorn en un puerto libre de localhost y lo detiene al finalizar. Se verificaron 19 casos: límite de popularidad 50/51, umbral serializado, separación de géneros, salud, predicción idéntica al PKL, categorías nuevas, doce entradas inválidas y esquema OpenAPI. Evidencias: `docs/local_test_results.json` y `docs/server.log`. El entrenamiento comprueba paridad de probabilidades y etiquetas después de serializar. Se verificó también la carga del PKL en un proceso aislado, sin importar módulos del proyecto. `docs/reproducibility.json` registra la comparación con un segundo entrenamiento independiente.

## Cloud y GitHub

El modelo ocupa 1.330.647 bytes (aproximadamente 1,33 MB), menos de 100 MB. Solo usa componentes estándar de scikit-learn y rutas relativas al código, sin clases de serialización propias ni rutas de Windows en la inferencia. Instalar `requirements.txt`, usar la versión de Python indicada y ejecutar el `Procfile`:

```sh
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

La plataforma debe proporcionar `PORT`. Si no interpreta `runtime.txt` o `Procfile`, configurar manualmente esos valores en su interfaz. Se verificó la ejecución local en Windows; aún no se realizó despliegue ni prueba en un proveedor cloud.

Repositorio indicado: [lolS7/Tarea_CC_SotoAlvarado](https://github.com/lolS7/Tarea_CC_SotoAlvarado). Los commits y el push se realizarán manualmente por el usuario; no se efectuaron operaciones de publicación ni commits.

Subir código, `model/model.pkl`, metadatos y evidencias. `.gitignore` excluye el entorno y CSV. **E8 pendiente de la acción manual del equipo:** registrar y publicar sus commits reales. E1–E7 cuentan con los archivos y evidencia local indicados arriba.
