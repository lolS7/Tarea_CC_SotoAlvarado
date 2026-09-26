# Clasificación de éxito de canciones

**Servicio público HTTPS:** [canciones-api en Google Cloud Run](https://canciones-api-412738758691.southamerica-west1.run.app).

**Probar la API:** [Swagger UI /docs](https://canciones-api-412738758691.southamerica-west1.run.app/docs) · **Estado:** [/health](https://canciones-api-412738758691.southamerica-west1.run.app/health).

Verificación externa: `/health`, `/docs`, `/openapi.json` y `POST /prediccion/` respondieron **HTTP 200 sin autenticación**. Respuestas registradas en [docs/cloud_test_results.json](docs/cloud_test_results.json). La raíz `/` no tiene endpoint y puede responder 404; usar los enlaces anteriores.

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

El entrenamiento local se realizó con Python **3.12.14**. Usar las versiones fijadas para cargar el PKL, especialmente scikit-learn **1.5.2**. En Cloud Run se selecciona la imagen `python312` (Python 3.12, parche administrado por el proveedor); no se afirma que su parche coincida con el entorno local.

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

## Despliegue en la nube

Proveedor: **Google Cloud Run**, servicio `canciones-api`, proyecto `final-project-cc-2026`, región `southamerica-west1` (Santiago). Se despliega desde el código mediante Cloud Build y Buildpacks, que construyen el contenedor y lo almacenan en Artifact Registry. No se necesita un Dockerfile propio con este procedimiento. El servicio carga `model/model.pkl` al iniciar y no entrena ni necesita el CSV.

El modelo ocupa 1.330.647 bytes (aproximadamente 1,33 MB), menos de 100 MB. Usa componentes estándar de scikit-learn y rutas relativas al código. Cloud Run se configuró con 1 CPU, 1 GiB de memoria, concurrencia 8, mínimo 0 y máximo 1 instancia, puerto 8080 y acceso público sin autenticación. El proyecto requiere facturación habilitada; estos límites no constituyen un presupuesto máximo de gasto.

### Pasos de configuración

1. Crear o seleccionar un proyecto de Google Cloud, habilitar facturación y abrir Cloud Shell. La cuenta que despliega debe tener permisos de Cloud Run, habilitación de APIs y uso de la cuenta de servicio.
2. Configurar el proyecto y habilitar APIs. Los comandos siguientes son para **Bash en Cloud Shell**. Un evaluador que quiera crear su propio servicio debe reemplazar el ID de proyecto por el suyo.

```bash
export PROJECT_ID="final-project-cc-2026"
export REGION="southamerica-west1"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

export PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
export BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${BUILD_SA}" \
  --role="roles/run.builder"
```

3. Clonar el repositorio, que incluye el pipeline entrenado y las dependencias fijadas.

```bash
git clone https://github.com/lolS7/Tarea_CC_SotoAlvarado.git
cd Tarea_CC_SotoAlvarado
ls app/main.py model/model.pkl model/metadata.json requirements.txt
```

4. Construir y publicar el servicio usando explícitamente la imagen compatible con Python 3.12.

```bash
gcloud run deploy canciones-api \
  --source . \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --base-image=python312 \
  --no-automatic-updates \
  --build-service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
  --set-build-env-vars="GOOGLE_ENTRYPOINT=uvicorn app.main:app --host 0.0.0.0 --port 8080" \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 8 \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated
```

5. Abrir la URL devuelta seguida de `/docs` y comprobar `/health` y una predicción con los comandos de la sección siguiente. Para actualizar el servicio después de un cambio, ejecutar `git pull` y repetir el despliegue; no se configuró despliegue automático desde GitHub.

### Variables de entorno utilizadas

| Variable | Ámbito | Valor y función |
| --- | --- | --- |
| `PROJECT_ID` | Terminal Cloud Shell | Proyecto donde se crea el servicio |
| `REGION` | Terminal Cloud Shell | `southamerica-west1` |
| `PROJECT_NUMBER` | Terminal Cloud Shell | Número obtenido con `gcloud projects describe` |
| `BUILD_SA` | Terminal Cloud Shell | Cuenta de servicio que construye la imagen |
| `GOOGLE_ENTRYPOINT` | Construcción | `uvicorn app.main:app --host 0.0.0.0 --port 8080` |
| `PORT` | Ejecución, provista por Cloud Run | `8080`, alineado con `--port` y el comando de Uvicorn |

La aplicación no requiere claves, secretos ni otras variables propias. `GOOGLE_PYTHON_VERSION` se usó en los intentos fallidos y se eliminó en la configuración final. El runtime final se selecciona mediante `--base-image=python312`. `runtime.txt` documenta el intérprete local; la elección explícita de la imagen determina el entorno cloud.

### Problema encontrado y solución

Los primeros despliegues fallaron al instalar Python. Con `GOOGLE_PYTHON_VERSION=3.12.14` se recibió `MANIFEST_UNKNOWN`. Cambiarlo a `3.12.x` también falló: el registro mostró que el constructor automático `google-24` solo ofrecía versiones 3.13 y 3.14.

**Solución aplicada:** seleccionar `--base-image=python312`, que corresponde al entorno `google-22` compatible con Python 3.12, y eliminar `GOOGLE_PYTHON_VERSION` de las variables de construcción. Se mantuvieron las dependencias de `requirements.txt`, incluido `scikit-learn==1.5.2`. El siguiente despliegue finalizó correctamente y la revisión quedó sirviendo el 100 % del tráfico.

Referencias: [imágenes de ejecución compatibles](https://docs.cloud.google.com/run/docs/configuring/services/runtime-base-images) y [configuración del despliegue desde código](https://docs.cloud.google.com/sdk/gcloud/reference/run/deploy).

### Llamadas curl públicas y respuestas

Desde Bash, sin token ni credenciales:

```bash
curl -i 'https://canciones-api-412738758691.southamerica-west1.run.app/health'
```

Resultado verificado: **HTTP 200**, cuerpo:

```json
{"status":"ok","model_loaded":true}
```

Ejemplo de inferencia completo, sin depender de archivos locales:

```bash
curl -i -X POST 'https://canciones-api-412738758691.southamerica-west1.run.app/prediccion/' \
  -H 'Content-Type: application/json' \
  --data '{"artist":"Britney Spears","duration_ms":211160,"danceability":0.751,"loudness":-5.444,"key":1,"genre":"pop","energy":0.834,"tempo":95.053}'
```

Resultado verificado: **HTTP 200**, cuerpo:

```json
{"EsExito":true,"ProbabilidadExito":1.0}
```

Estos cuerpos se obtuvieron de solicitudes HTTP al servicio público. La fecha, los códigos de estado y el JSON enviado se conservan en [docs/cloud_test_results.json](docs/cloud_test_results.json). También se verificó que `/docs` entrega el HTML de Swagger UI y que `/openapi.json` documenta el endpoint de predicción. Para repetir la verificación y actualizar la evidencia desde la raíz del proyecto:

```bash
python tests/verify_cloud.py
```
