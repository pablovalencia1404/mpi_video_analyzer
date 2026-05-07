# Rescue Video Analyzer

Proyecto de análisis de video con arquitectura `MPI + CUDA` para detección de personas por frame usando YOLO11 exportado a ONNX e inferencia con ONNX Runtime CUDA.

## Qué implementa

- `rank 0` actúa como maestro: lee el vídeo una sola vez y reparte lotes de fotogramas serializados.
- Los workers MPI reciben esos fotogramas por MPI y solo se encargan de preprocesar e inferir.
- Antes de inferencia, cada frame se normaliza a una resolución de procesamiento. Por defecto el flujo trabaja a `1920x1080`; si quieres conservar la resolución nativa del vídeo puedes pasar `--processing-width 0 --processing-height 0`.
- Cada worker usa CUDA para hacer `letterbox`, normalización y empaquetado `NCHW` del tensor del modelo, y esa memoria CUDA entra directamente en ONNX Runtime CUDA.
- Por defecto, cada worker se liga automáticamente a una GPU visible distinta de su nodo usando su `local worker rank`, lo que permite ejecutar el proyecto en nodos multi-GPU o en una granja de GPUs con MPI.
- El scheduler MPI es configurable: `dynamic`, `static-contiguous` o `static-round-robin`.
- Si ONNX Runtime CUDA no está disponible, el worker falla y el proceso termina.
- El conteo de personas sale por defecto del modelo `yolo11s_1088.onnx`.
- El maestro agrega resultados ordenados, mantiene el tracker solo para anotación visual y exporta:
  - `output/annotated_people.mp4` o `output/annotated_people.avi`
  - `output/frame_results.csv`
  - `output/summary.json`
- El video anotado incluye overlay con:
  - `Personas actuales`: tracks visibles en el frame en curso.
  - `Acumulado real`: personas únicas estimadas por IDs de tracking.

## Dependencias

Necesitas estas herramientas instaladas en Linux:

- `cmake`
- `OpenCV 4` (incluyendo módulo `dnn`)
- `OpenMPI` (`mpicxx`, `mpirun`)
- `CUDA Toolkit` (`nvcc`, `cudart`)
- `wget` o `curl` para descargar el modelo YOLO11

En Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev openmpi-bin libopenmpi-dev nvidia-cuda-toolkit wget
```

## Backend de inferencia (ONNX Runtime CUDA)

La inferencia ya no depende de OpenCV DNN CUDA. El proyecto usa `onnxruntime-gpu` dentro de la `.venv` y el binario ajusta automáticamente `LD_LIBRARY_PATH` al arrancar para encontrar las bibliotecas CUDA/cuDNN empaquetadas con esa venv.

En WSL sigues necesitando los drivers NVIDIA visibles dentro de WSL (`nvidia-smi`). Cuando la ruta existe, el binario añade `/usr/lib/wsl/lib` automáticamente antes de relanzarse, así que no hace falta exportarla a mano para `rescue_video_analyzer`.

## Entorno exacto (probado)

- La carga de bibliotecas busca primero en la `.venv` usando rutas de Python `3.12` y `3.11` (ORT + cu13 + cuDNN). Si tu venv usa otro minor, exporta `RESCUE_ORT_LIBRARY`.
- El binario pre-carga libs CUDA/cuDNN empaquetadas en `nvidia/cu13` y `nvidia/cudnn` para evitar fallos de símbolos.
- Si existe `/usr/lib/wsl/lib`, se añade automáticamente al `LD_LIBRARY_PATH`.
- La compilación CUDA usa `CMAKE_CUDA_ARCHITECTURES=89` por defecto (puedes sobrescribirlo al invocar CMake).
## Modelo YOLO11

El binario busca por defecto el modelo en `models/yolo11s_1088.onnx`.

Los vídeos de ejemplo del proyecto están en `dataset/`.

Importante: la resolución del vídeo anotado y la resolución de inferencia no son la misma cosa.

- El vídeo anotado sale a la resolución de procesamiento, que por defecto ahora es `1920x1080`.
- `--resize-width` controla el tamaño cuadrado de entrada del detector YOLO y debe coincidir con el ONNX cargado.
- El modelo por defecto `models/yolo11s_1088.onnx` está exportado a `1088x1088`, no a `1080x1080`, porque YOLO trabaja con tamaños múltiplo de `32`.

El proyecto ya incluye varias variantes para que puedas comparar:

- `models/yolo11n.onnx`: la más ligera.
- `models/yolo11s.onnx`: variante legacy a `640x640`.
- `models/yolo11s_960.onnx`: variante exportada a `960x960` para más detalle espacial en inferencia.
- `models/yolo11s_1088.onnx`: recomendada por defecto para el flujo `1080p`.
- `models/yolo11m.onnx`: más pesada y más lenta.
- `models/yolo11l.onnx`: todavía más pesada; en esta GPU no compensa.

En la RTX 4070 Laptop del proyecto, sobre `dataset/videoset.mp4` con `1` worker GPU, la comparación quedó así:

- `yolo11n`: `average_detection_ms=6.043`, `average_people_per_frame=21.947`
- `yolo11s`: `average_detection_ms=6.724`, `average_people_per_frame=25.188`
- `yolo11m`: `average_detection_ms=10.825`, `average_people_per_frame=18.217`
- `yolo11l`: `average_detection_ms=13.964`, `average_people_per_frame=15.689`

Por eso `yolo11s` es la opción elegida: visualmente recupera personas lejanas que `n` pierde y lo hace con un coste muy pequeño frente a `n`, mientras que `m` y `l` penalizan bastante más la latencia y, en este vídeo, además detectan menos personas con el mismo umbral.

Si quieres más resolución espacial en inferencia, puedes usar la variante `960`:

```bash
./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --model models/yolo11s_960.onnx \
  --resize-width 960 \
  --output-dir output_yolo11s_960 \
  --batch-size 8 \
  --no-annotated-video
```

En una prueba rápida sobre `dataset/videoset2.mp4`, esta variante subió de `average_people_per_frame=22.013` a `27.791`, a cambio de pasar de `average_detection_ms=7.791` a `12.812`.

Si quieres cambiar de modelo en una ejecución concreta:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset.mp4 \
  --output-dir output_yolo11m \
  --batch-size 8 \
  --resize-width 640 \
  --model models/yolo11m.onnx
```

Si quieres exportar nuevas variantes manualmente:

```bash
.venv/bin/python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11s.pt").export(format="onnx", imgsz=1088, opset=12, simplify=True)
PY
```

## Compilación

```bash
cmake -S . -B build
cmake --build build -j
```

## Memoria en LaTeX

La memoria técnica completa está en `memoria_proyecto.tex`.

Para compilarla:

```bash
pdflatex memoria_proyecto.tex
pdflatex memoria_proyecto.tex
```

Si usas TinyTeX local de usuario, puedes invocar:

```bash
~/.TinyTeX/bin/x86_64-linux/pdflatex memoria_proyecto.tex
~/.TinyTeX/bin/x86_64-linux/pdflatex memoria_proyecto.tex
```

## Ejecución

Si ejecutas el binario directamente, el programa lanza por defecto `1` maestro + `1` worker por GPU NVIDIA visible.

Configuración base recomendada en una máquina con una sola GPU: `1` maestro + `1` worker (`mpirun -np 2`) para minimizar contención. Si quieres estudiar escalado por CPU, puedes subir el número de workers, pero varios procesos compartirán la misma GPU y el rendimiento puede empeorar.

Ejemplo recomendado:

```bash
./build/rescue_video_analyzer \
  --input dataset/videoset.mp4 \
  --output-dir output_videoset \
  --batch-size 8 \
  --scheduler dynamic
```

En una granja de GPUs, el comportamiento por defecto ya reparte workers entre las GPUs visibles de cada nodo. Para que eso funcione bien, `--input` y `--model` deben apuntar a rutas accesibles desde todos los nodos (por ejemplo, un filesystem compartido).

Para automatizar el despliegue en red sin tener que escribir el archivo hostfile a mano, utiliza el orquestador automático de Python:
```bash
./scripts/launch_cluster.py \
  --ips localhost,192.168.1.55 \
  --args "--input dataset/videoset2.mp4 --output-dir output_videoset2 --batch-size 8 --scheduler dynamic"
```

Si prefieres controlar tú el tamaño del mundo MPI, puedes seguir usando `mpirun` manualmente.

Si necesitas forzar un dispositivo concreto dentro del conjunto visible del proceso:

```bash
mpirun -np 3 ./build/rescue_video_analyzer \
  --input dataset/videoset.mp4 \
  --output-dir output_fixed_gpu \
  --batch-size 8 \
  --gpu-device 0
```

El maestro también genera un video anotado con rectángulos verdes sobre las personas detectadas.

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 3 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --output-dir output_videoset2 \
  --batch-size 8 \
  --scheduler dynamic
```

Si quieres estudiar arquitecturas de planificación sin pagar el coste del vídeo anotado:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 3 ./build/rescue_video_analyzer \
  --input dataset/videoset.mp4 \
  --output-dir output_static_rr \
  --batch-size 8 \
  --scheduler static-round-robin \
  --no-annotated-video
```

Y si quieres lanzar varias combinaciones de workers y scheduler:

```bash
MPI_WORLD_SIZES="2 3 4" SCHEDULERS="dynamic static-contiguous static-round-robin" \
  ./scripts/benchmark_parallel.sh dataset/videoset.mp4 parallel_benchmarks
```

## Parámetros útiles

- `--model <path>`
- `--score-threshold <0..1>`
- `--nms-threshold <0..1>`
- `--top-k <n>`
- `--scheduler dynamic|static-contiguous|static-round-robin`
- `--gpu-device auto|<id>` (por defecto `auto`, reparte por `local worker rank`)
- `--resize-width <px>` (debe coincidir con el tamaño de exportación del ONNX cargado: `1088` para `models/yolo11s_1088.onnx`, `960` para `models/yolo11s_960.onnx`, `640` para `models/yolo11s.onnx`, `models/yolo11m.onnx` y `models/yolo11l.onnx`)
- `--processing-width <px>` y `--processing-height <px>` (por defecto `1920x1080`; usa ambos a `0` si quieres la resolución nativa del vídeo)
- `--no-annotated-video`

Aliases antiguos aceptados:

- `--yolo-model <path>`
- `--yolo-conf <0..1>`
- `--yolo-nms <0..1>`
- `--yolo-top-k <n>`

## Notas

- El binario requiere al menos `2` ranks MPI.
- En WSL, `rescue_video_analyzer` ya añade `/usr/lib/wsl/lib` automáticamente si la ruta existe. Solo exporta esa ruta a mano si vas a ejecutar otra herramienta fuera del bootstrap del binario.
- Si ejecutas el binario sin `mpirun`, el programa relanza `mpirun` automáticamente con `1` maestro + `1` worker por GPU visible. Si fijas `--gpu-device N`, lanza solo `1` worker.
- En esta máquina, la comparación de workers con `yolo11s` sobre `dataset/videoset.mp4` quedó así: `mpirun -np 2` -> `5.73 s` de pared, `mpirun -np 3` -> `4.56 s`, `mpirun -np 4` -> `4.80 s`.
- El detector usa `CUDA preprocess + ONNX Runtime CUDA` como ruta principal y no hace fallback silencioso a CPU.
- El vídeo anotado sale por defecto a `1920x1080`; el tamaño de entrada del detector por defecto es `1088x1088` en `models/yolo11s_1088.onnx`.
- Si tu modelo ONNX no usa los nombres estándar, puedes forzar `RESCUE_ORT_INPUT_NAME` y `RESCUE_ORT_OUTPUT_NAME`.
- El acumulado en overlay y `summary.json` se estima con IDs únicos de tracking. El tracker ahora tolera mejor oclusiones cortas y cruces, pero sigue siendo una heurística visual y puede fallar en ocultaciones largas o reidentificación difícil.
- `output/frame_results.csv` incluye `worker_rank`, `gpu_device_id` y tiempos separados por etapa (`read_ms`, `preprocess_ms`, `detection_ms`, `processing_ms`).
- `output/frame_results.csv` reporta `people_count` hasta el tope de cajas transmitidas (`kMaxDetectionsPerFrame`, 128) para mantener consistencia con el overlay.
- `output/summary.json` incluye `scheduler`, `mpi_world_size`, `requested_workers` (alias `requested_cuda_workers`), `cuda_worker_count`, `load_balance`, `worker_stats`, `distributed_processing_ms`, `output_write_ms` y `total_wall_ms`.
- El modelo ONNX por defecto es `models/yolo11s_1088.onnx`, pero puedes pasar cualquier otra variante compatible con `--model`.
- Si necesitas más recall, usa `--score-threshold 0.2` y `--nms-threshold 0.45`.
