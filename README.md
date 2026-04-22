# Rescue Video Analyzer

Proyecto de análisis de video con arquitectura `MPI + CUDA` para detección de personas por frame usando YOLO11 exportado a ONNX e inferencia con ONNX Runtime CUDA.

## Qué implementa

- `rank 0` actúa como maestro: reparte rangos de frames, no payloads de imágenes.
- Los workers MPI abren el vídeo por su cuenta y leen el tramo que les toca.
- Cada worker usa CUDA para hacer `letterbox`, normalización y empaquetado `NCHW` del tensor del modelo, y esa memoria CUDA entra directamente en ONNX Runtime CUDA.
- El scheduler MPI es configurable: `dynamic`, `static-contiguous` o `static-round-robin`.
- Si ONNX Runtime CUDA no está disponible, el worker falla y el proceso termina.
- El conteo de personas sale por defecto del modelo `yolo11s.onnx`.
- El maestro agrega resultados ordenados, mantiene el tracker solo para anotación visual y exporta:
  - `output/annotated_people.mp4` o `output/annotated_people.avi`
  - `output/frame_results.csv`
  - `output/summary.json`

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

En WSL sigues necesitando los drivers NVIDIA visibles dentro de WSL (`nvidia-smi`) y conviene exportar `LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}` como en el ejemplo de abajo.
## Modelo YOLO11

El binario busca por defecto el modelo en `models/yolo11s.onnx`.

El modelo incluido en este flujo tiene entrada fija `640x640`. Por eso la ejecución recomendada usa `--resize-width 640`; si pasas otro valor, el worker lo ignora para la ruta del detector y deja constancia en logs.

El proyecto ya incluye cuatro variantes para que puedas comparar:

- `models/yolo11n.onnx`: la más ligera.
- `models/yolo11s.onnx`: recomendada por defecto.
- `models/yolo11m.onnx`: más pesada y más lenta.
- `models/yolo11l.onnx`: todavía más pesada; en esta GPU no compensa.

En la RTX 4070 Laptop del proyecto, sobre `videoset.mp4` con `1` worker GPU, la comparación quedó así:

- `yolo11n`: `average_detection_ms=6.043`, `average_people_per_frame=21.947`
- `yolo11s`: `average_detection_ms=6.724`, `average_people_per_frame=25.188`
- `yolo11m`: `average_detection_ms=10.825`, `average_people_per_frame=18.217`
- `yolo11l`: `average_detection_ms=13.964`, `average_people_per_frame=15.689`

Por eso `yolo11s` es la opción elegida: visualmente recupera personas lejanas que `n` pierde y lo hace con un coste muy pequeño frente a `n`, mientras que `m` y `l` penalizan bastante más la latencia y, en este vídeo, además detectan menos personas con el mismo umbral.

Si quieres cambiar de modelo en una ejecución concreta:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 2 ./build/rescue_video_analyzer \
  --input videoset.mp4 \
  --output-dir output_yolo11m \
  --batch-size 8 \
  --resize-width 640 \
  --model models/yolo11m.onnx
```

Si quieres exportar nuevas variantes manualmente:

```bash
.venv/bin/python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11l.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
PY
```

## Compilación

```bash
cmake -S . -B build
cmake --build build -j
```

## Ejecución

Configuración recomendada en esta RTX 4070 Laptop: `1` maestro + `2` workers GPU (`mpirun -np 3`). En las pruebas del proyecto con `yolo11s` y `videoset.mp4`, esa configuración bajó el tiempo total frente a `1` solo worker; subir a `3` workers ya no compensó.

Ejemplo recomendado:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 3 ./build/rescue_video_analyzer \
  --input videoset.mp4 \
  --output-dir output_videoset \
  --batch-size 8 \
  --resize-width 640 \
  --scheduler dynamic
```

El maestro también genera un video anotado con rectángulos verdes sobre las personas detectadas.

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 3 ./build/rescue_video_analyzer \
  --input videoset2.mp4 \
  --output-dir output_videoset2 \
  --batch-size 8 \
  --resize-width 640 \
  --scheduler dynamic
```

Si quieres estudiar arquitecturas de planificación sin pagar el coste del vídeo anotado:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} mpirun -np 3 ./build/rescue_video_analyzer \
  --input videoset.mp4 \
  --output-dir output_static_rr \
  --batch-size 8 \
  --resize-width 640 \
  --scheduler static-round-robin \
  --no-annotated-video
```

Y si quieres lanzar varias combinaciones de workers y scheduler:

```bash
MPI_WORLD_SIZES="2 3 4" SCHEDULERS="dynamic static-contiguous static-round-robin" \
  ./scripts/benchmark_parallel.sh videoset.mp4 parallel_benchmarks
```

## Parámetros útiles

- `--model <path>`
- `--score-threshold <0..1>`
- `--nms-threshold <0..1>`
- `--top-k <n>`
- `--scheduler dynamic|static-contiguous|static-round-robin`
- `--no-annotated-video`

Aliases antiguos aceptados:

- `--yolo-model <path>`
- `--yolo-conf <0..1>`
- `--yolo-nms <0..1>`
- `--yolo-top-k <n>`

## Notas

- El binario requiere al menos `2` ranks MPI.
- En WSL, exporta `LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}` antes de ejecutar `mpirun`. En las pruebas del proyecto es necesario para que ONNX Runtime CUDA vea la `libcuda` del host.
- En esta máquina, la comparación de workers con `yolo11s` sobre `videoset.mp4` quedó así: `mpirun -np 2` -> `5.73 s` de pared, `mpirun -np 3` -> `4.56 s`, `mpirun -np 4` -> `4.80 s`.
- El detector usa `CUDA preprocess + ONNX Runtime CUDA` como ruta principal y no hace fallback silencioso a CPU.
- El tracker actual solo estabiliza la anotación visual; no forma parte del conteo temporal.
- `output/frame_results.csv` incluye `worker_rank` y tiempos separados por etapa (`read_ms`, `preprocess_ms`, `detection_ms`, `processing_ms`).
- `output/summary.json` incluye `scheduler`, `mpi_world_size`, `requested_cuda_workers`, `cuda_worker_count`, `load_balance`, `worker_stats`, `distributed_processing_ms`, `output_write_ms` y `total_wall_ms`.
- El modelo ONNX por defecto es `models/yolo11s.onnx`, pero puedes pasar cualquier otra variante compatible con `--model`.
- Si necesitas más recall, usa `--score-threshold 0.2` y `--nms-threshold 0.45`.
