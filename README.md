# MPI Video Analyzer

Analizador distribuido de video para detectar y contar personas por frame usando YOLO11 en formato ONNX. El proyecto combina paralelismo de grano grueso con MPI y aceleracion GPU con CUDA u OpenCL.

El caso de uso principal es procesar videos de vigilancia o rescate, repartir el trabajo entre procesos MPI y generar resultados reproducibles:

- `output/frame_results.csv`: metricas por frame.
- `output/summary.json`: resumen global, balanceo de carga y tiempos.
- `output/annotated_people.mp4` o `.avi`: video anotado con cajas y conteo visual.

## Contenido del proyecto

```text
.
|-- CMakeLists.txt
|-- README.md
|-- arquitectura_paralela.md
|-- dataset/                 # videos de prueba
|-- docs/diagrams/           # diagramas PlantUML y SVG generados
|-- include/rescue/          # cabeceras publicas del proyecto
|-- models/                  # modelos YOLO11 exportados a ONNX
|-- scripts/
|   |-- benchmark_parallel.sh
|   `-- launch_cluster.py
`-- src/                     # implementacion C++, CUDA y OpenCL
```

## Arquitectura

El binario es SPMD: todos los procesos ejecutan `rescue_video_analyzer`, pero el `rank 0` actua como maestro y los demas ranks como workers.

![Arquitectura general](docs/diagrams/architecture.svg)

El maestro lee el video, crea lotes de frames y los envia por MPI. Cada worker recibe un lote, prepara los frames para YOLO, ejecuta inferencia y devuelve al maestro un `FramePacket` por frame. El maestro ordena los resultados por `frame_index`, ejecuta el tracker solo para visualizacion y escribe las salidas.

## Flujo MPI

El scheduler puede ser dinamico o estatico. El modo recomendado es `dynamic`, porque permite que los workers rapidos reciban mas lotes sin esperar a los lentos.

![Secuencia MPI dinamica](docs/diagrams/mpi_sequence.svg)

Schedulers disponibles:

- `dynamic`: reparte un nuevo lote al worker que termina antes.
- `static-contiguous`: asigna bloques contiguos de frames a cada worker.
- `static-round-robin`: reparte lotes por turnos entre workers.

## Backends GPU

![Backends de inferencia](docs/diagrams/backends.svg)

Ruta CUDA principal:

- Preprocesado con kernels CUDA en `src/gpu_preprocess.cu`.
- Letterbox, normalizacion y empaquetado `NCHW` en GPU.
- Inferencia con ONNX Runtime CUDA en `src/yolo_detector_cuda.cpp`.
- Carga dinamica de `libonnxruntime` desde `.venv` o desde `RESCUE_ORT_LIBRARY`.

Ruta OpenCL alternativa:

- Preprocesado con OpenCV `UMat` en `src/gpu_preprocess_opencl.cpp`.
- Inferencia con OpenCV DNN y target OpenCL en `src/yolo_detector_opencl.cpp`.
- Si OpenCL no esta disponible en OpenCV, el backend puede caer a CPU para esa ruta.

## Despliegue

El proyecto puede ejecutarse en una sola maquina o en un cluster MPI. En cluster, las rutas de `dataset/` y `models/` deben existir en todos los nodos o estar en un filesystem compartido.

![Despliegue](docs/diagrams/deployment.svg)

El script `scripts/launch_cluster.py` genera un `hosts_cluster.txt` y lanza `mpiexec` con tantos slots como GPUs detecte. Para ejecuciones controladas y benchmarks, suele ser mas claro usar `mpirun` manualmente.

## Datos y salidas

![Datos y salidas](docs/diagrams/data_outputs.svg)

Campos principales de `frame_results.csv`:

- `frame_index`: indice del frame en el video.
- `worker_rank`: rank MPI que proceso el frame.
- `gpu_device_id`: GPU visible usada por el worker.
- `timestamp_ms`: instante del frame segun el FPS del video.
- `people_count`: detecciones de clase persona transmitidas para ese frame.
- `mean_intensity` y `edge_density`: metricas auxiliares del preprocesado.
- `read_ms`, `preprocess_ms`, `detection_ms`, `processing_ms`: tiempos por etapa.

Campos principales de `summary.json`:

- Metadatos del video: resolucion, FPS, frames y duracion.
- Configuracion efectiva: scheduler, mundo MPI, workers y resolucion de procesamiento.
- Estadisticas de deteccion: media, maximo, acumulado de detecciones y acumulado visual por tracking.
- Balanceo de carga: frames por worker, ratio maximo/media y estadisticas por rank.
- Tiempos globales: `distributed_processing_ms`, `output_write_ms`, `total_wall_ms`.

## Dependencias

En Ubuntu/WSL necesitas:

- CMake 3.24 o superior.
- OpenCV 4 con modulos `core`, `imgproc`, `videoio`, `objdetect` y `dnn`.
- OpenMPI (`mpicxx`, `mpirun`).
- CUDA Toolkit (`nvcc`, `cudart`) para el backend CUDA.
- Java para PlantUML.
- PlantUML para regenerar diagramas.

Instalacion habitual con permisos sudo:

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  libopencv-dev \
  openmpi-bin \
  libopenmpi-dev \
  nvidia-cuda-toolkit \
  default-jre \
  plantuml \
  graphviz
```

En esta WSL no habia permisos sudo sin contrasena, asi que PlantUML quedo instalado para el usuario:

```bash
command -v plantuml
```

El wrapper apunta a:

```text
~/.local/share/plantuml/plantuml.jar
```

Si `~/.local/bin` no esta en tu `PATH`, puedes anadirlo con:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Los diagramas de este README usan `!pragma layout smetana`, por lo que se pueden renderizar sin Graphviz.

## ONNX Runtime CUDA

El proyecto carga ONNX Runtime de forma dinamica. La ruta esperada por defecto esta dentro de `.venv`:

```text
.venv/lib/python3.12/site-packages/onnxruntime/capi/
.venv/lib64/python3.12/site-packages/onnxruntime/capi/
```

El detector CUDA tambien contempla rutas de Python 3.11 para localizar `libonnxruntime`, pero el bootstrap inicial de `LD_LIBRARY_PATH` esta preparado principalmente para Python 3.12. Si usas otra version de Python o una instalacion externa, exporta:

```bash
export RESCUE_ORT_LIBRARY=/ruta/a/libonnxruntime.so
```

En WSL, si existe `/usr/lib/wsl/lib`, el binario la anade automaticamente antes de relanzarse para encontrar las librerias del driver NVIDIA.

## Modelos incluidos

El modelo por defecto es:

```text
models/yolo11s_1088.onnx
```

Variantes incluidas:

- `models/yolo11n.onnx`: modelo ligero.
- `models/yolo11s.onnx`: variante legacy a `640x640`.
- `models/yolo11s_960.onnx`: mas resolucion espacial que `640`.
- `models/yolo11s_1088.onnx`: valor por defecto, pensado para flujo 1080p.
- `models/yolo11m.onnx`: mas pesado.
- `models/yolo11l.onnx`: todavia mas pesado.

El parametro `--resize-width` debe coincidir con el tamano de exportacion del ONNX:

- `1088` para `yolo11s_1088.onnx`.
- `960` para `yolo11s_960.onnx`.
- `640` para `yolo11s.onnx`, `yolo11m.onnx` y `yolo11l.onnx`.

## Compilacion

Compilacion por defecto:

```bash
cmake -S . -B build
cmake --build build -j
```

Compilacion sin CUDA, usando solo la ruta OpenCL/CPU:

```bash
cmake -S . -B build -DRESCUE_ENABLE_CUDA=OFF
cmake --build build -j
```

La arquitectura CUDA por defecto es `89`, adecuada para la RTX 4070 Laptop usada en el proyecto. Puedes cambiarla asi:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Ejecucion rapida

Ejecucion recomendada en una maquina con una sola GPU NVIDIA:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --output-dir output \
  --batch-size 8 \
  --scheduler dynamic
```

Para evitar el coste de escribir video anotado durante benchmarks:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --output-dir output_bench \
  --batch-size 8 \
  --scheduler dynamic \
  --no-annotated-video
```

El binario tambien puede relanzar `mpirun` automaticamente si lo ejecutas sin launcher MPI. En modo `--gpu-backend cuda`, crea `1` maestro y `1` worker por GPU CUDA visible. En modo `auto`, si hay CUDA visible, tambien puede crear un worker adicional para la ruta OpenCL/CPU; para resultados de rendimiento mas limpios en una sola GPU, usa `mpirun -np 2` explicitamente.

## Ejemplos utiles

Usar el modelo de `960x960`:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --model models/yolo11s_960.onnx \
  --resize-width 960 \
  --output-dir output_yolo11s_960 \
  --batch-size 8 \
  --no-annotated-video
```

Forzar un dispositivo CUDA concreto:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset.mp4 \
  --output-dir output_gpu0 \
  --batch-size 8 \
  --gpu-backend cuda \
  --gpu-device 0
```

Usar resolucion nativa del video como resolucion de procesamiento:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --output-dir output_native \
  --processing-width 0 \
  --processing-height 0 \
  --batch-size 8
```

Ejecutar backend OpenCL:

```bash
mpirun -np 2 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --output-dir output_opencl \
  --gpu-backend opencl \
  --batch-size 8
```

## Parametros de linea de comandos

```text
--input <video>                         video de entrada, obligatorio
--output-dir <dir>                      directorio de salida, por defecto output
--batch-size <N>                        frames por lote MPI
--processing-width <px>                 ancho de procesamiento, 0 para nativo
--processing-height <px>                alto de procesamiento, 0 para nativo
--gpu-device auto|<id>                  seleccion de GPU visible
--gpu-backend auto|cuda|opencl          backend de worker
--resize-width <px>                     tamano cuadrado de entrada YOLO
--edge-threshold <N>                    umbral auxiliar de bordes
--model <path>                          modelo ONNX
--score-threshold <0..1>                confianza minima
--nms-threshold <0..1>                  umbral de NMS
--top-k <N>                             maximo de candidatos antes de NMS
--scheduler dynamic|static-contiguous|static-round-robin
--no-annotated-video                    no genera video anotado
```

Aliases antiguos aceptados:

```text
--yolo-model
--yolo-conf
--yolo-nms
--yolo-top-k
```

## Benchmarks

Script para probar varias combinaciones de mundo MPI y scheduler:

```bash
MPI_WORLD_SIZES="2 3 4" \
SCHEDULERS="dynamic static-contiguous static-round-robin" \
./scripts/benchmark_parallel.sh dataset/videoset.mp4 parallel_benchmarks
```

El script escribe una tabla TSV por stdout y guarda cada ejecucion en:

```text
parallel_benchmarks/np<N>_<scheduler>/
```

En una maquina con una sola GPU NVIDIA, no conviene interpretar `np=3` o `np=4` como escalado multi-GPU puro: varios workers pueden compartir el mismo dispositivo o caer a OpenCL/CPU. Para una comparacion limpia de inferencia CUDA en una GPU, usa `np=2`.

Prueba local reciente sobre `dataset/videoset2.mp4`, `mpirun -np 2`, `--batch-size 8`, `--no-annotated-video`:

```json
{
  "average_people_per_frame": 29.863,
  "average_preprocess_ms": 3.841,
  "average_detection_ms": 16.158,
  "average_processing_ms": 20.090,
  "total_wall_ms": 6399.037
}
```

## Cluster MPI

Lanzamiento automatico:

```bash
./scripts/launch_cluster.py \
  --ips localhost,192.168.1.55 \
  --args "--input dataset/videoset2.mp4 --output-dir output_cluster --batch-size 8 --scheduler dynamic"
```

Lanzamiento manual con hostfile:

```bash
mpirun --hostfile hosts_cluster.txt -np 4 ./build/rescue_video_analyzer \
  --input dataset/videoset2.mp4 \
  --model models/yolo11s_1088.onnx \
  --output-dir output_cluster \
  --batch-size 8 \
  --scheduler dynamic
```

Requisitos practicos para cluster:

- La misma ruta de ejecutable debe existir en los nodos.
- `dataset/` y `models/` deben estar accesibles desde todos los nodos.
- Las variables de entorno de CUDA/ONNX Runtime deben resolverse en cada nodo.
- SSH sin interaccion debe estar configurado para OpenMPI.

## Diagramas PlantUML

Los fuentes estan en:

```text
docs/diagrams/*.puml
```

Renderizar todos los diagramas:

```bash
plantuml -tsvg docs/diagrams/*.puml
```

Si el comando `plantuml` no esta en `PATH`, usa:

```bash
~/.local/bin/plantuml -tsvg docs/diagrams/*.puml
```

Los SVG generados se guardan junto a los `.puml` y son los que se muestran en este README.

## Troubleshooting

`Error: The --input argument is required.`

Pasa siempre `--input <video>`.

`Model file does not exist`

Comprueba la ruta de `--model`. El valor por defecto es `models/yolo11s_1088.onnx`.

`No visible CUDA devices were found`

Comprueba `nvidia-smi` dentro de WSL/Ubuntu. En WSL tambien deben estar visibles las librerias en `/usr/lib/wsl/lib`.

`Unable to locate libonnxruntime`

Activa la `.venv`, instala `onnxruntime-gpu` o exporta `RESCUE_ORT_LIBRARY`.

Rendimiento peor con mas workers

En una sola GPU, mas workers no significa mas GPUs. Puede haber contencion por el mismo dispositivo o mezcla con OpenCL/CPU. Usa `mpirun -np 2` para medir la ruta CUDA principal.

PlantUML avisa de que falta `dot`

Los diagramas del repo usan `smetana`, asi que se renderizan sin Graphviz. Si quieres soporte completo para cualquier diagrama PlantUML, instala Graphviz con sudo:

```bash
sudo apt-get install -y graphviz
```

## Estado actual comprobado

Comprobado en esta maquina:

- `cmake --build build -j` termina correctamente.
- `mpirun -np 2 ./build/rescue_video_analyzer --input dataset/videoset2.mp4 --output-dir /tmp/rescue_project_check --batch-size 8 --no-annotated-video` termina correctamente.
- PlantUML de usuario renderiza todos los diagramas SVG de `docs/diagrams/`.
