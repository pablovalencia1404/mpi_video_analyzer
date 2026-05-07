# Análisis de Arquitectura Paralela y Distribuida
**Proyecto: Rescue Video Analyzer**

Este documento analiza en profundidad las decisiones de diseño desde la perspectiva de la asignatura de **Arquitecturas Paralelas**. El sistema implementa una topología híbrida que combina paso de mensajes (MPI) para paralelismo de grano grueso (distribuido) y aceleración por hardware (CUDA/OpenCL) para paralelismo de grano fino (SIMT).

---

## 1. Taxonomía de Flynn y Modelo de Ejecución

El proyecto sigue el modelo **SPMD (Single Program, Multiple Data)**, una subcategoría práctica del modelo **MIMD (Multiple Instruction, Multiple Data)** de la taxonomía de Flynn.
- **Single Program:** Un único binario (`rescue_video_analyzer`) es inyectado y ejecutado en todos los nodos físicos (CPUs).
- **Multiple Data:** El "Master" distribuye particiones de los datos (lotes de fotogramas diferentes) a cada trabajador.

El paralelismo ocurre a dos niveles fundamentales:
1. **Nivel de Nodo (Grano Grueso):** Paralelismo de tareas distribuidas en red mediante procesos del sistema operativo independientes, orquestados por OpenMPI. No existe estado compartido; toda la sincronización se realiza mediante paso de mensajes (`MPI_Send`, `MPI_Recv`).
2. **Nivel de Dispositivo (Grano Fino):** Paralelismo de datos masivo utilizando las tarjetas gráficas (GPUs) mediante arquitectura **SIMT (Single Instruction, Multiple Threads)**.

---

## 2. Paradigmas de Memoria y Cuellos de Botella (Overhead)

Uno de los mayores retos en arquitecturas paralelas es el *overhead* de las comunicaciones y el movimiento de datos a través de la jerarquía de memoria (RAM vs VRAM). El sistema aborda esto mediante un diseño polimórfico adaptado a la topología de la GPU:

### 2.1. Memoria Discreta (NVIDIA CUDA)
En los nodos equipados con gráficas dedicadas, los datos sufren un cuello de botella en el bus **PCI-Express**.
- **Problema:** Copiar fotogramas desde la RAM del Host hacia la VRAM del Device es costoso.
- **Solución implementada:** Se ha programado un kernel customizado en CUDA (`gpu_preprocess.cu`). El frame crudo se envía a la VRAM una sola vez. Las transformaciones de píxeles (conversión BGR a RGB, redimensionado, padding y normalización de tensores NCHW) ocurren de forma masivamente paralela dentro del *Device*, evitando viajes de ida y vuelta al *Host*. Los Tensor Cores de la arquitectura Ada Lovelace (RTX 4070) ejecutan la red neuronal directamente sobre este bloque de memoria.

### 2.2. Memoria Unificada (OpenCL - Gráficas Integradas)
En nodos con gráficas integradas (ej. Intel Iris Xe), la RAM y la VRAM comparten el mismo silicio físicamente.
- **Problema:** Los frameworks tradicionales fuerzan la copia de datos asumiendo que la GPU está separada, lo que destruye el rendimiento por contención de bus.
- **Solución implementada (Zero-Copy):** Se ha diseñado el backend de OpenCL para utilizar las estructuras `cv::UMat` (Unified Matrix). Esto permite que la CPU y la GPU (OpenCL) accedan a los mismos punteros físicos de memoria. El pipeline de *OpenCV DNN* procesa el tensor sin ejecutar transferencias explícitas de memoria Host-to-Device (H2D), eliminando completamente el *overhead* de transferencia.

---

## 3. Planificación (Scheduling) y Balanceo de Carga

En un clúster heterogéneo (NVIDIA RTX 4070 Laptop vs Intel Graphics), existe una **asimetría computacional severa**. La NVIDIA puede procesar un frame en ~17 ms, mientras que la Intel necesita ~190 ms.

### Fracaso del reparto estático (Round-Robin / Bloques contiguos)
Si el Maestro utilizara una planificación estática (asignar el lote 1 a la GPU A, el lote 2 a la GPU B), sufriríamos el peor escenario del paralelismo: **Sincronización de barrera implícita**.
La ejecución global de la aplicación se vería frenada por la GPU más lenta (el eslabón más débil). La GPU rápida terminaría su trabajo inmediatamente y pasaría el 90% del tiempo de ejecución en estado inactivo (*Idle*) esperando a que la GPU lenta terminase para recibir el siguiente bloque de datos.

### Éxito de la Planificación Dinámica (Dynamic Load Balancing)
Para maximizar el *Throughput* (Rendimiento global) y la Utilización de Hardware, se ha implementado un despachador maestro asíncrono usando `MPI_ANY_SOURCE`.
1. El Maestro inyecta un *job* inicial en la cola de todos los Workers.
2. El Maestro bloquea su ejecución esperando una respuesta de **cualquier** Worker.
3. Tan pronto como el Worker más rápido (NVIDIA) termina su ráfaga, devuelve los resultados. El Maestro le inyecta instantáneamente un nuevo lote.
4. El Worker lento (Intel) retiene su carga sin bloquear al sistema general.
**Resultado:** Se logra un balanceo de carga empírico perfecto. La tarjeta NVIDIA procesa asincrónicamente el 90% del vídeo, mientras que la Intel ayuda asumiendo el 10% restante. La ley de Amdahl se ve mitigada porque el cuello de botella secuencial (la lectura del vídeo por el maestro) es mínimo frente al coste de inferencia.

---

## 4. Topología de Red y Escalabilidad Horizontal

El sistema ha sido adaptado para permitir escalabilidad horizontal (añadir más nodos físicos a la red) en lugar de depender exclusivamente del escalamiento vertical (comprar GPUs más potentes).

- **Auto-descubrimiento y Mapeo Físico:** Cuando el orquestador inyecta procesos MPI en un nodo, la librería subyacente otorga un `local_rank` a los workers que comparten placa base. Nuestro diseño utiliza este identificador para ligar a cada hilo de ejecución a un motor de silicio distinto (Worker 0 -> `/dev/nvmeX` NVIDIA, Worker 1 -> `/dev/dri` OpenCL). Evitando condiciones de carrera y la sobre-suscripción del planificador de hardware de las GPUs.
- **Granularidad de Lotes (Batching):** Dado que la transferencia de datos por la red (Gigabit Ethernet) tiene una latencia muy alta comparada con el bus PCI-e, el sistema empaqueta fotogramas en *Batches*. Enviar lotes grandes amortiza el *overhead* de los paquetes TCP/IP y satura los pipelines aritméticos de las GPUs (manteniendo los núcleos CUDA ocupados sin latencias de red intermedias).

## 5. Resumen de Aportaciones al Rendimiento

1. **Paralelismo de Datos (MIMD + SIMT):** Partición espacial del vídeo en lotes distribuidos a nodos, e inferencia vectorial sobre redes neuronales dentro de la GPU.
2. **Eliminación de la penalización H2D/D2H:** Uso de CUDA C++ nativo para el preprocesado discreto y Memoria Unificada (Zero-Copy) para aceleradores integrados.
3. **Resiliencia Heterogénea:** Balanceo de carga dinámico que adapta el flujo de trabajo a las capacidades asimétricas del hardware subyacente, impidiendo *deadlocks* o desbalances de inactividad de recursos (Resource Starvation).
