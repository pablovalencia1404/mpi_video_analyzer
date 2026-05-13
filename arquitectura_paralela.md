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

El modelo normal de ejecución es lanzar **un worker por cada GPU participante** en el conjunto de máquinas usado por MPI, además del rank maestro. Cada worker queda ligado a una GPU visible de su nodo y procesa lotes al ritmo que permita ese dispositivo. El sistema no presupone que todas las GPUs sean idénticas ni que todos los nodos tengan el mismo rendimiento.

### Reparto estático (Round-Robin / Bloques contiguos)
Los modos estáticos (`static-contiguous` y `static-round-robin`) son útiles para comparar estrategias de planificación porque fijan de antemano qué lotes procesa cada worker. Su inconveniente es que, si las GPUs o nodos tienen rendimiento distinto, puede aparecer desequilibrio: algunos workers terminan su cola antes y quedan inactivos mientras otros siguen procesando los lotes asignados.

### Planificación Dinámica (Dynamic Load Balancing)
Para maximizar el *Throughput* (rendimiento global) y la utilización de hardware, se ha implementado un despachador maestro asíncrono usando `MPI_ANY_SOURCE`.
1. El Maestro envía un lote inicial a cada Worker disponible.
2. Cada Worker preprocesa e infiere sobre su GPU asignada.
3. Cuando cualquier Worker termina, devuelve sus resultados al Maestro.
4. Si quedan frames pendientes, el Maestro entrega otro lote a ese mismo Worker.

**Resultado:** El reparto se adapta a la capacidad efectiva de las GPUs participantes. Los dispositivos con más capacidad procesan más lotes porque quedan disponibles más veces, y los dispositivos con menos capacidad siguen aportando trabajo sin imponer una barrera global por lote. La ley de Amdahl se mitiga porque el cuello de botella secuencial (la lectura del vídeo por el maestro) es pequeño frente al coste de inferencia.

---

## 4. Topología de Red y Escalabilidad Horizontal

El sistema ha sido adaptado para permitir escalabilidad horizontal (añadir más nodos físicos a la red) en lugar de depender exclusivamente del escalamiento vertical (comprar GPUs más potentes).

- **Auto-descubrimiento y Mapeo Físico:** Cuando el orquestador inyecta procesos MPI en un nodo, la librería subyacente otorga un `local_rank` a los workers que comparten placa base. El diseño utiliza este identificador para ligar cada worker a una GPU visible distinta dentro del nodo siempre que sea posible, evitando la sobre-suscripción accidental del mismo dispositivo.
- **Granularidad de Lotes (Batching):** Dado que la transferencia de datos por la red (Gigabit Ethernet) tiene una latencia muy alta comparada con el bus PCI-e, el sistema empaqueta fotogramas en *Batches*. Enviar lotes grandes amortiza el *overhead* de los paquetes TCP/IP y satura los pipelines aritméticos de las GPUs (manteniendo los núcleos CUDA ocupados sin latencias de red intermedias).

## 5. Resumen de Aportaciones al Rendimiento

1. **Paralelismo de Datos (MIMD + SIMT):** Partición espacial del vídeo en lotes distribuidos a nodos, e inferencia vectorial sobre redes neuronales dentro de la GPU.
2. **Eliminación de la penalización H2D/D2H:** Uso de CUDA C++ nativo para el preprocesado discreto y Memoria Unificada (Zero-Copy) para aceleradores integrados.
3. **Resiliencia Heterogénea:** Balanceo de carga dinámico que adapta el flujo de trabajo a las capacidades asimétricas del hardware subyacente, impidiendo *deadlocks* o desbalances de inactividad de recursos (Resource Starvation).
