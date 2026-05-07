Corrige esto, en este orden:

## 1. Haz que la ruta principal de detección sea realmente GPU

Ahora mismo CUDA procesa el frame, pero la detección va por otra ruta y puede caer a CPU.

### Qué cambiar

En `src/yolo_detector.cpp`:

* elimina o desactiva el fallback automático a `OpenCvCpu` para las ejecuciones del proyecto;
* si ONNX Runtime con CUDA no está disponible, falla con error y termina;
* registra en log qué backend está usando cada worker.

### Qué conseguir

Que puedas afirmar de verdad:

* **MPI reparte**
* **CUDA/ORT GPU detecta**
* **el conteo sale de la ruta GPU**

Si no haces esto, tu arquitectura real no es MPI + CUDA, sino MPI + “a veces GPU, a veces CPU”.

## 2. Integra CUDA en el pipeline útil, no solo en métricas auxiliares

Ahora `preprocess_frame_cuda()` calcula:

* gris
* resize
* intensidad media
* edge density

pero `detector.detect(frames[i])` sigue usando el frame original.

### Qué cambiar

En `src/mpi_runtime.cpp`:

* deja de llamar a CUDA solo para sacar métricas;
* usa CUDA para una etapa que sí alimente la detección o el resultado final.

Tienes dos opciones:

### opción A

Mantener ONNX Runtime CUDA como detector principal y usar `preprocess_frame_cuda()` solo como **preprocesado real previo al detector**.

### opción B

Si no vas a conectar esa salida al detector, entonces baja el peso de ese módulo y no lo vendas como parte central de la detección. En ese caso, en la memoria debes decir que CUDA acelera:

* preprocesado,
* caracterización visual,
* y métricas por frame,
  pero no la detección completa.

### Qué te recomiendo

Haz esto:

* deja ONNX Runtime CUDA como ruta principal de inferencia;
* usa CUDA para preprocesado real del frame antes de la inferencia, o elimina del flujo principal las métricas CUDA si no aportan al núcleo.

## 3. No envíes frames completos por MPI

Este es el mayor problema arquitectónico.

Ahora el rank 0:

* lee el vídeo,
* decodifica,
* empaqueta todos los frames,
* y los manda como bytes a los workers.

Eso funciona, pero hace del maestro un cuello de botella.

### Qué cambiar

En `src/mpi_runtime.cpp` y `src/video_io.cpp`:

* deja de enviar el payload completo de los frames;
* envía solo el trabajo: rango de frames, índice inicial y número de frames;
* haz que cada worker abra el vídeo y lea sus propios frames.

### Qué necesitas para eso

Crear en `VideoReader` una forma de:

* posicionarte en un frame concreto;
* leer `N` frames desde ahí.

Por ejemplo, un método tipo:

* `read_batch_from(start_frame, batch_size, ...)`

El maestro puede usarlo para leer y serializar lotes; los workers no deberían abrir el vídeo.

### Qué conseguir

Pasar de esta arquitectura:

* maestro lee y serializa lotes de frames

a esta:

* maestro reparte lotes de frames
* workers reciben y procesan

Eso mejora mucho la coherencia arquitectónica.

## 4. Controla la competencia por la GPU compartida

Tienes una sola RTX 4070 Laptop. Si lanzas varios workers MPI usando todos CUDA al mismo tiempo, varios procesos pueden compartir la misma GPU y aumentar la contención.

### Qué cambiar

En la ejecución real del proyecto:

* usa **1 maestro + 1 worker** como configuración base oficial;
* si pruebas más workers, hazlo para analizar contención;
* no vendas muchos workers como escalado real si comparten la misma GPU.

### Qué añadir

En logs y memoria, deja claro:

* cuántos workers hay;
* si la GPU es compartida entre varios procesos;
* que el sistema reparte trabajo por núcleos de CPU, no por GPUs.

## 5. Haz que el tracking forme parte del resultado, no solo del vídeo anotado

Ahora el tracker solo se usa en `output_writer.cpp` para dibujar cajas más bonitas.

### Qué cambiar

Decide una de estas dos rutas:

### opción A

Si el objetivo es solo conteo por frame:

* elimina la idea de tracking de la lógica central;
* deja claro que el tracker es solo para estabilizar visualización.

### opción B

Si quieres vender “análisis temporal del vídeo”:

* mueve el tracking antes de generar el resultado final;
* usa el tracker para estabilizar detecciones y derivar el conteo temporal.

### Qué te recomiendo

Para no rehacer demasiado:

* deja el tracking solo para visualización,
* pero dilo explícitamente en la memoria.

No digas que el sistema hace seguimiento robusto si no es verdad.

## 6. Añade trazabilidad del backend y tiempos por etapa

Ahora mides `processing_ms`, pero no separas bien qué parte tarda qué.

### Qué cambiar

En `src/mpi_runtime.cpp`:

* mide por separado:

  * tiempo de preprocesado CUDA,
  * tiempo de detección,
  * tiempo total por frame,
  * opcionalmente tiempo de empaquetado/serialización si mantienes el diseño actual.

### Qué añadir

En los logs de cada worker:

* rank,
* backend detector,
* GPU activa o no,
* tiempo medio por lote,
* número de detecciones.

### Qué conseguir

Que luego puedas defender la arquitectura con datos de verdad.

## 7. Endurece el detector para clase persona y valida resultados

Asegúrate de que la detección realmente filtra bien la clase persona y no mezcla clases irrelevantes.

### Qué cambiar

En `src/yolo_detector.cpp`:

* verifica el filtro de clase;
* deja explícito qué `class_id` cuenta como persona;
* documenta el umbral de confianza;
* guarda el número de detecciones descartadas si quieres análisis más fino.

### Qué conseguir

Que `people_count` tenga una definición técnica clara.

## 8. Reduce el acoplamiento del bootstrap de librerías

Tu `main.cpp` hace bastante trabajo de entorno con `LD_LIBRARY_PATH` y `execv`, y ahora también añade `/usr/lib/wsl/lib` cuando existe para que CUDA arranque sin pasos manuales en WSL.

### Qué cambiar

Si funciona, no lo rompas ahora, pero:

* simplifica la carga de librerías si puedes;
* evita que la validez del proyecto dependa de una ruta `.venv` muy concreta;
* documenta exactamente el entorno necesario.

### Qué conseguir

Que la arquitectura no parezca dependiente de un hack de arranque.

## 9. Corrige la narrativa arquitectónica en la memoria

Aunque no toques mucho código, hay cosas que sí o sí debes corregir al describir el proyecto.

### No digas

* que CUDA hace toda la detección si no está garantizado;
* que MPI distribuye eficientemente el vídeo si el maestro manda frames completos;
* que el sistema hace tracking real para el conteo si solo lo usa al dibujar.

### Di

* que MPI implementa distribución maestro-trabajadores;
* que CUDA acelera preprocesado y, si está activo el backend correcto, la inferencia;
* que el tracking actual se usa para estabilizar la anotación visual;
* que la arquitectura actual está desplegada sobre nodo único con GPU compartida.

## 10. Qué corregiría archivo por archivo

### `src/yolo_detector.cpp`

* quitar fallback automático silencioso a CPU;
* registrar backend real;
* fallar si no hay backend GPU en modo experimental oficial;
* dejar clara la clase persona.

### `src/mpi_runtime.cpp`

* no enviar frames completos por MPI;
* enviar trabajo por índices/rangos;
* medir tiempos por etapa;
* integrar mejor CUDA con la ruta principal;
* limitar o controlar mejor uso de GPU por varios ranks.

### `src/video_io.cpp`

* añadir lectura por rango de frames;
* permitir que cada worker cargue su propio lote.

### `src/gpu_preprocess.cu`

* mantenerlo si realmente alimenta detección o salida útil;
* si no, reducirlo a módulo auxiliar y no venderlo como núcleo.

### `src/output_writer.cpp`

* dejar tracking como visualización, o moverlo al pipeline central si de verdad quieres análisis temporal.

### `src/main.cpp`

* mejorar robustez del entorno y documentar dependencias reales.

## Prioridad real si vas justo de tiempo

Si solo puedes arreglar tres cosas, arregla estas:

### 1.

**Forzar detección en GPU sin fallback silencioso**

### 2.

**Dejar de enviar frames completos por MPI**

### 3.

**Aclarar que el tracking actual solo afecta a la anotación visual**

Con esas tres, tu proyecto mejora bastante en coherencia arquitectónica.

## Resumen muy corto

Tienes que corregir esto:

* que la detección dependa realmente de GPU;
* que CUDA no esté desconectado del núcleo útil;
* que MPI reparta trabajo, no imágenes en bruto;
* que no vendas tracking como parte central si solo dibuja;
* y que el maestro no sea el gran cuello de botella.