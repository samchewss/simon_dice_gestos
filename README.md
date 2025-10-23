# Simon Dice con Gestos de Mano (MediaPipe + OpenCV)

---

## 🎮 Descripción general
Este proyecto implementa una versión digital del clásico juego **Simon Dice**, controlado completamente mediante **gestos de mano detectados en tiempo real** a través de la cámara.  
El jugador observa una **secuencia de gestos** y debe repetirla correctamente frente a la cámara para avanzar de ronda.  
Cada nueva ronda agrega un gesto más a la secuencia, aumentando la dificultad progresivamente.  

El sistema combina técnicas de **visión por computadora (OpenCV)** y un **modelo pre-entrenado (MediaPipe Hands)** que detecta y rastrea la posición de la mano sin necesidad de entrenamiento adicional o hardware especial.  

---

## 🧠 Modelo pre-entrenado utilizado: MediaPipe Hands
El reconocimiento de la mano se realiza mediante **MediaPipe Hands**, un modelo pre-entrenado de Google basado en aprendizaje profundo.  
Este modelo detecta y rastrea **21 puntos clave (landmarks)** de la mano en 3D —incluyendo muñeca, nudillos, articulaciones y puntas de los dedos—, y permite analizar la postura y orientación en tiempo real con baja latencia.  

### Características principales:
- **Modelo pre-entrenado**: no requiere entrenamiento adicional.  
- **Procesamiento en CPU**: funciona sin GPU, ideal para uso educativo.  
- **Salida en tiempo real**: coordenadas normalizadas de 21 landmarks por cuadro.  
- **Alta precisión y robustez** bajo condiciones normales de iluminación.  

En este proyecto, los landmarks extraídos por MediaPipe Hands se interpretan mediante **heurísticas geométricas** (distancias y ángulos entre articulaciones) para clasificar los siguientes **cuatro gestos principales**:
- ✋ **PALM** – mano abierta.  
- ✊ **FIST** – puño cerrado.  
- 👉 **POINT** – dedo índice extendido.  
- 👍 **THUMB_UP** – pulgar arriba.  

De esta manera, el modelo pre-entrenado se usa como base para la **detección de la mano**, y el algoritmo propio implementa la **clasificación de gestos** que controlan la lógica del juego.

---

## ⚙️ Instalación
Se recomienda utilizar **Python 3.11**, ya que la versión actual de MediaPipe es compatible hasta esa versión.

1. Crear un entorno virtual:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

Dependencias principales:
```
mediapipe
opencv-python
numpy
```

---

## ▶️ Ejecución
Con el entorno virtual activado, ejecutar:
```bash
python main.py
```

Al iniciar, se abrirá una ventana con la cámara y aparecerá el **menú principal** con fondo translúcido gris.  
Desde ahí se puede comenzar el juego presionando la tecla `i`.

---

## 🕹️ Controles
| Tecla | Acción |
|-------|---------|
| **i** | Iniciar el juego o volver al menú |
| **r** | Reiniciar la partida |
| **q** | Salir del programa |

---

## 🧩 Mecánica de juego
1. **Fase de menú:** muestra el título “SIMON DICE: GESTOS DE MANO”.  
   Presiona `i` para comenzar.  

2. **Fase SHOW:** el programa muestra una secuencia de gestos (uno por ronda).  
   Cada gesto aparece con texto centrado y color amarillo.  

3. **Fase INPUT:** el jugador debe reproducir la secuencia frente a la cámara.  
   - Se requieren varios **frames estables** con el mismo gesto para validar.  
   - Cada gesto tiene un **tiempo límite** para responder.  

4. Si el jugador acierta toda la secuencia, avanza de ronda.  
   Si falla o se agota el tiempo, aparece la pantalla **GAME OVER** con fondo rojo translúcido.  

---

## 🎨 Interfaz visual
- Fondo gris translúcido para el menú y mensajes.  
- Barra de estado con información de **ronda, puntuación y fase actual**.  
- Colores distintivos por fase:  
  - Amarillo: *SHOW*  
  - Verde: *INPUT*  
  - Rojo: *GAME OVER*  
- Textos centrados dinámicamente según el tamaño de la ventana.  

---

## 💡 Consejos de uso
- Coloca la cámara en un lugar fijo con buena **iluminación frontal**.  
- Evita fondos con colores similares a la piel.  
- Mantén la mano visible y centrada durante la detección.  
- Ajusta los parámetros `SHOW_GESTURE_MS`, `GESTURE_TIMEOUT_SEC` o `STABLE_FRAMES_REQUIRED` para adaptar la dificultad.  

---

## 🧾 Estructura del proyecto
```
simon_dice_gestos/
├── main.py             # Código principal del juego
├── requirements.txt    # Dependencias necesarias
└── README.md           # Documentación
```

---

## 👩‍💻 Créditos
**Equipo de desarrollo:**  
- Samantha Elizabeth Chew Arenas  
- Alonso Pérez Medrano  

**Materia:** Robótica Industrial  
**Institución:** Universidad Panamericana  
