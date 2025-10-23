# -*- coding: utf-8 -*-
"""
Simon Dice con Gestos de Mano (MediaPipe + OpenCV)
Versión mejorada con menú, fondo translúcido gris y texto centrado.
"""

import cv2
import time
import random
import numpy as np
import mediapipe as mp

GESTURES = ["PALM", "FIST", "POINT", "THUMBS_UP"]
WINDOW_NAME = "Simon Dice - Gestos (MediaPipe + OpenCV)"

SHOW_GESTURE_MS = 1500
SHOW_GAP_MS = 800
GESTURE_TIMEOUT_SEC = 6.0
STABLE_FRAMES_REQUIRED = 7

COLOR_SHOW = (255, 220, 100)
COLOR_INPUT = (150, 255, 150)
COLOR_GAMEOVER = (80, 80, 255)
COLOR_MENU = (255, 255, 255)

def put_text(img, text, org, scale=1.0, color=(255,255,255), thickness=2):
    x, y = org
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_overlay_box(frame, alpha=0.4, color=(40,40,40)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def draw_center_text(frame, text, y_ratio=0.5, scale=1.5, color=(255,255,255), thickness=3):
    h, w = frame.shape[:2]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (w - size[0]) // 2
    y = int(h * y_ratio)
    put_text(frame, text, (x, y), scale, color, thickness)

def draw_status_bar(frame, round_num, score, phase):
    h, w = frame.shape[:2]
    bar_h = int(0.11 * h)
    overlay = frame.copy()
    color = (60,60,60)
    if phase == "SHOW": color = COLOR_SHOW
    elif phase == "INPUT": color = COLOR_INPUT
    elif phase == "GAME_OVER": color = COLOR_GAMEOVER
    elif phase == "MENU": color = (100,100,100)
    cv2.rectangle(overlay, (0,0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    put_text(frame, f"Ronda: {round_num}", (10, 30), 0.8)
    put_text(frame, f"Puntuacion: {score}", (10, 60), 0.8)
    put_text(frame, f"Fase: {phase}", (10, 90), 0.8)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_checkmark(frame, center=None, radius=40, color=(80, 220, 80), thickness=10):
    """Dibuja un 'check' verde dentro de un círculo."""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, int(h * 0.5))
    # círculo
    cv2.circle(frame, center, radius, color, 4, cv2.LINE_AA)
    # palomita (dos líneas)
    p1 = (center[0] - int(radius*0.5), center[1] + int(radius*0.2))
    p2 = (center[0] - int(radius*0.15), center[1] + int(radius*0.55))
    p3 = (center[0] + int(radius*0.55), center[1] - int(radius*0.35))
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.line(frame, p2, p3, color, thickness, cv2.LINE_AA)

def finger_extended(landmarks, tip_id, pip_id, wrist_id=0, threshold=0.035):
    tip = landmarks[tip_id]; pip_ = landmarks[pip_id]; wrist = landmarks[wrist_id]
    d_tip = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_pip = np.hypot(pip_.x - wrist.x, pip_.y - wrist.y)
    return (d_tip - d_pip) > threshold

# IDs de landmarks
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 1

def vector(a, b):
    return np.array([b.x - a.x, b.y - a.y], dtype=np.float32)

def finger_extended(landmarks, tip_id, pip_id, wrist_id=WRIST, threshold=0.035):
    """
    Un dedo está extendido si tip->wrist es significativamente mayor que pip->wrist.
    threshold más alto = más estricto (menos falsos positivos).
    """
    tip = landmarks[tip_id]; pip_ = landmarks[pip_id]; wrist = landmarks[wrist_id]
    d_tip = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_pip = np.hypot(pip_.x - wrist.x, pip_.y - wrist.y)
    return (d_tip - d_pip) > threshold

def thumb_extended_with_direction(landmarks, handedness_label):
    """
    Pulgar extendido + apuntando predominantemente hacia arriba.
    Además chequea coherencia lateral (Left/Right).
    """
    tip = landmarks[THUMB_TIP]
    mcp = landmarks[THUMB_MCP]
    wrist = landmarks[WRIST]

    # ¿Extendido respecto a la muñeca?
    ext = finger_extended(landmarks, THUMB_TIP, 2, WRIST, threshold=0.02)

    # Dirección: y crece hacia abajo -> hacia arriba = componente vertical negativa dominante
    v = vector(wrist, tip)
    upwards = (-v[1]) > (abs(v[0]) * 0.35)   # 0.35 = tolerancia (sube si necesitas ser más estricto)

    # Coherencia lateral (pulgar apunta lateralmente correcto respecto a la mano)
    v_thumb = vector(mcp, tip)
    if handedness_label == "Right":
        lateral_ok = v_thumb[0] >= -0.03
    else:  # "Left"
        lateral_ok = v_thumb[0] <= 0.03

    return ext and upwards and lateral_ok

def classify_gesture(hand, handed="Right"):
    lm = hand.landmark

    # Threshold un poquito más estricto para los dedos (reduce falsos positivos)
    index  = finger_extended(lm, 8,  6, WRIST, threshold=0.04)
    middle = finger_extended(lm, 12, 10, WRIST, threshold=0.04)
    ring   = finger_extended(lm, 16, 14, WRIST, threshold=0.04)
    pinky  = finger_extended(lm, 20, 18, WRIST, threshold=0.04)
    others_ext = sum([index, middle, ring, pinky])

    # Chequeo de pulgar "arriba" robusto
    thumb_up = thumb_extended_with_direction(lm, handed)

    # --- Reglas ---
    # PALM: >=3 dedos (sin contar pulgar) realmente extendidos
    if others_ext >= 3:
        return "PALM"

    # FIST: ninguno extendido (o casi)
    if others_ext == 0 and not thumb_up:
        return "FIST"

    # POINT: solo índice extendido (pulgar indiferente)
    if index and not middle and not ring and not pinky:
        return "POINT"

    # THUMBS_UP: pulgar arriba y como mucho 1 dedo "medio extendido" por ruido
    if thumb_up and others_ext <= 1:
        return "THUMBS_UP"

    return "UNKNOWN"

class SimonSays:
    def __init__(self):
        self.sequence = []
        self.reset_game(full=True)

    def reset_game(self, full=False):
        if full or not self.sequence:
            self.sequence = [random.choice(GESTURES)]
        self.round = 1
        self.score = 0
        self.phase = "MENU"
        self.message = "Presiona 'i' para iniciar"
        self.show_index = 0
        self.input_index = 0
        self.stable_count = 0
        self.current_target = self.sequence[0]
        self.gesture_start_time = 0.0

    def start_game(self):
        self.phase = "SHOW"
        self.show_index = 0
        self.message = "Observa la secuencia..."

    def start_input(self):
        self.phase = "INPUT"
        self.input_index = 0
        self.current_target = self.sequence[self.input_index]
        self.stable_count = 0
        self.gesture_start_time = time.time()
        self.message = "Repite la secuencia"

    def advance_round(self):
        self.round = len(self.sequence)
        self.score += 10
        self.sequence.append(random.choice(GESTURES))
        self.start_game()

    def game_over(self):
        self.phase = "GAME_OVER"
        self.message = "¡Game Over! Presiona 'r' o 'i'."

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    hands = mp_hands.Hands(False, 1, 1, 0.5, 0.5)
    game = SimonSays()
    show_state = "SHOW_GESTURE"
    show_start = int(time.time()*1000)
    detected = "UNKNOWN"

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                hand_label = results.multi_handedness[0].classification[0].label
                detected = classify_gesture(hand, hand_label)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                       mp_styles.get_default_hand_landmarks_style(),
                                       mp_styles.get_default_hand_connections_style())
            else:
                detected = "UNKNOWN"

            # Fase MENU
            if game.phase == "MENU":
                draw_overlay_box(frame, 0.5, (40,40,40))
                # Título en dos líneas, escalado para no desbordar
                draw_center_text(frame, "SIMON DICE",     0.42, 2.0, COLOR_MENU, 4)
                draw_center_text(frame, "GESTOS DE MANO", 0.55, 1.4, COLOR_MENU, 3)
                draw_center_text(frame, "Presiona 'i' para iniciar", 0.78, 0.9, (220,255,220), 2)

            elif game.phase == "SHOW":
                now = int(time.time()*1000)
                elapsed = now - show_start
                if show_state == "SHOW_GESTURE":
                    target = game.sequence[game.show_index]
                    draw_center_text(frame, target, 0.45, 2.0, COLOR_SHOW, 4)
                    if elapsed >= SHOW_GESTURE_MS:
                        show_state = "GAP"; show_start = now
                else:
                    if elapsed >= SHOW_GAP_MS:
                        game.show_index += 1; show_state = "SHOW_GESTURE"; show_start = now
                        if game.show_index >= len(game.sequence):
                            game.start_input()

            elif game.phase == "INPUT":
                # Pequeño subtítulo con progreso (sin mostrar el gesto)
                draw_center_text(frame, f"Gesto {game.input_index+1} de {len(game.sequence)}", 
                                0.30, 1.0, COLOR_INPUT, 2)

                # Temporizador
                elapsed = time.time() - game.gesture_start_time
                rem = max(0, GESTURE_TIMEOUT_SEC - elapsed)
                put_text(frame, f"Tiempo restante: {rem:.1f}s", (10, int(frame.shape[0]*0.85)), 0.9, (255,255,255), 2)

                # Validación silenciosa
                target = game.current_target
                if detected == target:
                    game.stable_count += 1
                    # Muestra la palomita mientras va siendo estable
                    draw_checkmark(frame)
                else:
                    game.stable_count = 0

                # Si ya juntó N frames estables, avanza
                if game.stable_count >= STABLE_FRAMES_REQUIRED:
                    game.input_index += 1
                    if game.input_index >= len(game.sequence):
                        game.advance_round()
                        show_state = "SHOW_GESTURE"
                        show_start = int(time.time()*1000)
                    else:
                        game.current_target = game.sequence[game.input_index]
                        game.stable_count = 0
                        game.gesture_start_time = time.time()

                # Timeout -> Game Over
                if elapsed > GESTURE_TIMEOUT_SEC:
                    game.game_over()

            elif game.phase == "GAME_OVER":
                draw_overlay_box(frame, 0.5, (30,30,30))
                draw_center_text(frame, "GAME OVER", 0.45, 2.2, COLOR_GAMEOVER, 5)
                draw_center_text(frame, "Presiona 'r' para reiniciar o 'i' para menú", 0.75, 0.8, (255,255,255), 2)

            draw_status_bar(frame, len(game.sequence), game.score, game.phase)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r') and game.phase != "MENU":
                game.reset_game(full=True); game.start_game(); show_start = int(time.time()*1000)
            elif key == ord('i') and game.phase in ["MENU", "GAME_OVER"]:
                game.reset_game(full=True); game.start_game(); show_start = int(time.time()*1000)

        cap.release(); cv2.destroyAllWindows()
    except KeyboardInterrupt:
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
