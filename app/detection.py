import cv2
from collections import defaultdict, Counter
import math

COLOR_VALUES = {
    "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9, "silver": 10
}

COLOR_RANGES = {
    "black": ((0, 0, 0), (180, 255, 30)),
    "brown": ((10, 100, 20), (20, 255, 200)),
    "orange": ((10, 100, 100), (25, 255, 255)),
    "yellow": ((25, 100, 100), (35, 255, 255)),
    "green": ((35, 100, 100), (85, 255, 255)),
    "white": ((0, 0, 200), (180, 20, 255)),
    "silver": ((0, 0, 192), (180, 50, 255))
}


def calculate_distance_to_resistor(perceived_width, actual_width=0.01, focal_length=0.00367):
    distance = (actual_width * focal_length) / perceived_width
    return distance


def calculate_resistance(colors, color_values=COLOR_VALUES):
    if len(colors) < 4:
        return None

    first_digit = color_values[colors[0]]
    second_digit = color_values[colors[1]]
    multiplier = 10 ** color_values[colors[2]]
    resistance = (first_digit * 10 + second_digit) * multiplier

    tolerance_values = {
        "brown": 1, "red": 2, "green": 0.5, "blue": 0.25, "violet": 0.1,
        "gray": 0.05, "yellow": 5, "silver": 10
    }

    tolerance = tolerance_values.get(colors[3], 0)
    return resistance, tolerance


def is_parallel(resistor1, resistor2):
    x1, y1, x2, y2 = resistor1["bbox"]
    ox1, oy1, ox2, oy2 = resistor2["bbox"]

    center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
    center2_x, center2_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    distance = math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
    perceived_width = x2 - x1
    dist_to_camera = calculate_distance_to_resistor(perceived_width)
    distance_threshold = 50 * (1 + dist_to_camera)

    if distance < distance_threshold:
        if abs(y1 - oy1) < 10 and abs(y2 - oy2) < 10:
            return True
        if abs(x1 - ox1) < 10 and abs(x2 - ox2) < 10:
            return True
        if distance < 30:
            if abs(y1 - oy2) < 5 and abs(y2 - oy1) < 5:
                return True
            if abs(x1 - ox2) < 5 and abs(x2 - ox1) < 5:
                return True
    return False


def detect_resistors(frame, model, device):
    results = model.track(source=frame, conf=0.3, show_conf=False, persist=True, device=device)

    resistors = []
    color_detections = defaultdict(list)

    for result in results:
        for obj in result.boxes:
            if obj.id is not None:
                resistor_id = int(obj.id.item())
                x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())

                if x2 <= x1 or y2 <= y1:
                    continue

                width = x2 - x1
                height = y2 - y1
                orientation = "Horizontal" if width > height else "Vertical"

                roi = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                recognized_colors = []

                for color_name, (lower, upper) in COLOR_RANGES.items():
                    mask = cv2.inRange(hsv, lower, upper)
                    if cv2.countNonZero(mask) > 0 and color_name not in recognized_colors:
                        recognized_colors.append(color_name)

                color_detections[resistor_id].append(recognized_colors)

                most_common_colors = Counter(
                    [color for colors in color_detections[resistor_id] for color in colors]).most_common(4)
                best_colors = [color for color, _ in most_common_colors]

                if len(best_colors) >= 4:
                    result = calculate_resistance(best_colors[:4], COLOR_VALUES)
                    if result is not None:
                        resistance, tolerance = result
                        resistors.append({
                            "id": resistor_id,
                            "colors": best_colors[:4],
                            "resistance": resistance,
                            "tolerance": tolerance,
                            "orientation": orientation,
                            "bbox": [x1, y1, x2, y2]
                        })

    return resistors


def calculate_global_resistance(resistors):
    if not resistors:
        return 0

    groups = []
    unassigned = list(resistors)

    while unassigned:
        current = unassigned.pop(0)
        current_group = [current]

        i = 0
        while i < len(unassigned):
            if any(is_parallel(r, unassigned[i]) for r in current_group):
                current_group.append(unassigned.pop(i))
            else:
                i += 1

        groups.append(current_group)

    total_resistance = 0

    for group in groups:
        if len(group) == 1:
            total_resistance += group[0]["resistance"]
        else:
            parallel_resistance = sum(1 / r["resistance"] for r in group)
            if parallel_resistance > 0:
                total_resistance += 1 / parallel_resistance

    return total_resistance