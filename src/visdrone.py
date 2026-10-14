"""VisDrone class definitions and the mapping onto COCO classes used for zero-shot evaluation."""

CLASS_NAMES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

CLASS_COLORS = {
    0: "gray", 1: "crimson", 2: "darkorange", 3: "gold", 4: "limegreen", 5: "turquoise",
    6: "dodgerblue", 7: "darkorchid", 8: "magenta", 9: "sienna", 10: "pink", 11: "white",
}

# Trained models use YOLO IDs 0-9 = raw VisDrone IDs 1-10 (ignored/others dropped).
YOLO_TO_RAW = {i: i + 1 for i in range(10)}

# VisDrone ID -> COCO ID, for classes with a reasonable COCO counterpart.
# Unmapped (dropped from GT): ignored, tricycle, awning-tricycle, others.
VISDRONE_TO_COCO = {
    1: 0,   # pedestrian      -> person
    2: 0,   # people          -> person
    3: 1,   # bicycle         -> bicycle
    4: 2,   # car             -> car
    5: 2,   # van             -> car
    6: 7,   # truck           -> truck
    9: 5,   # bus             -> bus
    10: 3,  # motor           -> motorcycle
}
