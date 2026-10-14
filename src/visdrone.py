"""VisDrone class definitions and the mapping onto COCO classes used for zero-shot evaluation."""

# Raw VisDrone annotation IDs (column 6 of the DET annotation files).
CLASS_NAMES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

# VisDrone ID -> COCO ID, for classes with a reasonable COCO counterpart.
# Unmapped (dropped from GT): ignored, tricycle, awning-tricycle, others.
VISDRONE_TO_COCO = {
    1: 0,   # pedestrian      -> person
    2: 0,   # people          -> person  (VisDrone splits by pose; COCO doesn't)
    3: 1,   # bicycle         -> bicycle
    4: 2,   # car             -> car
    5: 2,   # van             -> car     (COCO labels vans/minivans as car)
    6: 7,   # truck           -> truck
    9: 5,   # bus             -> bus
    10: 3,  # motor           -> motorcycle
}
