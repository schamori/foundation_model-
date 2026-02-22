import multiprocessing as mp
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

mp.set_start_method("spawn", force=True)

FRAMES_ROOT = Path("/media/HDD1/moritz/foundential/Extracted Frames/MVD")
VIDEO_ID = "TITLE_003"
FRAMES_DIR = FRAMES_ROOT / VIDEO_ID
DATASET_NAME = f"{VIDEO_ID}-frames"

if not FRAMES_DIR.exists():
    raise FileNotFoundError(f"Missing frames folder: {FRAMES_DIR}")

if fo.dataset_exists(DATASET_NAME):
    dataset = fo.load_dataset(DATASET_NAME)
else:
    dataset = fo.Dataset.from_images_dir(str(FRAMES_DIR), name=DATASET_NAME, persistent=True)

if "video_id" not in dataset.get_field_schema():
    dataset.add_sample_field("video_id", fo.StringField)
dataset.set_values("video_id", [VIDEO_ID] * len(dataset))

model = foz.load_zoo_model("segment-anything-2-hiera-tiny-image-torch")
model.device = "cuda"

session = fo.launch_app(dataset)

print("App ready.")
print("Create prompt boxes in field: prompts")
print("Results write to field: sam2_out")



session.wait()
