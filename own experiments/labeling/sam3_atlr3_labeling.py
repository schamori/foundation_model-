from pathlib import Path
import multiprocessing as mp

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

mp.set_start_method("spawn", force=True)

FRAMES_ROOT = Path("/media/HDD1/moritz/foundential/Extracted Frames/MVD")
VIDEO_ID = "TITLE_003"
VIDEO_FRAMES_DIR = FRAMES_ROOT / VIDEO_ID

DATASET_NAME = f"{VIDEO_ID}-frames"

dataset = (
    fo.load_dataset(DATASET_NAME)
    if fo.dataset_exists(DATASET_NAME)
    else fo.Dataset.from_images_dir(str(VIDEO_FRAMES_DIR), name=DATASET_NAME, persistent=True)
)

foz.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")
model = foz.load_zoo_model("facebook/sam3")
model.device = "cuda"
model.operation = "visual_segmentation"

session = fo.launch_app(dataset)

print("App is open.")
print("Draw prompt boxes into field: prompts")
print("Press Enter in this terminal to run SAM 3 on prompted samples.")
print("Type q then Enter to quit.")

while True:
    cmd = input("> ").strip().lower()
    if cmd == "q":
        break

    view = dataset.match(F("prompts.detections").length() > 0)

    if len(view) == 0:
        print("No prompts found. Add boxes to field prompts in the App.")
        continue

    view.apply_model(
        model,
        label_field="sam3_out",
        prompt_field="prompts",
        batch_size=4,
        num_workers=0,
    )

    print(f"Updated {len(view)} samples. Toggle sam3_out in the App to see masks.")

session.wait()
