from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch
from datasets import load_dataset


def test_max_batch_size_training(
    model_name: str = "facebook/convnext-large-224-22k-1k",
    dataset_name: str = "huggingface/cats-image",
    start_batch_size: int = 1,
    step: int = 4,
    max_batch_size: int = 1024,
) -> None:
    """
    Incrementally tests larger batch sizes *while training* until a CUDA OOM happens,
    then reports the largest batch size that fits in VRAM.

    The model is trained with completely random labels so we only measure VRAM usage,
    not accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("CUDA is not available on this machine. Cannot test GPU VRAM batch size.")
        return

    print("Loading dataset and model...")
    dataset = load_dataset(dataset_name)
    image = dataset["test"]["image"][0]

    model = ConvNextForImageClassification.from_pretrained(model_name).to(device)
    model.train()

    num_labels = model.config.num_labels

    processor = ConvNextImageProcessor.from_pretrained(
        model_name,
        size={"shortest_edge": 384},
        crop_size={"height": 384, "width": 384},
    )

    # Prepare a single image tensor; we'll expand it along the batch dimension.
    base_inputs = processor(images=image, return_tensors="pt")
    pixel_values = base_inputs["pixel_values"].to(device)

    print(f"Single-image tensor shape: {pixel_values.shape}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_batch = 0
    batch_size = start_batch_size

    while batch_size <= max_batch_size:
        try:
            print(f"\nTrying batch size (training step): {batch_size}")

            # Expand along batch dimension without extra memory for the input tensor itself.
            batch_pixel_values = pixel_values.expand(batch_size, -1, -1, -1)

            # Create completely random labels for this batch.
            labels = torch.randint(
                low=0,
                high=num_labels,
                size=(batch_size,),
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=batch_pixel_values, labels=labels)
            loss = outputs.loss

            print(f"Dummy training loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

            best_batch = batch_size
            print(f"Success at batch size: {batch_size}")

            # Free any cached memory before next try
            if device == "cuda":
                torch.cuda.empty_cache()

            batch_size += step

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size: {batch_size}")
                # Clean up CUDA memory after OOM
                if device == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise

    if best_batch > 0:
        print(f"\nMaximum training batch size that fit without OOM: {best_batch}")
    else:
        print("\nNo batch size succeeded; try reducing model size or image resolution.")


if __name__ == "__main__":
    test_max_batch_size_training()
