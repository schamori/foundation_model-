import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from transformers import ConvNextModel
import argparse
import math


class DINOHead(nn.Module):
    """Projection head for DINO - must match training.py exactly for loading."""
    def __init__(self, in_dim, out_dim=1536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    """Student/Teacher model - must match training.py for checkpoint loading."""
    def __init__(self, model_name="facebook/convnext-large-224", out_dim=1536):
        super().__init__()
        self.backbone = ConvNextModel.from_pretrained(model_name)
        self.head = DINOHead(self.backbone.config.hidden_sizes[-1], out_dim)

    def forward(self, x):
        features = self.backbone(x).pooler_output.squeeze(-1).squeeze(-1)
        return self.head(features)


def load_model(checkpoint_path=None, device='cuda', use_teacher=True):
    """
    Load ConvNeXt model from checkpoint or ImageNet pretrained.

    Args:
        checkpoint_path: Path to DINO checkpoint (.pt file), or None for ImageNet pretrained
        device: Device to load model on
        use_teacher: If True, load teacher weights (recommended for inference); else student

    Returns:
        backbone: ConvNextModel ready for feature extraction
    """
    model_name = "facebook/convnext-large-224"

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Loading from checkpoint: {checkpoint_path}")

        # Create full DINOModel to load state dict
        full_model = DINOModel(model_name=model_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        key = 'teacher' if use_teacher else 'student'
        if key in checkpoint:
            full_model.load_state_dict(checkpoint[key])
            print(f"Loaded {key} weights from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading directly (older checkpoint format)
            full_model.load_state_dict(checkpoint)
            print("Loaded weights directly from checkpoint")

        backbone = full_model.backbone
    else:
        if checkpoint_path is not None:
            print(f"Checkpoint not found at {checkpoint_path}, falling back to ImageNet pretrained")
        else:
            print("No checkpoint specified, using ImageNet pretrained weights")

        backbone = ConvNextModel.from_pretrained(model_name)

    return backbone.to(device).eval()


class PatchSimilarityVisualizer:
    def __init__(self, model, device='cuda'):
        """
        model: HuggingFace ConvNextModel backbone
        """
        self.device = device
        self.model = model.to(device).eval()
        self.features = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on the last stage of HuggingFace ConvNeXt."""
        def hook(module, input, output):
            # HuggingFace ConvNeXt stage output is a tensor [B, C, H, W]
            self.features = output

        # HuggingFace ConvNextModel structure: encoder.stages[-1]
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'stages'):
            self.model.encoder.stages[-1].register_forward_hook(hook)
            print("Registered hook on encoder.stages[-1]")
        else:
            raise ValueError("Could not find stages in model. Check model architecture.")

    def extract_features(self, img_tensor):
        """Extract dense feature map from image."""
        with torch.no_grad():
            _ = self.model(img_tensor.to(self.device))
        return self.features

    def compute_similarity_map(self, features, query_h, query_w):
        """
        Compute cosine similarity between query patch and all patches.

        features: [1, C, H, W]
        query_h, query_w: coordinates in feature map space
        """
        B, C, H, W = features.shape

        # Get query feature vector [C]
        query_feat = features[0, :, query_h, query_w]

        # Reshape features to [H*W, C]
        feat_flat = features[0].permute(1, 2, 0).reshape(-1, C)

        # Normalize for cosine similarity
        query_norm = F.normalize(query_feat.unsqueeze(0), dim=1)
        feat_norm = F.normalize(feat_flat, dim=1)

        # Cosine similarity [H*W]
        similarity = torch.mm(feat_norm, query_norm.T).squeeze()

        # Reshape back to spatial [H, W]
        sim_map = similarity.reshape(H, W)

        return sim_map.cpu().numpy()

    def get_query_feature(self, features, query_h, query_w):
        """Return query feature vector from feature map."""
        return features[0, :, query_h, query_w]

    def visualize(self, image_path, query_points, figsize=(15, 15), output_size=512):
        """
        Create visualization like the DINOv3 figure.

        image_path: path to image
        query_points: list of (y, x) tuples in IMAGE coordinates (0-1 normalized)
        output_size: size to resize image for display
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)
        img_display = np.array(img.resize((output_size, output_size)))

        # Extract features
        features = self.extract_features(img_tensor)
        _, C, feat_H, feat_W = features.shape

        print(f"Feature map shape: {feat_H}x{feat_W} with {C} channels")

        # Create figure layout
        n_queries = len(query_points)
        n_cols = min(3, n_queries + 1)
        n_rows = (n_queries + n_cols) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        # Original image in center
        center_idx = len(axes) // 2
        axes[center_idx].imshow(img_display)
        axes[center_idx].set_title('Original Image')
        axes[center_idx].axis('off')

        # Plot similarity maps
        plot_idx = 0
        for i, (qy, qx) in enumerate(query_points):
            if plot_idx == center_idx:
                plot_idx += 1

            # Convert normalized coords to feature map coords
            feat_y = int(qy * feat_H)
            feat_x = int(qx * feat_W)
            feat_y = np.clip(feat_y, 0, feat_H - 1)
            feat_x = np.clip(feat_x, 0, feat_W - 1)

            # Compute similarity
            sim_map = self.compute_similarity_map(features, feat_y, feat_x)

            # Upsample to image size for overlay
            sim_map_upsampled = F.interpolate(
                torch.tensor(sim_map).unsqueeze(0).unsqueeze(0),
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            # Plot
            ax = axes[plot_idx]
            ax.imshow(img_display, alpha=0.3)
            im = ax.imshow(sim_map_upsampled, cmap='turbo', alpha=0.7,
                          vmin=0, vmax=1)

            # Mark query point with red cross
            img_y = int(qy * output_size)
            img_x = int(qx * output_size)
            ax.plot(img_x, img_y, 'r+', markersize=15, markeredgewidth=3)

            ax.set_title(f'Query {i+1}: ({qy:.2f}, {qx:.2f})')
            ax.axis('off')

            plot_idx += 1

        # Hide unused axes
        for idx in range(plot_idx, len(axes)):
            if idx != center_idx:
                axes[idx].axis('off')

        plt.tight_layout()
        return fig


class InteractiveSimilarityVisualizer(PatchSimilarityVisualizer):
    def interactive_visualize(self, image_path, output_size=512):
        """Click on left image to see similarity map on right. Requires %matplotlib widget."""
        img = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)
        img_display = np.array(img.resize((output_size, output_size)))

        features = self.extract_features(img_tensor)
        _, C, feat_H, feat_W = features.shape

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(img_display)
        ax1.set_title('Click to select query point')
        ax1.axis('off')
        ax2.imshow(img_display)
        ax2.set_title('Similarity map')
        ax2.axis('off')

        def onclick(event):
            if event.inaxes != ax1:
                return

            # Get click coordinates (normalized)
            qx = event.xdata / output_size
            qy = event.ydata / output_size

            # Convert to feature coords
            feat_y = int(qy * feat_H)
            feat_x = int(qx * feat_W)
            feat_y = np.clip(feat_y, 0, feat_H - 1)
            feat_x = np.clip(feat_x, 0, feat_W - 1)

            # Compute similarity
            sim_map = self.compute_similarity_map(features, feat_y, feat_x)
            sim_up = F.interpolate(
                torch.tensor(sim_map).unsqueeze(0).unsqueeze(0),
                size=(output_size, output_size), mode='bilinear', align_corners=False
            ).squeeze().numpy()

            # Update right plot
            ax2.clear()
            ax2.imshow(img_display, alpha=0.3)
            ax2.imshow(sim_up, cmap='turbo', alpha=0.7, vmin=0, vmax=1)
            ax2.plot(event.xdata, event.ydata, 'r+', markersize=20, markeredgewidth=3)
            ax2.set_title(f'Similarity from ({qy:.2f}, {qx:.2f})')
            ax2.axis('off')
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

def _sobel_grad_magnitude(img_rgb):
    """Compute Sobel gradient magnitude for an RGB image array."""
    # Convert to grayscale for edge magnitude
    gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gray = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze().numpy()
    return mag

def _pearson_corr(a, b):
    """Compute Pearson correlation between two same-shape arrays."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _grid_points(n_points, rows=4, cols=5, margin=0.1):
    """Generate normalized grid points (y, x)."""
    if rows * cols != n_points:
        rows = int(round(math.sqrt(n_points)))
        cols = int(math.ceil(n_points / rows))
    ys = np.linspace(margin, 1.0 - margin, rows)
    xs = np.linspace(margin, 1.0 - margin, cols)
    points = [(float(y), float(x)) for y in ys for x in xs]
    return points[:n_points]

def _local_contrast(sim_map, qy, qx, radius):
    """Compute local contrast score from a similarity map."""
    h, w = sim_map.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - qy) ** 2 + (xx - qx) ** 2)
    roi_mask = dist <= radius
    bg_mask = ~roi_mask
    roi_vals = sim_map[roi_mask]
    bg_vals = sim_map[bg_mask]
    if roi_vals.size == 0 or bg_vals.size == 0:
        return 0.0
    return float(roi_vals.mean() - bg_vals.mean())


def _structure_weighted_spread(
    sim_map: np.ndarray,
    img_rgb: np.ndarray,
    top_k: float = 0.20,
) -> float:
    """
    Spatial spread of top-k similarity pixels, weighted by edge strength.
    Random activations land on smooth regions and score low.
    Meaningful activations land on structures and score high.
    """
    h, w = sim_map.shape

    # Edge map as weight
    edge_mag = _sobel_grad_magnitude(img_rgb)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)

    # Top-k mask
    threshold = np.percentile(sim_map, (1 - top_k) * 100)
    mask = sim_map >= threshold

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0

    # Weight each active pixel by its edge strength
    weights = edge_mag[ys, xs]
    weights = weights + 0.05  # small floor so zero-edge pixels still count a little

    ys_norm = ys / h
    xs_norm = xs / w

    # Weighted std = spread that lands on real structures
    y_mean = np.average(ys_norm, weights=weights)
    x_mean = np.average(xs_norm, weights=weights)

    y_spread = np.sqrt(np.average((ys_norm - y_mean) ** 2, weights=weights))
    x_spread = np.sqrt(np.average((xs_norm - x_mean) ** 2, weights=weights))

    return float(y_spread + x_spread)

def compute_similarity_scores(
    image_path,
    checkpoint_path=None,
    device='cuda',
    n_points=20,
    grid_rows=4,
    grid_cols=5,
    output_size=512,
    use_teacher=True
):
    """Compute similarity metrics on an image with a grid of query points."""
    backbone = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        use_teacher=use_teacher
    )
    visualizer = PatchSimilarityVisualizer(backbone, device=device)

    img = Image.open(image_path).convert('RGB')
    img_display = np.array(img.resize((output_size, output_size)))
    img_tensor = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0)

    features = visualizer.extract_features(img_tensor)
    _, _, feat_h, feat_w = features.shape

    query_points = _grid_points(n_points, rows=grid_rows, cols=grid_cols)

    # Structure-weighted spread score
    sws_scores = []
    for qy, qx in query_points:
        feat_y = int(np.clip(qy * feat_h, 0, feat_h - 1))
        feat_x = int(np.clip(qx * feat_w, 0, feat_w - 1))
        sim_map = visualizer.compute_similarity_map(features, feat_y, feat_x)
        sim_up = F.interpolate(
            torch.tensor(sim_map).unsqueeze(0).unsqueeze(0),
            size=(output_size, output_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        sws_scores.append(_structure_weighted_spread(sim_up, img_display))
    structure_weighted_spread = float(np.mean(sws_scores)) if sws_scores else 0.0

    # Augmentation stability score
    aug_transforms = [
        ("hflip", transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),
        ("color_jitter", transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),
        ("blur", transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    ]
    stability_scores = {}
    for name, aug_t in aug_transforms:
        img_aug_tensor = aug_t(img).unsqueeze(0)
        features_aug = visualizer.extract_features(img_aug_tensor)
        _, _, feat_h_aug, feat_w_aug = features_aug.shape
        per_query = []
        for qy, qx in query_points:
            qy_adj, qx_adj = qy, qx
            if name == "hflip":
                qx_adj = 1.0 - qx
            feat_y = int(np.clip(qy * feat_h, 0, feat_h - 1))
            feat_x = int(np.clip(qx * feat_w, 0, feat_w - 1))
            sim_map = visualizer.compute_similarity_map(features, feat_y, feat_x)
            sim_up = F.interpolate(
                torch.tensor(sim_map).unsqueeze(0).unsqueeze(0),
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            feat_y_aug = int(np.clip(qy_adj * feat_h_aug, 0, feat_h_aug - 1))
            feat_x_aug = int(np.clip(qx_adj * feat_w_aug, 0, feat_w_aug - 1))
            sim_map_aug = visualizer.compute_similarity_map(features_aug, feat_y_aug, feat_x_aug)
            sim_up_aug = F.interpolate(
                torch.tensor(sim_map_aug).unsqueeze(0).unsqueeze(0),
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            per_query.append(_pearson_corr(sim_up, sim_up_aug))
        stability_scores[name] = float(np.mean(per_query)) if per_query else 0.0

    return {
        "structure_weighted_spread": structure_weighted_spread,
        "augmentation_stability": stability_scores,
    }


# ----- Usage Example -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity scores from an image.")
    parser.add_argument("--image", default="/media/HDD1/moritz/foundential/Extracted Frames/MVD/TITLE_002/frame_000447.jpg", help="Path to input image")
    parser.add_argument("--checkpoint", default=None, help="Path to DINO checkpoint (.pt)")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--output-size", type=int, default=512)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent / "dino_epoch1.pt"

    scores = compute_similarity_scores(
        image_path=args.image,
        checkpoint_path=checkpoint_path,
        device=args.device,
        n_points=args.n_points,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_size=args.output_size,
    )
    print("structure_weighted_spread:", scores["structure_weighted_spread"])
    print("augmentation_stability:", scores["augmentation_stability"])
