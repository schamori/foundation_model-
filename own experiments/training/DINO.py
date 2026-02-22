import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import sys
from tqdm import tqdm

# Assuming Augmentations folder is in the parent directory
sys.path.append(str(Path(__file__).parent.parent.parent / "Augmentations"))
try:
    from Augmentations import DINOAugmentations
except ImportError:
    print("Error: Could not import Augmentations. Check your path.")
    sys.exit(1)

from transformers import ConvNextModel


class DINOHead(nn.Module):
    """Projection head for DINO with bottleneck for memory efficiency."""
    def __init__(self, in_dim, out_dim=1536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim), # Bottleneck to save VRAM
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    """Student/Teacher model with ConvNext backbone + head."""
    def __init__(self, model_name="facebook/convnext-large-224", out_dim=1536):
        super().__init__()
        self.backbone = ConvNextModel.from_pretrained(model_name)
        self.head = DINOHead(self.backbone.config.hidden_sizes[-1], out_dim)

    def forward(self, x):
        features = self.backbone(x).pooler_output.squeeze(-1).squeeze(-1)
        return self.head(features)


class ImageDataset(Dataset):
    """Simple dataset that loads random images."""
    def __init__(self, root: Path, n_global=2, n_local=8):
        self.images = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
        if len(self.images) == 0:
            print(f"Warning: No images found in {root}")
        self.aug = DINOAugmentations()
        self.n_global, self.n_local = n_global, n_local

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        globals_ = [self.aug.apply_global(img, v+1)[0] for v in range(self.n_global)]
        locals_ = [self.aug.apply_local(img)[0] for _ in range(self.n_local)]
        return globals_, locals_


class DINOLoss(nn.Module):
    """DINO loss with centering and sharpening."""
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out, update_center=True):
        """Added update_center flag to separate loss calculation from center tracking."""
        student_out = student_out / self.student_temp
        teacher_out = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        
        if update_center:
            self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        self.center = self.center * self.center_momentum + teacher_out.mean(0, keepdim=True) * (1 - self.center_momentum)


@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    """EMA update of teacher from student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data = momentum * pt.data + (1 - momentum) * ps.data


def train_dino(data_root: Path, epochs=100, batch_size=23, lr=5e-4, device="cuda:0", resume_path=None):
    # 1. Automatic Learning Rate Scaling
    lr = lr * (batch_size / 256)
    
    student = DINOModel().to(device)
    teacher = DINOModel().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    dataset = ImageDataset(data_root, n_global=2, n_local=8)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_crops
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    scaler = torch.amp.GradScaler()
    criterion = DINOLoss(1536).to(device)

    start_epoch = 0

    # --- Resume Logic ---
    if resume_path is not None and Path(resume_path).exists():
        print(f"Resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        criterion.center = checkpoint['center'].to(device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")
    # --------------------

    n_global = 2
    n_local = 8
    n_crops = n_global + n_local

    print(f"Starting training on {len(dataset)} images for {epochs} epochs.")

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for globals_, locals_ in pbar:
            global_batch = torch.cat(globals_, dim=0).to(device, non_blocking=True)
            local_batch = torch.cat(locals_, dim=0).to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda"):
                # --- Teacher Forward ---
                with torch.no_grad():
                    teacher_out = teacher(global_batch)

                # --- Student Forward (Split by Resolution) ---
                student_global = student(global_batch)   # 224x224
                student_local = student(local_batch)     # 96x96
                student_out = torch.cat([student_global, student_local], dim=0)

                # Reshape for Vectorized Loss
                B = globals_[0].shape[0]
                teacher_out = teacher_out.view(n_global, B, -1)  # [2, B, D]
                student_out = student_out.view(n_crops, B, -1)   # [10, B, D]

                # --- Vectorized Loss Calculation ---
                # 1. Update Center (using flattened teacher output)
                criterion.update_center(teacher_out.flatten(0, 1))

                # 2. Create Match Grids
                s_grid = torch.arange(n_crops, device=device).unsqueeze(1).expand(-1, n_global) # [10, 2]
                t_grid = torch.arange(n_global, device=device).unsqueeze(0).expand(n_crops, -1) # [10, 2]

                # 3. Mask diagonal (skip where student crop == teacher crop)
                mask = s_grid != t_grid # [10, 2] boolean

                # 4. Flatten using mask
                s_batch = student_out[s_grid[mask]].flatten(0, 1) # [18*B, D]
                t_batch = teacher_out[t_grid[mask]].flatten(0, 1) # [18*B, D]

                # 5. Compute Loss (center update disabled here)
                loss = criterion(s_batch, t_batch, update_center=False)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            update_teacher(student, teacher, momentum=0.996)
            total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

        avg_loss = total_loss / len(loader)
        pbar.set_description(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Save everything needed to resume cleanly
        torch.save({
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "center": criterion.center,
            "epoch": epoch
        }, f"dino_epoch{epoch+1}.pt")

    return student, teacher    # 1. Automatic Learning Rate Scaling
    lr = lr * (batch_size / 256)
    
    student = DINOModel().to(device)
    teacher = DINOModel().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    dataset = ImageDataset(data_root, n_global=2, n_local=8)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_crops
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    scaler = torch.amp.GradScaler()
    criterion = DINOLoss(1536).to(device)

    n_global = 2
    n_local = 8
    n_crops = n_global + n_local

    print(f"Starting training on {len(dataset)} images for {epochs} epochs.")

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for globals_, locals_ in pbar:
            global_batch = torch.cat(globals_, dim=0).to(device, non_blocking=True)
            local_batch = torch.cat(locals_, dim=0).to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda"):
                # --- Teacher Forward ---
                with torch.no_grad():
                    teacher_out = teacher(global_batch)

                # --- Student Forward (Split by Resolution) ---
                student_global = student(global_batch)   # 224x224
                student_local = student(local_batch)     # 96x96
                student_out = torch.cat([student_global, student_local], dim=0)

                # Reshape for Vectorized Loss
                B = globals_[0].shape[0]
                teacher_out = teacher_out.view(n_global, B, -1)  # [2, B, D]
                student_out = student_out.view(n_crops, B, -1)   # [10, B, D]

                # --- Vectorized Loss Calculation ---
                # 1. Update Center (using flattened teacher output)
                criterion.update_center(teacher_out.flatten(0, 1))

                # 2. Create Match Grids
                s_grid = torch.arange(n_crops, device=device).unsqueeze(1).expand(-1, n_global) # [10, 2]
                t_grid = torch.arange(n_global, device=device).unsqueeze(0).expand(n_crops, -1) # [10, 2]

                # 3. Mask diagonal (skip where student crop == teacher crop)
                mask = s_grid != t_grid # [10, 2] boolean

                # 4. Flatten using mask
                s_batch = student_out[s_grid[mask]].flatten(0, 1) # [18*B, D]
                t_batch = teacher_out[t_grid[mask]].flatten(0, 1) # [18*B, D]

                # 5. Compute Loss (center update disabled here)
                loss = criterion(s_batch, t_batch, update_center=False)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            update_teacher(student, teacher, momentum=0.996)
            total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

        avg_loss = total_loss / len(loader)
        pbar.set_description(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Save every epoch
        torch.save({
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "epoch": epoch
        }, f"dino_epoch{epoch+1}.pt")

    return student, teacher


def collate_crops(batch):
    n_global = 2
    n_local = 8
    globals_ = [torch.stack([sample[0][i] for sample in batch]) for i in range(n_global)]
    locals_ = [torch.stack([sample[1][i] for sample in batch]) for i in range(n_local)]
    return globals_, locals_

if __name__ == "__main__":
    # Point this to a valid checkpoint file if you want to resume, else keep None
    checkpoint_to_resume = None # Example path, set to None to start fresh
    
    # Check if the file actually exists before passing it, otherwise pass None
    if checkpoint_to_resume and not Path(checkpoint_to_resume).exists():
        print(f"Checkpoint {checkpoint_to_resume} not found. Starting fresh.")
        checkpoint_to_resume = None

    train_dino(
        Path("/media/HDD1/moritz/foundential/Extracted Frames"), 
        resume_path=checkpoint_to_resume
    )