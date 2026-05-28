# Utils for c3m4_assignment

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm

from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights  # torchvision >= 0.13
    TV_DEFAULT_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    TV_DEFAULT_WEIGHTS = None  # will use pretrained=True fallback

def display_some_images(dataset):
    plt.figure(figsize=(15, 10))
    
    # Get 3 images for each label
    images_per_label = 3
    num_labels = 3
    
    for label in range(num_labels):
        # Find indices of images with this label
        label_indices = [i for i in range(len(dataset)) if dataset[i][1] == label]
        # Take first 3 images of this label
        selected_indices = label_indices[:images_per_label]
        
        for i, img_idx in enumerate(selected_indices):
            # Get image and label
            img, label = dataset[img_idx]
            
            # Convert tensor to numpy array and transpose to correct dimensions
            img = img.permute(1, 2, 0).numpy()
            
            # Denormalize the image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Plot in grid - each row is a label
            plt.subplot(3, 3, label * images_per_label + i + 1)
            plt.imshow(img)
            plt.title(f'Label: {dataset.classes[label]}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def compute_accuracy(model, loader, device):
    from tqdm.notebook import tqdm
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Create progress bar for evaluation
        pbar = tqdm(loader, desc='Computing Accuracy', leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            # Update progress bar with current accuracy
            acc = correct / max(total, 1)
            pbar.set_description(f'Accuracy: {acc:.4f}')
    return correct / max(total, 1)

def make_checkpoint(epoch, model, optimizer, loss, extra=None):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss),
    }
    if extra:
        ckpt.update(extra)
    return ckpt

def train_model(model, train_loader, dev_loader, num_epochs, optimizer, device, checkpoint=None, save_path="best_model.pt"): 
    # Load from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
        if 'val_acc' in checkpoint:
            best_acc = checkpoint['val_acc']
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc='Epochs', position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Create progress bar for batches
        batch_pbar = tqdm(train_loader, desc='Training', position=1, leave=False)
        
        # Train for one epoch
        model.train()
        total_loss, total = 0.0, 0
        for x, y in batch_pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            # Update batch progress bar
            batch_pbar.set_description(f"Batch Loss: {loss.item():.4f}")
            
        train_loss = total_loss / max(total, 1)
        batch_pbar.close()
        
        # Evaluate
        val_acc = compute_accuracy(model, dev_loader, device)
        
        # Update epoch progress bar description
        epoch_pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = make_checkpoint(epoch, model, optimizer, train_loss,
                                 extra={"val_acc": val_acc})
            torch.save(ckpt, "best_model.pt")
            epoch_pbar.write(f"New best accuracy: {val_acc:.4f}, saved model to best_model.pt")
    
    # Save final model
    ckpt = make_checkpoint(num_epochs-1, model, optimizer, train_loss,
                          extra={"val_acc": val_acc})
    torch.save(ckpt, save_path)
    
    # Final status update
    print(f"\nTraining completed:")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Final accuracy: {val_acc:.4f}")
    print(f"Final model saved to final_model.pt")
    
    return model, best_acc

def _iter_prunable_modules(model):
    """Yield (qualified_name, module) for Conv2d and Linear layers only."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def sparsity_report(model):
    """
    Compute per-layer and global sparsity (fraction of zeros) over Conv2d/Linear weights.
    Returns
    -------
    dict
        {
          "layers": { "<name>.weight": 0.52, ... },
          "global_sparsity": 0.47
        }
    """
    layers = {}
    zeros_total = 0
    elems_total = 0

    for name, module in _iter_prunable_modules(model):
        if not hasattr(module, "weight"):
            continue
        w = module.weight.detach()
        z = (w == 0).sum().item()
        n = w.numel()
        layers[f"{name}.weight"] = (z / n) if n > 0 else 0.0
        zeros_total += z
        elems_total += n

    global_sparsity = (zeros_total / elems_total) if elems_total > 0 else 0.0
    return {"layers": layers, "global_sparsity": global_sparsity}

def bench(m, iters=20, shape = (16, 3, 224, 224), device="cpu"):
    torch.manual_seed(17)
    m.eval()
    x = torch.randn(shape).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x)
    return (time.perf_counter() - start) / iters

class ToyNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
        )
        self.block = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),                 # Conv + ReLU
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),                      # Conv + BN
            nn.ReLU(inplace=False),                 # Conv + BN + ReLU (across indices)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16),
            nn.ReLU(inplace=False),                 # Linear + ReLU
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.head(x)
        return x

# Utility: pretty-print a few layers
def list_children(module, title):
    print(f"\n== {title} ==")
    for name, child in module.named_modules():
        # show only immediate children of top-level sequentials
        if name in {"stem", "block", "head"}:
            print(f"\n[{name}]")
            for i, sub in enumerate(child.children()):
                print(f"  {i}: {sub.__class__.__name__}")


# Utility: count intrinsic fused layers by class name
def count_fused_layers(model):
    names = []
    for m in model.modules():
        cls = m.__class__.__name__
        if any(k in cls for k in ["ConvReLU2d", "ConvBn2d", "ConvBnReLU2d", "LinearReLU"]):
            names.append(cls)
    return {k: names.count(k) for k in sorted(set(names))}

import torch
import torch.nn as nn
try:
    # PyTorch >= 1.13 path
    from torch.ao.quantization import QuantStub, DeQuantStub
except Exception:
    # Older fallback
    from torch.quantization import QuantStub, DeQuantStub


class QATBasicBlock(nn.Module):
    """
    ResNet BasicBlock variant where fusible parts are inside nn.Sequential,
    and residual add uses FloatFunctional (quantization-friendly).
    Pattern inside self.block: Conv-BN-ReLU, Conv-BN  -> your fuse pass will fuse these.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # Main path as a single Sequential so your fuse pass can see adjacent ops
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        # Downsample path as Sequential (or Identity) so it can be fused too
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = nn.Identity()

        # FloatFunctional add so the residual add is quantization-aware
        self.skip_add = torch.nn.quantized.FloatFunctional()
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if not isinstance(self.downsample, nn.Identity):
            identity = self.downsample(x)

        out = self.block(x)
        out = self.skip_add.add(out, identity)  # quantization-friendly residual add
        out = self.out_relu(out)
        return out


class QATResNet18(nn.Module):
    """
    A ResNet18 layout rearranged for your fuse_model_inplace (fuses only inside nn.Sequential).
    Includes QuantStub/DeQuantStub so QAT prepare/convert works cleanly.
    """
    def __init__(self, num_classes=1000, use_quant_stubs=False):
        super().__init__()
        self.use_quant_stubs = use_quant_stubs
        if use_quant_stubs:
            self.quant = QuantStub()
        else:
            self.quant = nn.Identity()

        # Stem as Sequential so Conv-BN-ReLU can fuse
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 stages: [2,2,2,2] blocks, strides: [1,2,2,2]
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * QATBasicBlock.expansion, num_classes)

        if use_quant_stubs:
            self.dequant = DeQuantStub()
        else:
            self.dequant = nn.Identity()

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = [QATBasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(QATBasicBlock(planes, planes, stride=1))
        # Put the stack into a Sequential so your fuse pass recurses into it
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)

        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x

@torch.no_grad()
def load_imagenet_pretrained_into_qat_resnet18(model: nn.Module,
                                              weights=TV_DEFAULT_WEIGHTS,
                                              strict=False):
    """
    Map torchvision ResNet18 ImageNet weights into your QATResNet18 structure.

    Returns:
        model (nn.Module): same instance with weights loaded.
        missing_keys (list[str]), unexpected_keys (list[str])
    """
    # Build a vanilla torchvision resnet18 with ImageNet weights
    if TV_DEFAULT_WEIGHTS is not None:
        tv = resnet18(weights=weights)
    else:
        tv = resnet18(pretrained=True)  # older torchvision

    tv_sd = tv.state_dict()
    new_sd = {}

    for k, v in tv_sd.items():
        # Top stem: conv1/bn1 -> stem.0/.1 (ReLU has no params)
        if k.startswith("conv1."):
            nk = "stem.0." + k.split(".", 1)[1]
        elif k.startswith("bn1."):
            nk = "stem.1." + k.split(".", 1)[1]

        # Stages: layer{l}.{b}.(conv1|bn1|conv2|bn2|downsample.{0,1})
        elif k.startswith("layer"):
            parts = k.split(".")
            # e.g. ['layer1','0','conv1','weight'] or ['layer2','1','downsample','0','weight']
            layer = parts[0]           # 'layer1'..'layer4'
            block_idx = parts[1]       # '0' or '1'
            name = parts[2]            # 'conv1'|'bn1'|'conv2'|'bn2'|'downsample'

            if name == "conv1":
                nk = f"{layer}.{block_idx}.block.0." + ".".join(parts[3:])
            elif name == "bn1":
                nk = f"{layer}.{block_idx}.block.1." + ".".join(parts[3:])
            elif name == "conv2":
                nk = f"{layer}.{block_idx}.block.3." + ".".join(parts[3:])
            elif name == "bn2":
                nk = f"{layer}.{block_idx}.block.4." + ".".join(parts[3:])
            elif name == "downsample":
                # downsample.0 (conv), downsample.1 (bn)
                nk = f"{layer}.{block_idx}.downsample." + ".".join(parts[3:])
            else:
                # Unrecognized subkey under layer* -> skip
                continue

        # FC head (only load if same num_classes)
        elif k.startswith("fc."):
            # If shapes match, keep; otherwise skip (e.g., custom num_classes)
            if getattr(model.fc, "weight", None) is not None and model.fc.weight.shape == v.shape:
                nk = k
            else:
                continue

        else:
            # No params to map (relu/maxpool/FloatFunctional/etc.)
            continue

        new_sd[nk] = v

    incompat = model.load_state_dict(new_sd, strict=strict)
    return model, list(incompat.missing_keys), list(incompat.unexpected_keys)



def resnet18_qat_ready_pretrained(num_classes=1000, use_quant_stubs=False):
    """
    Build your QATResNet18, load ImageNet weights, and return the FP32 model.
    If num_classes != 1000, the FC layer is randomly initialized.
    """
    model_fp32 = QATResNet18(num_classes=num_classes, use_quant_stubs=use_quant_stubs)
    # Load ImageNet weights where shapes match
    model_fp32, missing, unexpected = load_imagenet_pretrained_into_qat_resnet18(
        model_fp32, weights=TV_DEFAULT_WEIGHTS, strict=False
    )
    return model_fp32
