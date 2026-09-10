import math
import torch
import torch.nn.functional as F

# Buckets de resolución nativos de Krea 2
KREA2_BUCKETS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688),
    (1568, 672),
]

def choose_krea2_bucket(width: int, height: int):
    """Encuentra el bucket de Krea 2 con la relación de aspecto más cercana."""
    target_ratio = float(width) / float(height)
    return min(KREA2_BUCKETS, key=lambda b: abs(math.log((b[0] / float(b[1])) / target_ratio)))

def resize_image_tensor(image: torch.Tensor, height: int, width: int, mode: str = "bicubic") -> torch.Tensor:
    """
    Redimensiona un tensor de imagen de ComfyUI [B, H, W, C] -> [B, target_H, target_W, C].
    """
    x = image.permute(0, 3, 1, 2)
    if mode in ("bicubic", "bilinear"):
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode)
    return x.permute(0, 2, 3, 1)

def resize_mask_tensor(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Redimensiona un tensor de máscara de ComfyUI [B, H, W] o [H, W] a [B, target_H, target_W].
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return F.interpolate(
        mask.unsqueeze(1),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

def prepare_mask_tensor(mask: torch.Tensor, batch_size: int, height: int, width: int, device, dtype) -> torch.Tensor:
    """
    Normaliza y ajusta la máscara al tamaño de destino, batch y dispositivo.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Formato de MASK inválido. Se esperaba [B, H, W] o [H, W], recibido {mask.shape}")
    
    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.repeat(batch_size, 1, 1)
    
    if mask.shape[1] != height or mask.shape[2] != width:
        mask = resize_mask_tensor(mask, height, width)
        
    return mask.clamp(0.0, 1.0)

def blur_mask_tensor(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Aplica desenfoque gaussiano 2D separable a un tensor de máscara [B, H, W]."""
    radius = int(radius)
    if radius <= 0:
        return mask
    sigma = max(radius / 3.0, 0.5)
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    
    x = mask.unsqueeze(1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(1)

def dilate_mask_tensor(mask: torch.Tensor, amount: int) -> torch.Tensor:
    """Aplica dilatación morfológica (max-pooling) a un tensor de máscara [B, H, W]."""
    amount = int(amount)
    if amount <= 0:
        return mask
    kernel_size = amount * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=amount).squeeze(1)

def local_color_match(generated: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Alinea las estadísticas de color (media y desviación estándar) de la imagen generada
    con la imagen de referencia en un anillo perimetral alrededor de la máscara.
    """
    strength = float(strength)
    if strength <= 0.0:
        return generated
    
    h, w = generated.shape[1], generated.shape[2]
    ring_width = max(24, int(round(min(h, w) * 0.035)))
    outer = dilate_mask_tensor(mask, ring_width)
    ring = (outer - mask).clamp(0.0, 1.0)
    weight = ring.unsqueeze(-1)
    
    count = weight.sum(dim=(1, 2), keepdim=True)
    valid = (count >= 64.0).to(generated.dtype)
    denom = count.clamp_min(64.0)
    
    mean_g = (generated * weight).sum(dim=(1, 2), keepdim=True) / denom
    mean_r = (reference * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_g = (((generated - mean_g) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_r = (((reference - mean_r) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    
    std_g = var_g.clamp_min(1e-6).sqrt()
    std_r = var_r.clamp_min(1e-6).sqrt()
    
    matched = (generated - mean_g) * (std_r / std_g).clamp(0.75, 1.35) + mean_r
    mixed = generated * (1.0 - strength) + matched * strength
    return generated * (1.0 - valid) + mixed * valid
