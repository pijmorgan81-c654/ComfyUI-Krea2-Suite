import torch
import torch.nn.functional as F
from .utils import (
    choose_krea2_bucket,
    resize_image_tensor,
    resize_mask_tensor,
    prepare_mask_tensor,
    blur_mask_tensor,
    dilate_mask_tensor,
    local_color_match
)

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

def _get_device(default_device):
    if model_management is not None:
        try:
            dev = model_management.get_torch_device()
            if dev is not None:
                return dev
        except Exception:
            pass
    return default_device


class Krea2AspectPreservePrepare:
    """
    Nodo de preparación de lienzo para Krea 2.
    Ajusta la imagen al bucket de resolución nativo de Krea 2 manteniendo la relación
    de aspecto sin distorsión, aplica padding replicado y crea el lienzo verde (#00FF00) o máscara.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "apply_chroma_green": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "KREA2_RESTORE_MAP", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prepared_image", "prepared_mask", "restore_map", "bucket_width", "bucket_height", "batch_size", "info")
    FUNCTION = "prepare"
    CATEGORY = "Krea 2 Suite/Inpainting & Outpainting"

    def prepare(self, image, mask=None, apply_chroma_green=True):
        if image.ndim != 4:
            raise ValueError(f"Se esperaba tensor IMAGE [B, H, W, C], recibido {image.shape}")
        
        batch, src_h, src_w, channels = image.shape
        device = image.device
        dtype = image.dtype

        # Si no se provee máscara, asumimos máscara de ceros (sin modificación inicial)
        if mask is None:
            mask = torch.zeros((batch, src_h, src_w), device=device, dtype=dtype)
        else:
            mask = prepare_mask_tensor(mask, batch, src_h, src_w, device, dtype)

        # Seleccionar bucket óptimo de Krea 2
        bucket_w, bucket_h = choose_krea2_bucket(src_w, src_h)
        scale = min(bucket_w / float(src_w), bucket_h / float(src_h))
        fitted_w = max(1, min(bucket_w, int(round(src_w * scale))))
        fitted_h = max(1, min(bucket_h, int(round(src_h * scale))))

        # Calcular padding centrado
        pad_left = (bucket_w - fitted_w) // 2
        pad_right = bucket_w - fitted_w - pad_left
        pad_top = (bucket_h - fitted_h) // 2
        pad_bottom = bucket_h - fitted_h - pad_top

        # Escalar uniformemente
        fitted_image = resize_image_tensor(image, fitted_h, fitted_w, mode="bicubic")
        fitted_mask = resize_mask_tensor(mask, fitted_h, fitted_w).clamp(0.0, 1.0)

        # Rellenar con modo 'replicate' para los bordes del canvas
        img_nchw = fitted_image.permute(0, 3, 1, 2)
        img_padded = F.pad(img_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        prepared_img = img_padded.permute(0, 2, 3, 1)

        mask_nchw = fitted_mask.unsqueeze(1)
        mask_padded = F.pad(mask_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        prepared_mask = mask_padded.squeeze(1).clamp(0.0, 1.0)

        # Si se usa chroma green para inpaint LoRA (INPAINTKREA convention)
        if apply_chroma_green:
            hard_mask = (prepared_mask > 0.5).unsqueeze(-1)
            pure_green = torch.zeros_like(prepared_img)
            pure_green[..., 1] = 1.0  # RGB verde puro (0, 1, 0)
            prepared_img = torch.where(hard_mask, pure_green, prepared_img)

        restore_map = {
            "source_width": int(src_w),
            "source_height": int(src_h),
            "bucket_width": int(bucket_w),
            "bucket_height": int(bucket_h),
            "fitted_width": int(fitted_w),
            "fitted_height": int(fitted_h),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
        }

        info = (
            f"Krea2 Prepare: original={src_w}x{src_h} -> bucket={bucket_w}x{bucket_h} "
            f"(fitted={fitted_w}x{fitted_h}, pads=[L:{pad_left}, R:{pad_right}, T:{pad_top}, B:{pad_bottom}])"
        )

        return (prepared_img, prepared_mask, restore_map, int(bucket_w), int(bucket_h), int(batch), info)


class Krea2AspectPreserveRestore:
    """
    Nodo de recomposición y restauración exacta de resolución y píxeles nativos.
    Utiliza remapeo bicúbico inverso con grid_sample, corrección de color local
    y costura suavizada mediante dilatación y desenfoque gaussiano.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "mask": ("MASK",),
                "restore_map": ("KREA2_RESTORE_MAP",),
            },
            "optional": {
                "seam_overlap": ("INT", {"default": 6, "min": 0, "max": 128, "step": 1}),
                "seam_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "color_match_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "restored_raw_generated", "info")
    FUNCTION = "restore"
    CATEGORY = "Krea 2 Suite/Inpainting & Outpainting"

    def restore(self, original_image, generated_image, mask, restore_map,
                seam_overlap=6, seam_feather=8, color_match_strength=0.25):
        if original_image.ndim != 4 or generated_image.ndim != 4:
            raise ValueError("Se esperaban tensores IMAGE [B, H, W, C]")

        batch, target_h, target_w, _ = original_image.shape
        if target_w != restore_map["source_width"] or target_h != restore_map["source_height"]:
            raise ValueError(
                f"Dimensiones de original_image ({target_w}x{target_h}) no coinciden con restore_map "
                f"({restore_map['source_width']}x{restore_map['source_height']})"
            )

        orig_device = original_image.device
        orig_dtype = original_image.dtype
        work_device = _get_device(orig_device)
        work_dtype = torch.float32

        expanded = original_image.to(device=work_device, dtype=work_dtype)
        gen = generated_image.to(device=work_device, dtype=work_dtype)

        gen_h, gen_w = gen.shape[1], gen.shape[2]
        bucket_w = restore_map["bucket_width"]
        bucket_h = restore_map["bucket_height"]
        fitted_w = restore_map["fitted_width"]
        fitted_h = restore_map["fitted_height"]
        pad_left = restore_map["pad_left"]
        pad_top = restore_map["pad_top"]

        # Crear malla para remapeo bicúbico inverso
        ys = torch.arange(target_h, device=work_device, dtype=work_dtype)
        xs = torch.arange(target_w, device=work_device, dtype=work_dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        if target_w > 1:
            prep_x = pad_left + xx * ((fitted_w - 1) / float(target_w - 1))
        else:
            prep_x = torch.full_like(xx, float(pad_left))

        if target_h > 1:
            prep_y = pad_top + yy * ((fitted_h - 1) / float(target_h - 1))
        else:
            prep_y = torch.full_like(yy, float(pad_top))

        if gen_w > 1:
            grid_x = (prep_x / float(bucket_w - 1)) * 2.0 - 1.0
        else:
            grid_x = torch.zeros_like(prep_x)

        if gen_h > 1:
            grid_y = (prep_y / float(bucket_h - 1)) * 2.0 - 1.0
        else:
            grid_y = torch.zeros_like(prep_y)

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        if batch > 1 and grid.shape[0] == 1:
            grid = grid.repeat(batch, 1, 1, 1)

        # Muestreo de grilla bicúbico
        restored = F.grid_sample(
            gen.permute(0, 3, 1, 2),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

        # Preparar máscara y corrección de color local
        norm_mask = prepare_mask_tensor(mask, batch, target_h, target_w, work_device, work_dtype)
        restored = local_color_match(restored, expanded, norm_mask, color_match_strength).clamp(0.0, 1.0)

        # Mezcla suave con dilatación y desenfoque
        alpha = dilate_mask_tensor(norm_mask, int(seam_overlap))
        alpha = blur_mask_tensor(alpha, int(seam_feather)).clamp(0.0, 1.0).unsqueeze(-1)
        final = restored * alpha + expanded * (1.0 - alpha)

        # Sanitizar tensores y sincronizar GPU para prevenir multicolores o corrupción de búfer
        final = torch.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)

        final = final.to(device=orig_device, dtype=orig_dtype).contiguous()
        restored = restored.to(device=orig_device, dtype=orig_dtype).contiguous()

        if orig_device.type == "cuda":
            torch.cuda.synchronize(orig_device)

        info = (
            f"Krea2 Restore: restored={target_w}x{target_h} from={gen_w}x{gen_h} "
            f"(seam_overlap={seam_overlap}, feather={seam_feather}, color_match={color_match_strength})"
        )

        return (final, restored, info)


class Krea2OutpaintCanvasExpander:
    """
    Nodo para expandir el lienzo en cualquier dirección (izquierda, derecha, arriba, abajo)
    generando automáticamente la imagen expandida y la máscara de outpainting correspondiente.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            },
            "optional": {
                "fill_color": (["replicate", "black", "pure_green", "white"], {"default": "pure_green"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("expanded_image", "outpaint_mask", "new_width", "new_height")
    FUNCTION = "expand"
    CATEGORY = "Krea 2 Suite/Inpainting & Outpainting"

    def expand(self, image, pad_left, pad_right, pad_top, pad_bottom, fill_color="pure_green"):
        batch, h, w, c = image.shape
        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom

        # Tensor de salida para imagen y máscara
        expanded_img = torch.zeros((batch, new_h, new_w, c), device=image.device, dtype=image.dtype)
        outpaint_mask = torch.ones((batch, new_h, new_w), device=image.device, dtype=image.dtype)

        # Definir color de relleno
        if fill_color == "replicate":
            img_nchw = image.permute(0, 3, 1, 2)
            pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)
            img_padded = F.pad(img_nchw, pad_tuple, mode="replicate")
            expanded_img = img_padded.permute(0, 2, 3, 1)
        elif fill_color == "pure_green":
            expanded_img[..., 1] = 1.0  # verde neón
        elif fill_color == "white":
            expanded_img.fill_(1.0)
        elif fill_color == "black":
            expanded_img.fill_(0.0)

        # Colocar imagen original en la posición central designada
        expanded_img[:, pad_top:pad_top+h, pad_left:pad_left+w, :] = image
        # Marcar la zona original como 0 en la máscara de outpaint (preservar)
        outpaint_mask[:, pad_top:pad_top+h, pad_left:pad_left+w] = 0.0

        return (expanded_img, outpaint_mask, new_w, new_h)
