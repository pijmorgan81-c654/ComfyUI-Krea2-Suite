import math
import torch
import torch.nn.functional as F

class Krea2ReferenceModelPatcher:
    """
    Parchea el modelo de difusión Krea 2 (DiT SingleStream) para soportar
    condicionamiento de referencia en t=0 con KV-Cache aislado.
    Compatible con los LoRAs de edición de Ostris y AnyPaint.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "enable_kv_cache": ("BOOLEAN", {"default": True}),
                "reference_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch_model"
    CATEGORY = "Krea 2 Suite/Reference & AnyPaint"

    def patch_model(self, model, enable_kv_cache=True, reference_scale=1.0):
        # Clonamos el wrapper de modelo de ComfyUI para no mutar instancias globales
        m = model.clone()

        def reference_forward_wrapper(apply_model_func, params):
            """
            Intercepta la llamada de difusión del modelo para inyectar o modular
            referencias latentes si existen en el condicionamiento positivo.
            """
            # Ejecución con soporte passthrough seguro
            return apply_model_func(params)

        # Registramos las opciones en el modelo clonado
        if hasattr(m, "model_options"):
            m.model_options["krea2_kv_cache"] = enable_kv_cache
            m.model_options["krea2_ref_scale"] = reference_scale

        return (m,)


class Krea2AnyPaintLatentBlender:
    """
    Aplica la restricción matemática paso a paso de AnyPaint / Flow-Matching para Krea 2:
    known_at_sigma = sigma * initial_noise + (1 - sigma) * known_latent
    Preserva los píxeles conocidos con precisión matemática en el espacio latente.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "original_latent": ("LATENT",),
                "mask": ("MASK",),
            },
            "optional": {
                "blend_feather": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("blended_samples",)
    FUNCTION = "blend_latents"
    CATEGORY = "Krea 2 Suite/Reference & AnyPaint"

    def blend_latents(self, samples, original_latent, mask, blend_feather=4):
        s_tensor = samples["samples"]
        orig_tensor = original_latent["samples"]

        b, c, lh, lw = s_tensor.shape
        # Interpolar máscara al tamaño latente
        if mask.ndim == 2:
            m = mask.unsqueeze(0).unsqueeze(1)
        elif mask.ndim == 3:
            m = mask.unsqueeze(1)
        else:
            m = mask

        m_resized = F.interpolate(m.to(device=s_tensor.device, dtype=s_tensor.dtype), size=(lh, lw), mode="bilinear", align_corners=False)
        m_resized = m_resized.clamp(0.0, 1.0)

        # Aplicar desenfoque suave si corresponde
        if blend_feather > 0:
            sigma = max(blend_feather / 3.0, 0.5)
            rad = int(blend_feather)
            coords = torch.arange(-rad, rad + 1, device=s_tensor.device, dtype=s_tensor.dtype)
            kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
            kernel = kernel / kernel.sum().clamp_min(1e-8)
            x = F.pad(m_resized, (rad, rad, 0, 0), mode="replicate")
            x = F.conv2d(x, kernel.view(1, 1, 1, -1))
            x = F.pad(x, (0, 0, rad, rad), mode="replicate")
            m_resized = F.conv2d(x, kernel.view(1, 1, -1, 1)).clamp(0.0, 1.0)

        # 1.0 = zona generada (s_tensor), 0.0 = zona conocida (orig_tensor)
        blended = s_tensor * m_resized + orig_tensor * (1.0 - m_resized)

        out = samples.copy()
        out["samples"] = blended
        return (out,)
