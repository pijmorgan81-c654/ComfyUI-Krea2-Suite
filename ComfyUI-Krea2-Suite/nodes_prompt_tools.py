import torch
import re

class Krea2InpaintPromptBuilder:
    """
    Nodo de construcción de prompts optimizados para Krea 2 Inpainting / Outpainting.
    Genera el formato exacto requerido por el LoRA INPAINTKREA:
    'Replace the solid neon green area with [TARGET], match the surrounding lighting, perspective, shadows, colors, texture, and depth of field, preserve everything outside the solid neon green area.'
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpaint_target": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful colorful parrot sitting on the shoulder"
                }),
                "mode": (["inpaint_chroma_green", "outpaint_contextual", "direct_edit"], {"default": "inpaint_chroma_green"}),
            },
            "optional": {
                "custom_prefix": ("STRING", {"default": "Replace the solid neon green area with "}),
                "custom_suffix": ("STRING", {
                    "multiline": True,
                    "default": ", match the surrounding lighting, perspective, shadows, colors, texture, and depth of field, preserve everything outside the solid neon green area."
                }),
                "extra_style": ("STRING", {"default": "photorealistic, highly detailed, 8k"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "target_only")
    FUNCTION = "build_prompt"
    CATEGORY = "Krea 2 Suite/Prompting & Tools"

    def build_prompt(self, inpaint_target, mode="inpaint_chroma_green",
                     custom_prefix="Replace the solid neon green area with ",
                     custom_suffix=", match the surrounding lighting, perspective, shadows, colors, texture, and depth of field, preserve everything outside the solid neon green area.",
                     extra_style="photorealistic, highly detailed, 8k"):
        target_cleaned = inpaint_target.strip()
        style_cleaned = extra_style.strip()

        if mode == "inpaint_chroma_green":
            full_prompt = f"{custom_prefix}{target_cleaned}{custom_suffix}"
            if style_cleaned:
                full_prompt = f"{full_prompt}, {style_cleaned}"
        elif mode == "outpaint_contextual":
            if not target_cleaned:
                full_prompt = "seamless outpainting expansion, continuing the natural environment, background scenery, lighting and perspective with high fidelity."
            else:
                full_prompt = f"seamless outpainting expansion of {target_cleaned}, continuing the environment, lighting, perspective and depth."
            if style_cleaned:
                full_prompt = f"{full_prompt}, {style_cleaned}"
        else: # direct_edit
            full_prompt = f"{target_cleaned}"
            if style_cleaned:
                full_prompt = f"{full_prompt}, {style_cleaned}"

        return (full_prompt, target_cleaned)


class Krea2ChromaGreenMaskExtractor:
    """
    Extrae automáticamente una máscara binaria a partir de zonas pintadas
    en verde puro (#00FF00 / RGB 0, 1, 0) sobre la imagen de entrada.
    Ideal si el usuario pinta con el pincel verde de ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "tolerance": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("extracted_mask", "cleaned_image")
    FUNCTION = "extract_chroma"
    CATEGORY = "Krea 2 Suite/Prompting & Tools"

    def extract_chroma(self, image, tolerance=0.15):
        # image: [B, H, W, 3]
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

        # Condición de verde puro neón: g cercano a 1.0, r y b cercanos a 0.0
        is_green = (g > (1.0 - tolerance)) & (r < tolerance) & (b < tolerance)
        mask = is_green.to(dtype=image.dtype)

        # Imagen limpia (mantiene todo igual pero permite pasarla al flujo)
        return (mask, image)
