"""
ComfyUI-Krea2-Suite
Custom Nodes profesionales para ComfyUI optimizados para Krea 2 Turbo / Raw.
Soporte completo para Inpainting, Outpainting con preservación de resolución y relación de aspecto,
edición por Chroma Green (#00FF00), AnyPaint y LoRAs de edición (Ostris / INPAINTKREA).
"""

from .nodes_inoutpaint import (
    Krea2AspectPreservePrepare,
    Krea2AspectPreserveRestore,
    Krea2OutpaintCanvasExpander
)

from .nodes_prompt_tools import (
    Krea2InpaintPromptBuilder,
    Krea2ChromaGreenMaskExtractor
)

from .nodes_reference import (
    Krea2ReferenceModelPatcher,
    Krea2AnyPaintLatentBlender
)

NODE_CLASS_MAPPINGS = {
    "Krea2AspectPreservePrepare": Krea2AspectPreservePrepare,
    "Krea2AspectPreserveRestore": Krea2AspectPreserveRestore,
    "Krea2OutpaintCanvasExpander": Krea2OutpaintCanvasExpander,
    "Krea2InpaintPromptBuilder": Krea2InpaintPromptBuilder,
    "Krea2ChromaGreenMaskExtractor": Krea2ChromaGreenMaskExtractor,
    "Krea2ReferenceModelPatcher": Krea2ReferenceModelPatcher,
    "Krea2AnyPaintLatentBlender": Krea2AnyPaintLatentBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2AspectPreservePrepare": "Krea 2 Aspect-Preserve Prepare (Lienzo & Bucket)",
    "Krea2AspectPreserveRestore": "Krea 2 Aspect-Preserve Restore (Remapeo & Seam Blend)",
    "Krea2OutpaintCanvasExpander": "Krea 2 Outpaint Canvas Expander (Expansión Direccional)",
    "Krea2InpaintPromptBuilder": "Krea 2 Inpaint Prompt Builder (Formato LoRA)",
    "Krea2ChromaGreenMaskExtractor": "Krea 2 Chroma Green Mask Extractor (#00FF00)",
    "Krea2ReferenceModelPatcher": "Krea 2 Reference Model Patcher (KV-Cache & DiT)",
    "Krea2AnyPaintLatentBlender": "Krea 2 AnyPaint Latent Blender (Flow-Matching Blend)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
