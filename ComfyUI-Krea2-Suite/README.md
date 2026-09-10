# ComfyUI Krea 2 Suite 🎨✨

Paquete de **Nodos Personalizados para ComfyUI** diseñado para **Krea 2 (Turbo y Raw)**, permitiendo **Inpainting de ultra alta calidad**, **Outpainting con preservación estricta de Aspect Ratio y Resolución Nativa**, extracción de máscaras por **Chroma Green (#00FF00)**, y compatibilidad con LoRAs de edición como **INPAINTKREA**, **AnyPaint** y **Ostris Edit**.

---

## 📦 Instalación

1. Clona o copia esta carpeta dentro del directorio `custom_nodes` de tu instalación de ComfyUI:
   ```bash
   cd ComfyUI/custom_nodes/
   # Si copias la carpeta:
   cp -r /ruta/a/ComfyUI-Krea2-Suite ./
   ```
2. Instala las dependencias si no las tienes (PyTorch >= 2.0):
   ```bash
   pip install -r ComfyUI-Krea2-Suite/requirements.txt
   ```
3. Reinicia ComfyUI.

---

## 🚀 Nodos Incluidos

### 1. `Krea 2 Aspect-Preserve Prepare (Lienzo & Bucket)`
- **Función:** Ajusta dinámicamente cualquier imagen de entrada al **resolution bucket** nativo más adecuado de Krea 2 (`672x1568` a `1568x672`) sin deformar ni estirar la relación de aspecto.
- Aplica escalado uniforme y padding replicado centrado.
- Genera el lienzo coloreado con **verde neón (#00FF00)** si se usa el LoRA de inpaint de Krea 2.
- Emite un objeto de metadatos `restore_map` para restaurar con precisión de píxel 1:1.

### 2. `Krea 2 Aspect-Preserve Restore (Remapeo & Seam Blend)`
- **Función:** Remapea la imagen generada por Krea 2 de vuelta a las dimensiones y proporciones exactas del original mediante `torch.nn.functional.grid_sample` (interpolación bicúbica inversa).
- **Local Color Match:** Ajusta automáticamente la media y desviación estándar de color en un anillo perimetral alrededor de la máscara para evitar saltos tonales.
- **Seam Overlap & Feathering:** Aplica dilatación morfológica y desenfoque gaussiano para eliminar costuras o bordes visibles.
- **CUDA Synchronization:** Previene corrupción de tensores o bandas de colores en GPU.

### 3. `Krea 2 Outpaint Canvas Expander (Expansión Direccional)`
- **Función:** Permite expandir el lienzo en píxeles hacia la izquierda, derecha, arriba y abajo (`pad_left`, `pad_right`, `pad_top`, `pad_bottom`).
- Genera automáticamente el canvas expandido y la máscara binaria exacta de outpainting.

### 4. `Krea 2 Inpaint Prompt Builder (Formato LoRA)`
- **Función:** Construye el prompt exacto optimizado para el LoRA `INPAINTKREA` / `krea2-anypaint`:
  > *"Replace the solid neon green area with [TU_OBJETO], match the surrounding lighting, perspective, shadows, colors, texture, and depth of field, preserve everything outside the solid neon green area."*

### 5. `Krea 2 Chroma Green Mask Extractor (#00FF00)`
- **Función:** Permite pintar directamente con el pincel verde sobre la imagen en ComfyUI y extrae automáticamente la máscara binaria para alimentar el flujo de inpaint.

### 6. `Krea 2 Reference Model Patcher (KV-Cache & DiT)`
- **Función:** Parchea el modelo DiT SingleStream de Krea 2 para soportar atención a imágenes de referencia en $t=0$ y reutilización de KV-Cache aislado.

### 7. `Krea 2 AnyPaint Latent Blender (Flow-Matching Blend)`
- **Función:** Aplica la ecuación de Flow-Matching para preservar los latentes conocidos:
  $$\text{known\_at\_sigma} = \sigma \cdot \text{initial\_noise} + (1 - \sigma) \cdot \text{known\_latent}$$

---

## 🛠️ Flujos de Trabajo Típicos

### A. Inpainting con Chroma Green (#00FF00) + INPAINTKREA LoRA
```text
LoadImage (con máscara o pintada en verde)
   ↓
Krea 2 Aspect-Preserve Prepare
   ↓ (prepared_image, bucket_width, bucket_height)
VAE Encode -> KSampler (Krea 2 Turbo + INPAINTKREA LoRA, 8 steps, CFG 1.0)
   ↓
VAE Decode
   ↓ (generated_image)
Krea 2 Aspect-Preserve Restore (recibe original_image, generated_image, mask, restore_map)
   ↓
SaveImage (Imagen final a resolución 100% nativa sin costuras)
```

### B. Outpainting Direccional
```text
LoadImage
   ↓
Krea 2 Outpaint Canvas Expander (pad_left=300, pad_right=300, fill="pure_green")
   ↓ (expanded_image, outpaint_mask)
Krea 2 Aspect-Preserve Prepare
   ↓
KSampler (Krea 2 Turbo + LoRA) -> VAE Decode
   ↓
Krea 2 Aspect-Preserve Restore
   ↓
SaveImage
```

---

## 📄 Licencia
MIT License. Desarrollado para la comunidad de ComfyUI y creadores con Krea 2.
