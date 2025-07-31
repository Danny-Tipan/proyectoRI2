# encoder.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class MultimodalEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Determina si usar GPU (cuda) o CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"CLIP model loaded on: {self.device}")

    def encode_image(self, image_path):
        # Carga y preprocesa la imagen
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Genera el embedding de la imagen
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy() # Mueve el tensor a CPU y convierte a NumPy

    def encode_text(self, text):
        # Preprocesa el texto
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Genera el embedding del texto
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy() # Mueve el tensor a CPU y convierte a NumPy

if __name__ == '__main__':
    # Pequeña prueba para verificar el encoder
    encoder = MultimodalEncoder()
    print("Encoder initialized. You can now use its methods.")
    # No se recomienda ejecutar pruebas aquí sin una imagen de prueba real
    # ya que causaría un error si la imagen no existe.