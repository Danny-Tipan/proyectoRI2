# retriever.py
import faiss
import numpy as np
from encoder import MultimodalEncoder # Importar la clase MultimodalEncoder
import os

class Retriever:
    def __init__(self, index_path="faiss_index.bin", metadata_path="metadata.npy"):
        self.encoder = MultimodalEncoder() # Inicializa el encoder para codificar las consultas
        try:
            self.index = faiss.read_index(index_path)
            self.metadata = np.load(metadata_path, allow_pickle=True)
            print(f"Índice FAISS cargado con {self.index.ntotal} elementos y {len(self.metadata)} entradas de metadatos.")
        except FileNotFoundError:
            print(f"Error: Archivos de índice/metadatos no encontrados en '{index_path}' o '{metadata_path}'.")
            print("Por favor, asegúrate de haber ejecutado 'indexer.py' primero para crearlos.")
            self.index = None
            self.metadata = None

    def retrieve_by_image(self, image_path, k=5):
        if self.index is None:
            return []
        try:
            query_embedding = self.encoder.encode_image(image_path)
            # Realiza la búsqueda en el índice. `astype('float32')` es importante para FAISS.
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            results = []
            # Iterar sobre los índices de los resultados
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata): # Asegurarse de que el índice es válido
                    item = self.metadata[idx]
                    results.append({
                        "image_path": item["image_path"],
                        "description": item["description"],
                        "distance": distances[0][i] # La distancia calculada por FAISS
                    })
                else:
                    print(f"Advertencia: Índice {idx} fuera de los límites de los metadatos. Skipeando.")
            return results
        except FileNotFoundError:
            print(f"Error: La imagen de consulta no se encontró en {image_path}.")
            return []
        except Exception as e:
            print(f"Error al recuperar por imagen: {e}")
            return []


    def retrieve_by_text(self, query_text, k=5):
        if self.index is None:
            return []
        try:
            query_embedding = self.encoder.encode_text(query_text)
            # Realiza la búsqueda en el índice
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    item = self.metadata[idx]
                    results.append({
                        "image_path": item["image_path"],
                        "description": item["description"],
                        "distance": distances[0][i]
                    })
                else:
                    print(f"Advertencia: Índice {idx} fuera de los límites de los metadatos. Skipeando.")
            return results
        except Exception as e:
            print(f"Error al recuperar por texto: {e}")
            return []

if __name__ == '__main__':
    # Pequeña prueba para verificar el retriever
    retriever = Retriever()

    if retriever.index is not None:
        # Prueba con texto
        text_query = "a man standing on the street"
        print(f"\nRecuperando para consulta de texto: '{text_query}'")
        text_results = retriever.retrieve_by_text(text_query)
        for i, res in enumerate(text_results):
            print(f"  Resultado {i+1}: Descripción: '{res['description']}', Imagen: {os.path.basename(res['image_path'])}, Distancia: {res['distance']:.4f}")

        # Prueba con imagen (asegúrate de que esta ruta exista y sea una imagen real de tu Flickr30k)
        # Ejemplo: Puedes tomar una imagen al azar de tu carpeta data/flickr30k_images
        sample_image_path = "data/flickr30k_images/36979.jpg"# <--- ¡CAMBIA ESTA RUTA POR UNA IMAGEN REAL DE TU FLICKR30K!
        if os.path.exists(sample_image_path):
            print(f"\nRecuperando para consulta de imagen: '{sample_image_path}'")
            image_results = retriever.retrieve_by_image(sample_image_path)
            for i, res in enumerate(image_results):
                print(f"  Resultado {i+1}: Descripción: '{res['description']}', Imagen: {os.path.basename(res['image_path'])}, Distancia: {res['distance']:.4f}")
        else:
            print(f"\nAdvertencia: La ruta de la imagen de prueba no se encontró en {sample_image_path}. Saltando prueba de recuperación de imagen.")
    else:
        print("\nEl Retriever no pudo inicializarse. Por favor, revisa los mensajes de error anteriores.")