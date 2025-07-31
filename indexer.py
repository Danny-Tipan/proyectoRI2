# indexer.py
import os
import numpy as np
import faiss
import pandas as pd # Importar pandas para leer CSV
from encoder import MultimodalEncoder # Importar la clase MultimodalEncoder

def build_index(image_folder, annotations_file, encoder, index_path="faiss_index.bin", metadata_path="metadata.npy", limit=1000):
    image_paths = []
    image_ids = []
    descriptions = []

    print(f"Cargando anotaciones desde {annotations_file}...")
    # Leer el archivo CSV usando pandas, asumiendo el formato 'image_name|comment_number|comment'
    df = pd.read_csv(annotations_file, sep='|', header=None, names=['image_name', 'comment_number', 'comment'])

    processed_images_count = 0
    unique_images_processed = set()

    # Itera sobre las filas del DataFrame
    for index, row in df.iterrows():
        # Si el límite está activo y ya hemos procesado suficientes imágenes únicas, salir
        if limit and len(unique_images_processed) >= limit:
            break

        image_filename_raw = str(row['image_name']).strip() # Asegurarse de que sea string y limpiar espacios
        desc = str(row['comment']).strip() # Asegurarse de que sea string y limpiar espacios

        image_id = image_filename_raw # Para Flickr30k, image_name ya es el ID completo del archivo

        full_image_path = os.path.join(image_folder, image_id)

        # Verificar si el archivo de imagen existe. Intentar con .jpg si no se especifica.
        if not os.path.exists(full_image_path):
            if not image_id.lower().endswith('.jpg'):
                full_image_path = os.path.join(image_folder, image_id + '.jpg')
            if not os.path.exists(full_image_path):
                # print(f"Archivo de imagen no encontrado para: {image_id}. Saltando entrada.")
                continue # Saltar esta entrada si la imagen no existe

        # Solo agregar si no hemos alcanzado el límite de imágenes únicas
        if limit and image_id not in unique_images_processed and len(unique_images_processed) >= limit:
            continue

        image_paths.append(full_image_path)
        image_ids.append(image_id)
        descriptions.append(desc)
        unique_images_processed.add(image_id)
        processed_images_count += 1

    print(f"Procesando {len(unique_images_processed)} imágenes únicas y {len(descriptions)} descripciones.")
    if len(descriptions) == 0:
        print("Advertencia: No se encontraron descripciones válidas o imágenes. Verifique las rutas y el formato del CSV.")
        return # Salir si no hay datos para procesar

    all_embeddings = []
    metadata = [] # Para almacenar image_path, image_id y description

    # Codificar descripciones y asociarlas con las rutas de las imágenes
    for i, (img_path, img_id, desc) in enumerate(zip(image_paths, image_ids, descriptions)):
        if i % 500 == 0: # Imprimir progreso cada 500 descripciones
            print(f"Codificando descripción {i}/{len(descriptions)}")
        text_embedding = encoder.encode_text(desc)
        all_embeddings.append(text_embedding.flatten()) # Aplanar a 1D para FAISS
        metadata.append({"image_path": img_path, "image_id": img_id, "description": desc})

    embeddings_array = np.array(all_embeddings).astype('float32')

    if len(embeddings_array) == 0:
        print("No se generaron embeddings. Verifique las rutas de sus datos y límites.")
        return

    print(f"Construyendo índice FAISS con {embeddings_array.shape[0]} embeddings de dimensión {embeddings_array.shape[1]}")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension) # Usar IndexFlatL2 para búsqueda de vecinos más cercanos (distancia euclidiana)
    index.add(embeddings_array)

    # Guardar el índice y los metadatos
    faiss.write_index(index, index_path)
    np.save(metadata_path, metadata)

    print(f"Índice FAISS guardado en {index_path}")
    print(f"Metadatos guardados en {metadata_path}")
    print(f"Imágenes únicas procesadas: {len(unique_images_processed)}")


if __name__ == '__main__':
    # --- ¡AJUSTA ESTAS RUTAS PARA TU CONFIGURACIÓN DE DATOS DE FLICKR30K! ---
    # La carpeta que contiene las imágenes (ej. 1000092795.jpg)
    IMAGE_FOLDER = "data/flickr30k_images"
    # Tu archivo results.csv
    ANNOTATIONS_FILE = "data/flickr30k/results.csv"

    # Verificaciones para asegurar que las rutas existen
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: La carpeta de imágenes no se encontró en {IMAGE_FOLDER}. Por favor, ajusta la ruta.")
        exit()
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: El archivo de anotaciones no se encontró en {ANNOTATIONS_FILE}. Por favor, ajusta la ruta.")
        exit()

    # Inicializa el encoder
    encoder = MultimodalEncoder()
    # Puedes ajustar el 'limit' para procesar más o menos imágenes.
    # Flickr30k es un dataset grande, 1000 imágenes únicas es un buen punto de partida para pruebas.
    build_index(IMAGE_FOLDER, ANNOTATIONS_FILE, encoder, limit=4000)