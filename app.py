# app.py
from flask import Flask, render_template, request, jsonify
from retriever import Retriever # Importa la clase Retriever
from generator import TextGenerator # Importa la clase TextGenerator
import os
import base64 # Para codificar imágenes a base64 para HTML

app = Flask(__name__)

# Inicializa el retriever y el generador una única vez al inicio de la aplicación
# Esto es importante para no recargar los modelos con cada solicitud.
try:
    retriever = Retriever()
    generator = TextGenerator()
except Exception as e:
    print(f"Error al inicializar Retriever o TextGenerator: {e}")
    print("Asegúrate de haber ejecutado 'indexer.py' para crear 'faiss_index.bin' y 'metadata.npy'.")
    retriever = None # Establecer a None si la inicialización falla
    generator = None # Establecer a None si la inicialización falla


@app.route('/')
def index():
    # Renderiza la plantilla HTML principal
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Verifica que el sistema esté inicializado correctamente
    if not retriever or not generator:
        return jsonify({"error": "El sistema no está inicializado. Por favor, revisa los logs del servidor."}), 500

    query_type = request.form.get('query_type') # Obtiene el tipo de consulta (texto o imagen)
    results = []
    generated_response = ""
    retrieved_descriptions = []

    if query_type == 'text':
        query_text = request.form.get('query_text')
        if query_text:
            results = retriever.retrieve_by_text(query_text)
            # Extrae solo las descripciones para pasarlas al generador
            retrieved_descriptions = [res['description'] for res in results]
            if retrieved_descriptions:
                generated_response = generator.generate_response(query_text, retrieved_descriptions)
            else:
                generated_response = "No se encontró información relevante para generar una respuesta."
        else:
            return jsonify({"error": "La consulta de texto no puede estar vacía."}), 400

    elif query_type == 'image':
        # Verifica si se subió un archivo de imagen
        if 'query_image' not in request.files:
            return jsonify({"error": "No se encontró parte de imagen en la solicitud."}), 400
        file = request.files['query_image']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ninguna imagen."}), 400
        if file:
            # Guarda la imagen subida temporalmente para que el retriever pueda leerla
            temp_image_path = "temp_query_image.jpg"
            file.save(temp_image_path)

            try:
                # Realiza la recuperación por imagen
                results = retriever.retrieve_by_image(temp_image_path)
                retrieved_descriptions = [res['description'] for res in results]
                # Para la generación, podemos usar una consulta genérica para imágenes
                if retrieved_descriptions:
                    generated_response = generator.generate_response("an image query", retrieved_descriptions)
                else:
                    generated_response = "No se encontró información relevante para generar una respuesta a partir de la imagen."
            finally:
                # Limpia: elimina la imagen temporal después de usarla
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    else:
        return jsonify({"error": "Tipo de consulta no válido."}), 400

    # Prepara los resultados para enviarlos al HTML:
    # Convierte las imágenes a base64 para poder mostrarlas directamente en el navegador.
    display_results = []
    for res in results:
        image_path = res['image_path']
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                display_results.append({
                    "description": res['description'],
                    "image_b64": encoded_string, # Imagen codificada en base64
                    "distance": f"{res['distance']:.4f}" # Formatear distancia
                })
        except FileNotFoundError:
            print(f"Advertencia: Imagen no encontrada para mostrar: {image_path}")
            display_results.append({
                "description": res['description'],
                "image_b64": "", # No hay imagen para mostrar
                "distance": f"{res['distance']:.4f}"
            })
        except Exception as e:
            print(f"Error al codificar imagen {image_path}: {e}")
            display_results.append({
                "description": res['description'],
                "image_b64": "",
                "distance": f"{res['distance']:.4f}"
            })

    # Devuelve los resultados y la respuesta generada como JSON
    return jsonify({
        "results": display_results,
        "generated_response": generated_response
    })

if __name__ == '__main__':
    app.run(debug=True) # `debug=True` habilita el modo de depuración (recarga automática)