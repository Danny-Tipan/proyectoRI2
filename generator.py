# generator.py
import google.generativeai as genai
import os
from dotenv import load_dotenv # Si usas .env, sino puedes quitar estas dos líneas

class TextGenerator:
    def __init__(self):

        # --- Carga la clave API ---
        # Si estás usando el archivo .env (recomendado):
        load_dotenv() # Carga las variables del archivo .env
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        # Si no usas .env y la has hardcodeado (MENOS SEGURO):
        if not GOOGLE_API_KEY: # Si no se cargó de .env, la asigna directamente
            GOOGLE_API_KEY = "AIzaSyAggMkVW2px2MKCrGRWkMHllN75M24Qx4M" # <-- ¡TU CLAVE AQUÍ!
        # --- Fin de carga de clave API ---

        if not GOOGLE_API_KEY:
            raise ValueError("La clave API de Google no se encontró. Asegúrate de que esté configurada como variable de entorno o en un archivo .env, o hardcodeada si es solo para pruebas.")

        # --- CAMBIO IMPORTANTE AQUÍ EN LA INICIALIZACIÓN ---
        # Configura la API de forma global.
        genai.configure(api_key=GOOGLE_API_KEY)

        # Selecciona el modelo que quieres usar
        # Usamos "gemini-2.5-flash" como indicaste
        self.model_name = "gemini-2.5-flash"
        # Inicializa el modelo generativo directamente con genai.GenerativeModel
        self.model = genai.GenerativeModel(self.model_name)

        print(f"Modelo Gemini '{self.model_name}' inicializado.")

    def generate_response(self, query, retrieved_descriptions):
        # Concatena las descripciones recuperadas para formar un contexto.
        # Filtra descripciones vacías.
        context = "Contexto: " + " ".join([desc for desc in retrieved_descriptions if desc])

        # Construye el prompt para el modelo generativo.
        # Hago el prompt más robusto para guiar mejor a Gemini.
        prompt_content = f"""
        Eres un asistente útil y objetivo. Tu tarea es generar una respuesta descriptiva o informativa.
        La respuesta debe ser **estrictamente basada en la información proporcionada en el contexto**.
        No agregues información nueva, opiniones, inferencias, o interpretaciones personales.
        Sé conciso y directo al punto. Si la información no está en el contexto, indica que no la tienes.

        ---
        Consulta: {query}

        Contexto:
        {context}
        ---

        Respuesta:
        """

        try:
            # Llama al modelo para generar contenido
            response = self.model.generate_content(
                contents=[{"parts": [{"text": prompt_content}]}], # Formato de contenidos para generate_content
                generation_config={
                    "temperature": 0.7,      # Controla la aleatoriedad de la respuesta (0.0 = más determinista, 1.0 = más creativo)
                    "max_output_tokens": 250 # Número máximo de tokens en la respuesta
                },
                # --- ¡ESTA ES LA SECCIÓN QUE TE FALTABA O ESTABA COMENTADA! ---
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                # -----------------------------------------------------------
            )
            # Verifica si la respuesta contiene contenido antes de intentar acceder a .text
            if response.parts:
                return response.text
            else:
                # Si no hay partes pero no hubo una excepción, es un bloqueo de seguridad
                finish_reason = response.candidates[0].finish_reason if response.candidates else "Desconocido"
                safety_details = []
                if response.candidates and response.candidates[0].safety_ratings:
                    for rating in response.candidates[0].safety_ratings:
                        safety_details.append(f"{rating.category.name}: {rating.probability.name}")
                print(f"Respuesta bloqueada por seguridad. Razón: {finish_reason}. Detalles: {', '.join(safety_details) if safety_details else 'N/A'}")
                return "Lo siento, la respuesta fue bloqueada por los filtros de seguridad del modelo."

        except Exception as e:
            print(f"Error al generar respuesta con Gemini: {e}")
            return "Lo siento, no pude generar una respuesta en este momento. Hubo un error con el modelo de IA."

if __name__ == '__main__':
    # Pequeña prueba para verificar el generador con Gemini
    generator = TextGenerator()
    query_example = "What activities are happening based on these images?"
    retrieved_descriptions_example = [
        "A group of people are walking down a sunny street, some carrying bags.",
        "Two young children are playing with a ball in a green field.",
        "A man is cooking food on a grill outdoors."
    ]
    generated_text = generator.generate_response(query_example, retrieved_descriptions_example)
    print(f"\nRespuesta Generada por Gemini con {generator.model_name}:\n{generated_text}")