from http.server import BaseHTTPRequestHandler
import json
import pickle
import os
import tempfile

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Content-Type": "application/json"
}

FEATURES = [
    "Puntos (Est)", "es_Marti", "es_Fede", "es_Mati", "es_Joaco", "es_Valen", "es_Obra",
    "es_Modelado", "es_Gestion", "es_Documentacion", "es_Investigacion", "es_Buffer",
    "es_Revit", "es_AutoCAD", "es_SketchUp", "es_Rhino",
    "es_Vivienda", "es_Industria", "es_Espacio Público", "es_Hoteleria", "es_Educacional",
    "es_Salud", "es_Incertidumbre", "es_Complejidad"
]

PERSONA_MAP = {"Marti": "es_Marti", "Fede": "es_Fede", "Mati": "es_Mati", "Joaco": "es_Joaco", "Valen": "es_Valen"}
TIPO_MAP = {"Obra": "es_Obra", "Modelado": "es_Modelado", "Gestion": "es_Gestion", "Documentacion": "es_Documentacion", "Investigacion": "es_Investigacion"}
SOFTWARE_MAP = {"Revit": "es_Revit", "AutoCAD": "es_AutoCAD", "SketchUp": "es_SketchUp", "Rhino": "es_Rhino"}
CATEGORIA_MAP = {"Vivienda": "es_Vivienda", "Industria": "es_Industria", "Espacio Público": "es_Espacio Público", "Hoteleria": "es_Hoteleria", "Educacional": "es_Educacional", "Salud": "es_Salud"}

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        try:
            import pandas as pd

            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))

            # Construir vector de features
            tarea = {col: 0 for col in FEATURES}
            tarea["Puntos (Est)"] = float(body.get("puntos_est", 0))

            persona = body.get("persona", "")
            if persona in PERSONA_MAP:
                tarea[PERSONA_MAP[persona]] = 1

            tipo = body.get("tipo_tarea", "")
            if tipo in TIPO_MAP:
                tarea[TIPO_MAP[tipo]] = 1

            sw = body.get("software", "")
            if sw in SOFTWARE_MAP:
                tarea[SOFTWARE_MAP[sw]] = 1

            cat = body.get("categoria", "")
            if cat in CATEGORIA_MAP:
                tarea[CATEGORIA_MAP[cat]] = 1

            if body.get("es_buffer"):
                tarea["es_Buffer"] = 1
            if body.get("es_incertidumbre"):
                tarea["es_Incertidumbre"] = 1
            if body.get("es_complejidad"):
                tarea["es_Complejidad"] = 1

            # Cargar modelo
            model_path = os.path.join(tempfile.gettempdir(), "modelo_rf.pkl")
            if not os.path.exists(model_path):
                # Si no hay modelo, entrenar primero
                self.send_response(400)
                for k, v in CORS_HEADERS.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Modelo no entrenado. Llamá a /api/train primero."}).encode())
                return

            with open(model_path, 'rb') as f:
                modelo = pickle.load(f)

            df_input = pd.DataFrame([tarea])
            prediccion = float(modelo.predict(df_input)[0])

            self.send_response(200)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps({"prediccion": round(prediccion, 2)}).encode())

        except Exception as e:
            self.send_response(500)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
