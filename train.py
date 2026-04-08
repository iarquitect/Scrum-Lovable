from http.server import BaseHTTPRequestHandler
import json
import pickle
import os
import tempfile

def get_model_path():
    return os.path.join(tempfile.gettempdir(), "modelo_rf.pkl")

def get_metrics_path():
    return os.path.join(tempfile.gettempdir(), "metrics.json")

SHEET_ID = "1gqb65zbZzcZCTiua5AlXFzANr8jXpNtTLWKYqFUDoeo"
SHEET_NAME = "DB_ML_Export"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

FEATURES = [
    "Puntos (Est)", "es_Marti", "es_Fede", "es_Mati", "es_Joaco", "es_Valen", "es_Obra",
    "es_Modelado", "es_Gestion", "es_Documentacion", "es_Investigacion", "es_Buffer",
    "es_Revit", "es_AutoCAD", "es_SketchUp", "es_Rhino",
    "es_Vivienda", "es_Industria", "es_Espacio Público", "es_Hoteleria", "es_Educacional",
    "es_Salud", "es_Incertidumbre", "es_Complejidad"
]

EQUIPO = ['es_Marti', 'es_Fede', 'es_Mati', 'es_Joaco', 'es_Valen']

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Content-Type": "application/json"
}

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        return self.do_GET()

    def do_GET(self):
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, r2_score

            # 1. Cargar datos desde Google Sheets
            df = pd.read_csv(SHEET_URL)
            df = df.dropna(subset=['Horas (Real) [Y]', 'Puntos (Est)'])

            # 2. Preparar features
            X = df[FEATURES].fillna(0)
            y = df["Horas (Real) [Y]"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 3. Entrenar modelo
            modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            modelo_rf.fit(X_train, y_train)

            predicciones = modelo_rf.predict(X_test)
            r2 = r2_score(y_test, predicciones)
            mae = mean_absolute_error(y_test, predicciones)

            # 4. Feature importance
            importances = dict(zip(FEATURES, modelo_rf.feature_importances_.tolist()))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

            # 5. Métricas de calibración
            df['Error_Estimacion'] = df['Horas (Real) [Y]'] - df['Puntos (Est)']
            df_planificado = df[df['es_Buffer'] == 0].copy()

            tasa_global = float(df_planificado['Horas (Real) [Y]'].sum() / df_planificado['Puntos (Est)'].sum())

            calibracion = []
            for persona in EQUIPO:
                df_persona = df_planificado[df_planificado[persona] == 1]
                if len(df_persona) > 0:
                    tasa = float(df_persona['Horas (Real) [Y]'].sum() / df_persona['Puntos (Est)'].sum())
                    calibracion.append({"nombre": persona.replace("es_", ""), "tasa": round(tasa, 2)})

            # 6. Capacity
            horas_totales = float(df['Horas (Real) [Y]'].sum())
            horas_urgencias = float(df[df['es_Buffer'] == 1]['Horas (Real) [Y]'].sum())
            capacity_libre = round(100 - (horas_urgencias / horas_totales) * 100, 1)

            # 7. Context switching
            switching = []
            for col in EQUIPO:
                nombre = col.replace('es_', '')
                df_m = df[df[col] == 1].copy()
                df_m['sufrio_interrupcion'] = df_m['es_Buffer'].shift(-1).fillna(0)
                df_norm = df_m[df_m['es_Buffer'] == 0]
                for estado_val, estado_label in [(0.0, 'Sin Interrupción'), (1.0, 'Interrumpida')]:
                    grupo = df_norm[df_norm['sufrio_interrupcion'] == estado_val]
                    if len(grupo) > 0:
                        switching.append({
                            "arquitecto": nombre,
                            "estado": estado_label,
                            "tareas": int(len(grupo)),
                            "errorPromedio": round(float(grupo['Error_Estimacion'].mean()), 2)
                        })

            # 8. Comparación humano vs IA (hold-out)
            df_prueba = df.sample(n=min(15, len(df)), random_state=42)
            df_train_comp = df.drop(df_prueba.index)
            X_tr = df_train_comp[FEATURES].fillna(0)
            y_tr = df_train_comp["Horas (Real) [Y]"]
            modelo_comp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            modelo_comp.fit(X_tr, y_tr)
            pred_comp = modelo_comp.predict(df_prueba[FEATURES].fillna(0))

            comparacion = []
            for i, (_, row) in enumerate(df_prueba.iterrows()):
                comparacion.append({
                    "estimacion": float(row['Puntos (Est)']),
                    "prediccionIA": round(float(pred_comp[i]), 2),
                    "realidad": float(row['Horas (Real) [Y]'])
                })

            error_humano = round(float(np.abs(df_prueba['Horas (Real) [Y]'].values - df_prueba['Puntos (Est)'].values).mean()), 2)
            error_ia = round(float(np.abs(df_prueba['Horas (Real) [Y]'].values - pred_comp).mean()), 2)

            # 9. Guardar modelo
            with open(get_model_path(), 'wb') as f:
                pickle.dump(modelo_rf, f)

            # 10. Guardar métricas
            result = {
                "r2": round(r2, 2),
                "mae": round(mae, 2),
                "tasa_global": round(tasa_global, 2),
                "capacity_libre": capacity_libre,
                "calibracion": calibracion,
                "switching": switching,
                "comparacion": comparacion,
                "error_humano": error_humano,
                "error_ia": error_ia,
                "feature_importance": [{"name": k, "importance": round(v, 4)} for k, v in top_features],
                "registros": len(df),
            }

            with open(get_metrics_path(), 'w') as f:
                json.dump(result, f)

            self.send_response(200)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

