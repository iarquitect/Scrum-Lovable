from http.server import BaseHTTPRequestHandler
import json
import os
import tempfile

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Content-Type": "application/json"
}

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        metrics_path = os.path.join(tempfile.gettempdir(), "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            self.send_response(200)
        else:
            data = {"error": "No hay métricas. Llamá a /api/train primero."}
            self.send_response(404)

        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
        from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

