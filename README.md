# Sprint Predictor - API ML

API de Machine Learning para predecir horas reales de tareas de arquitectura.

## Endpoints

- **POST /api/train** — Entrena el modelo desde Google Sheets y devuelve todas las métricas
- **POST /api/predict** — Predice horas para una nueva tarea
- **GET /api/metrics** — Devuelve las últimas métricas calculadas

## Deploy en Vercel

1. Subí este repo a GitHub
2. Importá el repo en [vercel.com](https://vercel.com) → New Project
3. Deploy automático
4. Copiá la URL y configurala en el dashboard

## Cron

El modelo se reentrena automáticamente cada lunes a las 3am (UTC).
