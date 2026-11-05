# backend/infer

Stateless CPU move server: ONNX Runtime + `packages/doppel_model`. Replaces `demo/server.py`.

- Loads every model package under `$DOPPEL_MODELS` (default `models/`, produced by `python -m doppel_lab.export_package`).
- `GET /api/bots`, `POST /api/move {bot, elo, opp_elo?, moves, temperature}`, `GET /api/health`.
- No torch. ChessMimic is never served (PolyForm Noncommercial, lab only).

Stage 0 (bigbox, behind `tailscale serve` on :8460):

    DOPPEL_STATIC=demo/static .venv/bin/uvicorn doppel_infer.app:app --host 127.0.0.1 --port 8612
