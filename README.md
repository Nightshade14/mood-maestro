# mood-maestro
Personalized and real-time generative playlist curation with AI Agents

## Running the API

Start the FastAPI server (uvicorn is required, it's included in `fastapi[all]`):

```bash
python -m uvicorn mood_maestro.main:app --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl "http://127.0.0.1:8000/getPlaylist?user=alice"
```

The endpoint returns a JSON playlist for the provided `user` query parameter.
