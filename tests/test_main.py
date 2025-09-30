from fastapi.testclient import TestClient

from mood_maestro.app.main import app

client = TestClient(app)


def test_get_playlist_success():
    resp = client.get("/getPlaylist", params={"user": "alice"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user"] == "alice"
    assert "tracks" in data
    assert isinstance(data["tracks"], list)


def test_get_playlist_missing_user():
    resp = client.get("/getPlaylist")
    assert resp.status_code == 422 or resp.status_code == 400
