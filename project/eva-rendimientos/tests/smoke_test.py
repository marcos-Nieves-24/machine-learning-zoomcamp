"""Smoke tests de la API (T5). Ejecutar con la API levantada:
    uvicorn api.app:app --app-dir . --port 8000
    python tests/smoke_test.py
"""
import sys
import urllib.request
import json

BASE = "http://127.0.0.1:8000"


def req(method, path, payload=None):
    data = json.dumps(payload).encode() if payload else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return resp.status, json.loads(resp.read())


def main() -> int:
    errors = []
    # health
    status, body = req("GET", "/health")
    assert status == 200 and body["status"] == "ok", ("health", body)
    print("health OK:", body["historial_filas"], "filas,", body["series"], "series")
    # predict con historial largo
    status, body = req("POST", "/predict", {
        "cultivo": "Maíz", "municipio": "Campoalegre",
        "anio": 2026, "periodo": "A", "area_sembrada_ha": 200,
        "departamento": "Huila",
    })
    assert status == 200
    assert body["confianza"] == "alta" and body["hist_len"] >= 50
    assert 0 <= body["rendimiento_predicho_t_ha"] <= 20
    assert body["intervalo_90"][0] <= body["rendimiento_predicho_t_ha"] <= body["intervalo_90"][1]
    print("predict (alta confianza):", body["rendimiento_predicho_t_ha"],
          "t/ha", body["intervalo_90"])
    # predict con serie corta → aviso
    status, body = req("POST", "/predict", {
        "cultivo": "Café", "municipio": "La Unión",
        "anio": 2026, "periodo": "anual", "area_sembrada_ha": 10,
        "departamento": "Antioquia",
    })
    assert status == 200 and body["confianza"] == "baja"
    assert len(body["avisos"]) >= 1
    print("predict (baja confianza):", body["rendimiento_predicho_t_ha"],
          body["confianza"], body["avisos"][0][:40])
    # cultivo no cubierto → 404
    try:
        req("POST", "/predict", {
            "cultivo": "Quinoa", "municipio": "Medellín",
            "anio": 2026, "periodo": "A", "area_sembrada_ha": 10,
        })
        errors.append("cultivo no cubierto debió ser 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404, ("expected 404", e.code)
        print("404 cultivo no cubierto OK")
    # municipio ambiguo sin departamento → 422
    try:
        req("POST", "/predict", {
            "cultivo": "Café", "municipio": "La Unión",
            "anio": 2026, "periodo": "anual", "area_sembrada_ha": 10,
        })
        errors.append("municipio ambiguo sin depto debió ser 422")
    except urllib.error.HTTPError as e:
        assert e.code == 422, ("expected 422", e.code)
        print("422 municipio ambiguo OK")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("\nTodos los smoke tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
