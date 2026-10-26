---
tags: [ml, serialization, file-extensions, conventions]
created: 2026-10-26
related: [[Serialización - Concepto]], [[Módulos de serialización]], [[Serializar un Pipeline]]
---

# Extensiones de archivo

## La verdad fundamental

**La extensión no cambia el contenido.** Python no mira la extensión; mira los bytes del archivo.

Podés renombrar `pipeline.bin` a `pipeline.pkl` y `pickle.load` lo carga igual:

```bash
cp pipeline.bin pipeline.pkl
python -c "import pickle; print(pickle.load(open('pipeline.pkl','rb')))"
```

La extensión es solo una **etiqueta para humanos**.

## Extensiones comunes

| Extensión | Significado convencional | Librería típica |
|---|---|---|
| `.pkl` | "Esto es un pickle" | `pickle` |
| `.bin` | "Archivo binario genérico" | cualquiera |
| `.joblib` | "Serializado con joblib" | `joblib` |
| `.sav` | Usado por scikit-learn/SPSS en algunos contextos | varía |

## Por qué el homework usa `.bin`

Es una convención del curso (y de muchos deployments) para indicar **"artifact binario congelado"**, sin atar el nombre a una librería específica. El `Dockerfile` y `predict.py` solo referencian el nombre del archivo, no la extensión.

## Lo que realmente determina si un archivo se puede cargar

No es la extensión, sino:

1. **Qué librería lo serializó** (`pickle` vs `joblib` vs `cloudpickle`).
2. **Qué versión** de esa librería y de scikit-learn/NumPy había al serializar.
3. **Qué objeto** contiene (un `Pipeline`, un `LogisticRegression` suelto, etc.).

## Regla práctica

Usá la extensión que prefieras, pero sé **consistente** en tu proyecto. Si cambiás la extensión, actualizá las referencias en `predict.py`, `Dockerfile` y scripts. El checksum SHA256 no cambia porque el contenido no cambia.

## Ver también

- [[Serialización - Concepto]]
- [[Módulos de serialización]]
- [[Serializar un Pipeline]]
