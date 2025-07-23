# Troof

Proyecto de la asignatura **Procesamiento digital de imágenes** 2025-I. Esta versión está destinada para ser desplegada en sistemas embebidos y realizar inferencias ligeras.

## Instalación:

Ejecuta el siguiente comando

```bash
git clone https://github.com/J-Guevara12/troof
cd troof
virtualenv venv
source venv/bin/activate
pip install opencv-python numpy ai-edge-litert
```
## Pruebalo tú mismo:

Puedes descargar cualquier imagen satelital y pasarla al módulo:

```bash
python inference.py {YOUR_IMAGE}
```

He acá un ejemplo

```bash
curl https://project.inria.fr/aerialimagelabeling/files/2011/12/vie1.jpg > image.jpg
python inference.py image.jpg
```
