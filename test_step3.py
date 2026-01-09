#!/usr/bin/env python3
import sys
import os

# Escribir a stderr también
print("STEP 3: Script iniciado", file=sys.stderr, flush=True)
print("STEP 3: Script iniciado", file=sys.stdout, flush=True)

# Verificar que estamos en el directorio correcto
print(f"Current dir: {os.getcwd()}", file=sys.stderr, flush=True)
print(f"Python: {sys.executable}", file=sys.stderr, flush=True)
print(f"Version: {sys.version}", file=sys.stderr, flush=True)

# Forzar salida
sys.stdout.flush()
sys.stderr.flush()

print("STEP 3: Script terminado", flush=True)
