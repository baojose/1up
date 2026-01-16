# 🐍 Instalación de Python 3.12.10

## ¿Por qué necesitamos Python 3.12?

PyTorch (requerido por SAM 3) **NO soporta Python 3.14** aún. El proyecto requiere **Python 3.12.10** (última versión estable de 3.12, April 8, 2025).

## 📥 Instalación (NO requiere permisos especiales)

### Opción 1: Descargar desde python.org (Recomendado)

1. **Descargar Python 3.12.10:**
   - Ve a: https://www.python.org/downloads/release/python-31210/
   - **Mac Intel (x86_64)**: Descarga `python-3.12.10-macos11.pkg`
   - **Mac M1/M2 (ARM)**: Descarga `python-3.12.10-macos11-arm64.pkg` 
   - **Enlace directo Mac Intel**: https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg
   - **Enlace directo Mac M1/M2**: https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11-arm64.pkg

2. **Instalar:**
   - Abre el archivo .pkg descargado
   - Sigue el instalador (NO requiere permisos de administrador, se instala en tu usuario)
   - Asegúrate de marcar "Add Python 3.12 to PATH" si aparece la opción

3. **Verificar instalación:**
   ```bash
   python3.12 --version
   # Debería mostrar: Python 3.12.10
   ```

4. **Recrear venv con Python 3.12:**
   ```bash
   cd /Users/jba7790/Desktop/1UP_2
   rm -rf venv
   python3.12 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Opción 2: Usar pyenv (Alternativa)

Si prefieres gestionar múltiples versiones de Python:

```bash
# Instalar pyenv (requiere Homebrew con permisos)
brew install pyenv

# Configurar pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Instalar Python 3.12.10
pyenv install 3.12.10

# Usar Python 3.12.10 en este proyecto
cd /Users/jba7790/Desktop/1UP_2
pyenv local 3.12.10

# Verificar
python --version  # Debería mostrar: Python 3.12.10

# Recrear venv
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🤔 ¿Por qué funcionaba en tu ordenador anterior?

Probablemente tu ordenador anterior tenía Python 3.12 instalado (no 3.14). Este nuevo ordenador solo tiene Python 3.14, que es demasiado nuevo y PyTorch aún no lo soporta.

## ✅ Después de instalar Python 3.12.10

Una vez instalado, ejecuta:

```bash
cd /Users/jba7790/Desktop/1UP_2
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python PyYAML anthropic pillow torch torchvision Flask
```

Luego instala SAM 3:

```bash
cd sam3
pip install -e .
cd ..
```

## 🚀 Después de todo instalado

```bash
export CLAUDE_API_KEY='sk-ant-api03-...'
./run_live_detection_with_claude.sh
```
