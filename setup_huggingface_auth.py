#!/usr/bin/env python3
"""
Script para autenticar HuggingFace y acceder a SAM 3.
Ejecuta este script después de solicitar acceso al repositorio facebook/sam3.
"""
import os
from pathlib import Path

def check_hf_token():
    """Verifica si hay un token de HuggingFace guardado."""
    token_file = Path.home() / ".huggingface" / "token"
    
    if token_file.exists():
        with open(token_file) as f:
            token = f.read().strip()
            if token:
                print(f"✅ Token encontrado en {token_file}")
                print(f"   Token: {token[:10]}...{token[-4:]}")
                return token
    else:
        print("⚠️  No hay token guardado")
    
    return None

def login_interactive():
    """Autenticación interactiva con HuggingFace."""
    from huggingface_hub import login
    
    print("\n" + "="*60)
    print("🔐 Autenticación HuggingFace para SAM 3")
    print("="*60)
    print("\n⚠️  IMPORTANTE:")
    print("   1. Primero debes solicitar acceso a:")
    print("      https://huggingface.co/facebook/sam3")
    print("   2. Espera a que Meta/Facebook apruebe tu solicitud")
    print("   3. Una vez aprobado, genera un token aquí:")
    print("      https://huggingface.co/settings/tokens")
    print()
    print("   Tipo de token: Read (solo lectura, suficiente)")
    print()
    
    # Verificar si ya hay token
    existing_token = check_hf_token()
    if existing_token:
        print(f"\n💡 Ya hay un token guardado")
        response = input("   ¿Quieres usar el token existente? (s/n): ").strip().lower()
        if response == 's':
            return existing_token
    
    # Solicitar token
    print("\n📋 Pega tu token de HuggingFace (hf_xxxxx):")
    print("   (Puedes obtenerlo en: https://huggingface.co/settings/tokens)")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ Token vacío")
        return None
    
    if not token.startswith('hf_'):
        print("⚠️  El token debería empezar con 'hf_'")
        response = input("   ¿Continuar de todas formas? (s/n): ").strip().lower()
        if response != 's':
            return None
    
    # Intentar login
    try:
        print("\n🔐 Autenticando con HuggingFace...")
        login(token=token, add_to_git_credential=False)
        print("✅ Autenticación exitosa!")
        return token
    except Exception as e:
        print(f"❌ Error al autenticar: {e}")
        print("\n💡 Verifica que:")
        print("   1. El token sea correcto")
        print("   2. Tengas acceso al repositorio facebook/sam3")
        print("   3. Tu solicitud de acceso haya sido aprobada")
        return None

def test_access():
    """Prueba el acceso al repositorio SAM 3."""
    from huggingface_hub import hf_hub_download
    
    print("\n" + "="*60)
    print("🧪 Probando acceso a SAM 3...")
    print("="*60)
    
    try:
        # Intentar descargar el config (archivo pequeño)
        print("\n📥 Intentando descargar config.json (prueba de acceso)...")
        path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="config.json",
            token=None  # Usará el token guardado
        )
        print(f"✅ ¡Acceso exitoso!")
        print(f"   Archivo descargado: {path}")
        return True
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "Unauthorized" in error_str:
            print("❌ Error 401: No autorizado")
            print("\n💡 Esto significa que:")
            print("   1. No estás autenticado, O")
            print("   2. No tienes acceso al repositorio facebook/sam3")
            print("\n📋 Pasos a seguir:")
            print("   1. Solicita acceso: https://huggingface.co/facebook/sam3")
            print("   2. Espera aprobación (puede tomar horas/días)")
            print("   3. Ejecuta este script nuevamente después de ser aprobado")
        elif "403" in error_str or "Forbidden" in error_str:
            print("❌ Error 403: Acceso prohibido")
            print("\n💡 Tu solicitud de acceso puede no haber sido aprobada aún")
            print("   Verifica tu email o el estado en:")
            print("   https://huggingface.co/facebook/sam3")
        else:
            print(f"❌ Error desconocido: {e}")
        return False

def main():
    print("="*60)
    print("🔐 Setup HuggingFace Authentication for SAM 3")
    print("="*60)
    
    # Paso 1: Verificar token existente
    token = check_hf_token()
    
    # Paso 2: Login si no hay token o el usuario quiere cambiar
    if not token:
        token = login_interactive()
        if not token:
            print("\n❌ No se pudo autenticar. Saliendo...")
            return 1
    
    # Paso 3: Probar acceso
    if test_access():
        print("\n" + "="*60)
        print("✅ ¡Todo configurado correctamente!")
        print("="*60)
        print("\n💡 Ahora puedes ejecutar:")
        print("   ./run_live_detection_with_claude.sh")
        return 0
    else:
        print("\n❌ No se pudo verificar el acceso")
        return 1

if __name__ == "__main__":
    exit(main())
