"""
Interactive camera selector
Helps identify which camera is which by showing previews.
Max 350 lines.
"""
import cv2
import yaml
import logging
from camera_utils import enumerate_cameras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_camera():
    """Interactive camera selection with preview."""
    
    print("\n" + "="*60)
    print("🎥 SELECTOR DE CÁMARA")
    print("="*60)
    
    cameras = enumerate_cameras()
    
    if not cameras:
        print("❌ No se encontraron cámaras")
        return None
    
    print(f"\n📹 Cámaras disponibles: {len(cameras)}")
    for i, cam in enumerate(cameras):
        print(f"  {i+1}. Cámara {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
    
    print("\n" + "-"*60)
    print("Presiona el número de la cámara que quieres usar")
    print("(o 'q' para salir sin cambiar)")
    print("-"*60)
    
    while True:
        choice = input("\nSelecciona cámara (1-{}): ".format(len(cameras))).strip().lower()
        
        if choice == 'q':
            print("Cancelado")
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cameras):
                selected = cameras[idx]
                camera_index = selected['index']
                
                print(f"\n✅ Mostrando preview de cámara {camera_index}...")
                print("   Presiona 'y' para confirmar, 'n' para probar otra, 'q' para salir")
                
                # Show preview
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print(f"❌ No se pudo abrir cámara {camera_index}")
                    continue
                
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, selected['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, selected['height'])
                
                preview_time = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Add text overlay
                    cv2.putText(
                        frame, 
                        f"Camera {camera_index} - {selected['width']}x{selected['height']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        "Press Y to confirm, N for another, Q to quit",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    cv2.imshow('Camera Preview - Y=Confirm, N=Another, Q=Quit', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('y') or key == ord('Y'):
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # Update config
                        print(f"\n✅ Cámara {camera_index} seleccionada")
                        print("Actualizando config.yaml...")
                        
                        with open("config.yaml", "r") as f:
                            config = yaml.safe_load(f)
                        
                        config['camera']['index'] = camera_index
                        
                        with open("config.yaml", "w") as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                        print(f"✅ config.yaml actualizado: camera.index = {camera_index}")
                        return camera_index
                        
                    elif key == ord('n') or key == ord('N'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                        
                    elif key == ord('q') or key == ord('Q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Cancelado")
                        return None
                    
                    preview_time += 1
                    if preview_time > 300:  # ~10 seconds at 30fps
                        print("\n⏱️  Timeout - presiona una tecla en la ventana")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"❌ Número inválido. Usa 1-{len(cameras)}")
        except ValueError:
            print("❌ Entrada inválida. Usa un número o 'q'")
        except KeyboardInterrupt:
            print("\n\nCancelado")
            return None


if __name__ == "__main__":
    select_camera()

