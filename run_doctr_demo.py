import os
import sys
import subprocess

# OCR-service klasörünü Python path'ine ekle
ocr_service_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr-service")
sys.path.append(ocr_service_path)

print("Doctr demo uygulaması başlatılıyor...")
print(f"OCR servis yolu: {ocr_service_path}")

# Demo uygulamasını çalıştır
demo_path = os.path.join(ocr_service_path, "demo", "app.py")
subprocess.run(["streamlit", "run", demo_path]) 