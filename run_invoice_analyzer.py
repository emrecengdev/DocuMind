import subprocess
import sys
import os

def check_and_install_streamlit():
    """Streamlit paketini kontrol et ve yoksa yükle"""
    try:
        import streamlit
        print("Streamlit paketi kurulu. Versiyonu:", streamlit.__version__)
        return True
    except ImportError:
        print("Streamlit paketi kurulu değil. Yükleniyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("Streamlit başarıyla yüklendi.")
            return True
        except Exception as e:
            print(f"Streamlit yüklenirken hata oluştu: {str(e)}")
            return False

def main():
    """Ana fonksiyon - Uygulamayı başlatır"""
    print("Fatura Analiz Sistemi başlatılıyor...")
    
    # Streamlit'i kontrol et
    if not check_and_install_streamlit():
        print("Streamlit kurulu olmadığı için uygulama başlatılamıyor.")
        print("Manuel olarak şu komutu çalıştırın: pip install streamlit")
        return
    
    # Belge şablonları klasörünü oluştur
    doc_templates_dir = "document_templates"
    if not os.path.exists(doc_templates_dir):
        os.makedirs(doc_templates_dir)
        print(f"'{doc_templates_dir}' klasörü oluşturuldu.")
    
    # Uygulamayı çalıştır
    app_file = "invoice_analyzer_app.py"
    if not os.path.exists(app_file):
        print(f"Hata: {app_file} dosyası bulunamadı.")
        return
    
    print(f"'{app_file}' çalıştırılıyor...")
    subprocess.run(["streamlit", "run", app_file])

if __name__ == "__main__":
    main() 