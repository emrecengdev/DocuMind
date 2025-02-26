# DocuMind - Akıllı Belge Analiz Sistemi

Bu proje, gelişmiş OCR teknolojileri ve yapay zeka modelleri kullanarak belgeleri otomatik olarak analiz eden, veri çıkarımı yapan ve yönetebilen akıllı bir sistemdir.

Bu projeye ait detaylı dökümantasyon ve kullanım kılavuzuna aşağıdaki bağlantıdan ulaşabilirsiniz:

👉 [Birleşik Gümrük Müşavirliği Case Çalışması](https://learned-silkworm-839.notion.site/Birle-ik-G-mr-k-M-avirli-i-Case-al-mas-1a6831570df0803fb76ac9a2a788b1ee?pvs=4)


## Özellikler

- **Gelişmiş OCR İşleme**
  - Yüksek doğruluklu OCR
  - Çoklu sayfa desteği
  - Otomatik sayfa yönü algılama
  - Metin segmentasyonu ve görselleştirme

- **Akıllı Belge Tipi Algılama**
  - Otomatik belge tipi tanıma
  - Özelleştirilebilir belge şablonları
  - Benzer belge tiplerini algılama yeteneği

- **LLM Entegrasyonu**
  - Gemini API desteği (varsayılan)
  - Yerel Ollama modeli desteği
  - Özelleştirilebilir veri çıkarım şablonları

- **Kullanıcı Dostu Arayüz**
  - Streamlit tabanlı modern arayüz
  - İnteraktif görselleştirmeler
  - Gerçek zamanlı analiz sonuçları

## Gereksinimler

### Temel Gereksinimler
- Python 3.8+
- CUDA uyumlu GPU (isteğe bağlı)

### Python Paketleri
```bash
# Temel paketler
streamlit>=1.22.0
matplotlib>=3.5.0
opencv-python>=4.5.0
numpy>=1.24.0
python-dotenv>=1.0.0
torch>=2.0.0
Pillow>=10.0.0

# OCR için
python-doctr>=0.6.0

# LLM entegrasyonu için
langchain>=0.1.0
langchain-community>=0.0.13
google-generativeai>=0.3.0
requests>=2.31.0  # Ollama API için
```

## Kurulum

1. **Projeyi Klonlama**
   ```bash
   git clone https://github.com/emrecengdev/DocuMind.git
   cd docmind
   ```

2. **Sanal Ortam Oluşturma**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Gereksinimlerin Kurulumu**
   ```bash
   pip install -r requirements.txt
   ```

4. **Çevre Değişkenlerinin Ayarlanması**
   - Bu case özelinde varsayuılan .env dosyası kullanılabilir
   - `.env.example` dosyasını `.env` olarak kopyalayın
   - Gemini API anahtarınızı ekleyin
   - Diğer ayarları ihtiyacınıza göre düzenleyin

## Kullanım

1. **Uygulamayı Başlatma**
   ```bash
   python run_invoice_analyzer.py
   ```

2. **Belge Analizi**
   - Sol menüden belge yükleyin (PDF/PNG/JPEG)
   - OCR modelini ve parametrelerini seçin
   - LLM modelini seçin (Gemini veya Ollama)
   - "Belgeyi Analiz Et" butonuna tıklayın

3. **Belge Tipleri Yönetimi**
   - "Belge Tipleri Yönetimi" sekmesine geçin
   - Yeni belge tipleri oluşturun
   - Mevcut şablonları düzenleyin
   - JSON şemalarını özelleştirin

## Belge Tipi Şablonları

Sistem, `document_templates` klasöründe JSON formatında belge şablonları saklar:

```json
{
  "name": "fatura_tipi",
  "description": "Standart fatura şablonu",
  "fields_to_extract": [
    "fatura_no",
    "tarih",
    "toplam_tutar"
  ],
  "json_schema": {
    "fatura": {
      "no": "",
      "tarih": "",
      "toplam": 0
    }
  }
}
```

## Özelleştirme

1. **OCR Parametreleri**
   - Metin algılama modeli seçimi
   - Metin tanıma modeli seçimi
   - Sayfa düzleştirme ve yön algılama ayarları

2. **LLM Ayarları**
   - Model seçimi (Gemini/Ollama)
   - Özel prompt şablonları
   - Veri çıkarım alanları

## Hata Ayıklama

- Debug modu için `.env` dosyasında `DEBUG=True` ayarlayın
- OCR sonuçlarını görselleştirmek için interaktif görüntüleyiciyi kullanın
- Hata mesajları ve log kayıtları için konsolu takip edin

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Dalınıza push yapın (`git push origin yeni-ozellik`)
5. Bir Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 
