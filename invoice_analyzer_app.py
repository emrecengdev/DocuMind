import os
import sys
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import subprocess
import requests
from dotenv import load_dotenv
import matplotlib.patches as patches
from matplotlib.figure import Figure
import mplcursors  # mplcursors kütüphanesini ekle
import io
import base64
import tempfile
import webbrowser
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import datetime

# Gemini için Google AI SDK'yı ekle
import google.generativeai as genai

# Doctr modüllerini import et
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page

# PyTorch backend'i kullan
import torch

# Cihaz seçimi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Çevre değişkenlerini yükle
load_dotenv()

# Gemini API anahtarını ayarla
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD3ivYLocYfqmaNnRhafE5qWc0FeLksQSM")
genai.configure(api_key=GEMINI_API_KEY)

# Doctr için model mimarileri
DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large", "linknet_resnet18", "linknet_resnet34", "linknet_resnet50"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large", "sar_resnet31"]

# Belge şablonları dizini
DOCUMENT_TEMPLATES_DIR = "document_templates"
os.makedirs(DOCUMENT_TEMPLATES_DIR, exist_ok=True)

# LLM için prompt şablonu
EXTRACTION_PROMPT = """
Aşağıdaki OCR çıktısından istenen bilgileri çıkar ve JSON formatında döndür.
Çıkarılacak bilgiler: {extraction_fields}

OCR Metni:
{ocr_text}

JSON formatında yanıt ver:
"""

# Belge tipi algılama için prompt
DOCUMENT_TYPE_DETECTION_PROMPT = """
Aşağıdaki OCR metnini analiz edip belge tipini belirle.
Mevcut belge tipleri: {document_types}

Kurallar:
1. Eğer bu belge mevcut tiplerden biriyle eşleşiyorsa, o tipin TAM AYNI ADINI döndür.
2. Eğer belge tipi bulunamazsa, şirket adı + belge türü formatını kullan (örn. "GlobalTech Invoice" veya "GlobalTech Fatura").
3. Yanıtını sadece belge tipi olarak ver, açıklama ekle.

Format tutarlılığı için:
- Sonucu küçük harflerle döndür
- "invoice", "fatura", "invoce" gibi terimlerden BİRİNİ kullan, karıştırma

OCR Metni:
{ocr_text}

Belge tipi:
"""

prompt_template = PromptTemplate(
    input_variables=["extraction_fields", "ocr_text"],
    template=EXTRACTION_PROMPT,
)

document_type_detection_template = PromptTemplate(
    input_variables=["document_types", "ocr_text"],
    template=DOCUMENT_TYPE_DETECTION_PROMPT,
)

# Belge tipini algıla
def detect_document_type(llm, ocr_text, document_types):
    """LLM kullanarak belge tipini algıla ve normalize et"""
    prompt = document_type_detection_template.format(
        document_types=", ".join(document_types),
        ocr_text=ocr_text
    )
    
    try:
        # Hangi model tipi kullanıldığını kontrol et
        if hasattr(llm, 'invoke'):
            # Ollama veya diğer LangChain tabanlı LLM'ler için
            response = llm.invoke(prompt)
            detected_type = response.strip().lower()
        elif hasattr(llm, 'generate_content'):
            # Gemini modeli için
            response = llm.generate_content(prompt)
            detected_type = response.text.strip().lower()
        else:
            st.error("Desteklenmeyen LLM tipi")
            return None
        
        print(f"LLM tarafından algılanan tip: {detected_type}")
        print(f"Mevcut tipler: {document_types}")
        
        # Belge tipini normalize et - genellikle kullanılan varyasyonları standardize et
        normalized_type = normalize_document_type(detected_type)
        print(f"Normalize edilmiş tip: {normalized_type}")
        
        # Normalize edilmiş tip ile mevcut normalize edilmiş tipler arasında eşleşme ara
        normalized_document_types = [normalize_document_type(dt) for dt in document_types]
        print(f"Normalize edilmiş mevcut tipler: {normalized_document_types}")
        
        for i, norm_dt in enumerate(normalized_document_types):
            if normalized_type == norm_dt:
                print(f"Tam eşleşme bulundu: {document_types[i]}")
                return document_types[i]
        
        # Kısmi eşleşme kontrolü - örneğin firma adı eşleşiyor mu?
        company_name = extract_company_name(normalized_type)
        if company_name:
            for i, dt in enumerate(document_types):
                norm_dt = normalize_document_type(dt)
                dt_company = extract_company_name(norm_dt)
                if dt_company and company_name == dt_company:
                    print(f"Firma adı eşleşmesi bulundu: {company_name}")
                    return document_types[i]
        
        # Eşleşme bulunamadıysa, algılanan tipi olduğu gibi döndür
        return detected_type
    except Exception as e:
        st.error(f"Belge tipi algılanırken hata oluştu: {str(e)}")
        return None

def normalize_document_type(doc_type):
    """Belge tipini normalize et - tutarlı bir format oluştur"""
    normalized = doc_type.lower().strip()
    
    # Yaygın varyasyonları standartlaştır
    replacements = {
        "faturası": "fatura",
        "invoce": "invoice", 
        "invoys": "invoice",
        "invois": "invoice",
        "factuur": "invoice",
        "rechnung": "invoice",
        "factura": "invoice"
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # Gereksiz karakterleri kaldır
    for char in ["'", "\"", ",", ".", "-", "_"]:
        normalized = normalized.replace(char, " ")
    
    # Fazla boşlukları temizle
    normalized = " ".join(normalized.split())
    
    return normalized

def extract_company_name(doc_type):
    """Belge tipinden şirket adını çıkar - örn. 'globaltech invoice' -> 'globaltech'"""
    # Yaygın belge türü ifadelerini tanımla
    doc_type_keywords = ["invoice", "fatura", "makbuz", "receipt", "bill", "statement"]
    
    words = doc_type.split()
    if len(words) > 1:
        # Son kelime belge türü mü kontrol et
        if words[-1] in doc_type_keywords:
            return " ".join(words[:-1])  # Son kelimeyi hariç tut
    
    # Belge türü ifadesi bulunamazsa, tüm metni döndür
    return doc_type

# Belge tiplerini yükle
def load_document_types():
    """Kayıtlı belge tiplerini yükle"""
    document_types = []
    
    if not os.path.exists(DOCUMENT_TEMPLATES_DIR):
        print(f"Klasör bulunamadı: {DOCUMENT_TEMPLATES_DIR}")
        os.makedirs(DOCUMENT_TEMPLATES_DIR)
        return []
    
    print(f"Klasör kontrol ediliyor: {DOCUMENT_TEMPLATES_DIR}")
    files_in_dir = os.listdir(DOCUMENT_TEMPLATES_DIR)
    print(f"Klasördeki dosyalar: {files_in_dir}")
    
    for filename in files_in_dir:
        if filename.endswith('.json'):
            document_type = filename.replace('.json', '')
            document_types.append(document_type)
            print(f"Belge tipi yüklendi: {document_type}")
    
    # Hata ayıklama için dosya listesini yazdır
    print(f"Yüklenen belge tipleri: {document_types}")
    
    return document_types

# Belge şablonunu yükle
def load_document_template(document_type):
    """Belge tipi için kayıtlı şablonu yükle"""
    template_path = os.path.join(DOCUMENT_TEMPLATES_DIR, f"{document_type}.json")
    
    if not os.path.exists(template_path):
        return None
    
    with open(template_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Belge şablonunu kaydet
def save_document_template(document_type, template_data):
    """Belge tipi şablonunu kaydet"""
    if not os.path.exists(DOCUMENT_TEMPLATES_DIR):
        os.makedirs(DOCUMENT_TEMPLATES_DIR)
        print(f"Klasör oluşturuldu: {DOCUMENT_TEMPLATES_DIR}")
    
    # Dosya adı için belge tipini normalleştir
    # Boşlukları koru, özel karakterleri koru, sadece küçük harfe çevir
    normalized_type = document_type
    
    template_path = os.path.join(DOCUMENT_TEMPLATES_DIR, f"{normalized_type}.json")
    print(f"Şablon kaydediliyor: {template_path}")
    
    try:
        with open(template_path, 'w', encoding='utf-8') as file:
            json.dump(template_data, file, ensure_ascii=False, indent=4)
        print(f"Şablon başarıyla kaydedildi: {template_path}")
        return template_path
    except Exception as e:
        print(f"Şablon kaydedilirken hata: {str(e)}")
        return None

# LLM kurulumu
def setup_llm(model_name, model_type):
    """LLM modelini kur"""
    if model_type == "ollama":
        return Ollama(model=model_name)
    elif model_type == "gemini":
        model = genai.GenerativeModel(model_name)
        return model
    return None

# Ollama modellerini al
def get_ollama_models():
    """Lokalde mevcut Ollama modellerini al"""
    try:
        # Ollama API'sini kullanarak modelleri al
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            # Modelleri döndür
            models = [model["name"] for model in data["models"]]
            print(f"Bulunan Ollama modelleri: {models}")
            return models
        else:
            print(f"Ollama API hatası: {response.status_code}")
            return ["llama2"]
    except Exception as e:
        print(f"Ollama modellerini alırken hata: {str(e)}")
        return ["llama2"]

# LLM ile veri çıkar
def extract_data_with_llm(llm, ocr_text, extraction_fields, model_type, template=None):
    """LLM ile veriyi çıkar"""
    # Belirli bir şablon yoksa, extraction_fields kullan
    fields_to_extract = extraction_fields
    
    # Eğer şablon varsa ve fields_to_extract içeriyorsa, onu kullan
    if template and 'fields_to_extract' in template:
        fields_to_extract = ", ".join(template['fields_to_extract'])
        
    # Eğer şablon varsa ve json_schema içeriyorsa, onu prompt'a ekle
    json_schema_str = ""
    if template and 'json_schema' in template and template['json_schema']:
        json_schema_str = f"\nLütfen aşağıdaki JSON şemasına uygun olarak veri çıkarımı yap:\n{json.dumps(template['json_schema'], ensure_ascii=False, indent=2)}"
    
    prompt = f"""
Aşağıdaki OCR çıktısından istenen bilgileri çıkar ve JSON formatında döndür.
Çıkarılacak bilgiler: {fields_to_extract}
{json_schema_str}

OCR Metni:
{ocr_text}

JSON formatında yanıt ver:
"""
    
    try:
        if model_type == "ollama":
            response = llm(prompt)
            try:
                # JSON formatında mı kontrol et
                json_data = json.loads(response)
                return json_data
            except:
                # JSON formatında değilse, düz metin olarak döndür
                return {"text": response}
        elif model_type == "gemini":
            response = llm.generate_content(prompt)
            try:
                # Gemini yanıtından JSON bölümünü çıkarmaya çalış
                text = response.text
                # JSON kısmını bul (ilk ve son süslü parantez arası)
                json_str = text[text.find('{'):text.rfind('}')+1]
                json_data = json.loads(json_str)
                return json_data
            except:
                # JSON formatında değilse, düz metin olarak döndür
                return {"text": response.text}
    except Exception as e:
        st.error(f"Veri çıkarımı sırasında hata oluştu: {str(e)}")
        return {"error": str(e)}

def load_predictor(det_arch, reco_arch, assume_straight_pages=True, straighten_pages=False, 
                  export_as_straight_boxes=False, disable_page_orientation=False, 
                  disable_crop_orientation=False, bin_thresh=0.3, box_thresh=0.1, device=None):
    """OCR modelini yükle"""
    # device parametresini ayrı olarak ele al
    predictor = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=not disable_page_orientation,
        detect_language=True
    )
    
    # Cihazı manuel olarak ayarla
    if device is not None and hasattr(predictor, 'det_predictor') and hasattr(predictor.det_predictor, 'model'):
        predictor.det_predictor.model = predictor.det_predictor.model.to(device)
    if device is not None and hasattr(predictor, 'reco_predictor') and hasattr(predictor.reco_predictor, 'model'):
        predictor.reco_predictor.model = predictor.reco_predictor.model.to(device)
    
    return predictor

def forward_image(predictor, img, device=None):
    """Görüntüyü modele ilet ve segmentasyon haritasını döndür"""
    # Basit bir segmentasyon haritası simülasyonu
    # Gerçek uygulamada, modelin ara katmanlarından segmentasyon haritası alınabilir
    # Bu örnek için basit bir gri tonlama dönüşümü kullanıyoruz
    if isinstance(img, np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # PIL Image ise
        gray = np.array(img.convert('L'))
    
    # Basit bir kenar tespiti uygula
    edges = cv2.Canny(gray, 100, 200)
    return edges

def create_interactive_visualization(page_export, page_img):
    """Daha detaylı interaktif görselleştirme oluştur"""
    # Matplotlib figürü oluştur
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Görüntüyü göster
    ax.imshow(page_img)
    ax.axis('off')
    
    # Her kelime için dikdörtgen ve metin ekle
    for block_idx, block in enumerate(page_export["blocks"]):
        for line_idx, line in enumerate(block["lines"]):
            for word_idx, word in enumerate(line["words"]):
                # Kelime koordinatlarını al
                coords = word["geometry"]
                x_min, y_min = coords[0][0] * page_img.shape[1], coords[0][1] * page_img.shape[0]
                x_max, y_max = coords[1][0] * page_img.shape[1], coords[1][1] * page_img.shape[0]
                width, height = x_max - x_min, y_max - y_min
                
                # Dikdörtgen oluştur
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, 
                    linewidth=1, edgecolor='r', facecolor='none', alpha=0.7
                )
                
                # Dikdörtgeni ekle
                ax.add_patch(rect)
                
                # Metin ve doğruluk oranını ekle
                text = f"{word['value']} ({word['confidence']:.2f})"
                ax.text(
                    x_min, y_min - 5, text, 
                    color='blue', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
    
    fig.tight_layout()
    return fig

def create_simple_visualization(page_export, page_img):
    """Basit bir görselleştirme oluştur - sadece kutucukları göster"""
    # Matplotlib figürü oluştur
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Görüntüyü göster
    ax.imshow(page_img)
    ax.axis('off')
    
    # Her kelime için dikdörtgen ekle
    for block_idx, block in enumerate(page_export["blocks"]):
        for line_idx, line in enumerate(block["lines"]):
            for word_idx, word in enumerate(line["words"]):
                # Kelime koordinatlarını al
                coords = word["geometry"]
                x_min, y_min = coords[0][0] * page_img.shape[1], coords[0][1] * page_img.shape[0]
                x_max, y_max = coords[1][0] * page_img.shape[1], coords[1][1] * page_img.shape[0]
                width, height = x_max - x_min, y_max - y_min
                
                # Dikdörtgen oluştur
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, 
                    linewidth=1, edgecolor='r', facecolor='none', alpha=0.7
                )
                
                # Dikdörtgeni ekle
                ax.add_patch(rect)
    
    fig.tight_layout()
    return fig

def process_page(predictor, page, device=None):
    """Tek bir sayfayı işle ve sonuçları döndür"""
    # Görüntüyü modele ilet
    seg_map = forward_image(predictor, page, device)
    seg_map = np.squeeze(seg_map)
    seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # OCR çıktısını al
    out = predictor([page])
    page_export = out.pages[0].export()
    
    # OCR metnini çıkar
    full_text = ""
    for block in page_export["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                full_text += word["value"] + " "
    
    return {
        "seg_map": seg_map,
        "ocr_result": out,
        "page_export": page_export,
        "full_text": full_text
    }

def combine_texts(page_results):
    """Tüm sayfalardan gelen metinleri birleştir"""
    combined_text = ""
    for i, result in enumerate(page_results):
        combined_text += f"\n--- SAYFA {i+1} ---\n"
        combined_text += result["full_text"]
    return combined_text

def create_doctr_visualization(ocr_result, page_idx=0):
    """Doctr'ın kendi görselleştirme metodunu kullanarak interaktif görselleştirme oluştur"""
    # Matplotlib figürü oluştur
    fig = plt.figure(figsize=(10, 10))
    
    # Doctr'ın kendi görselleştirme fonksiyonunu kullan
    # Bu, üzerine gelindiğinde metin ve güven değerini gösteren interaktif bir görselleştirme sağlar
    fig = visualize_page(
        ocr_result.pages[page_idx].export(), 
        ocr_result.pages[page_idx].page, 
        interactive=True,
        add_labels=True
    )
    
    return fig

def save_interactive_visualization(ocr_result, page_idx=0):
    """OCR sonucunu interaktif görselleştirme olarak HTML dosyasına kaydet ve açar"""
    # HTML içeriği oluştur
    html_content = """
    <html>
    <head>
        <title>Doctr OCR Metin Kontrolü</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1000px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Doctr OCR Metin Kontrolü</h1>
            <div class="info">
                <p>Bu görselleştirme, Doctr kütüphanesinin interaktif görselleştirme özelliğini kullanmaktadır.</p>
                <p>Fare imlecini metin kutularının üzerine getirerek tanınan metni ve güven değerini görebilirsiniz.</p>
                <p><strong>Not:</strong> Bu özellik, tarayıcınızda matplotlib ve mplcursors kütüphanelerini kullanarak çalışır.</p>
            </div>
            <p>Görselleştirme yükleniyor, lütfen bekleyin...</p>
            <p>Eğer görselleştirme yüklenmezse, lütfen Python konsolundan açılan matplotlib penceresine bakın.</p>
        </div>
    </body>
    </html>
    """
    
    # Geçici HTML dosyası oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name
    
    # HTML dosyasını tarayıcıda aç
    webbrowser.open('file://' + temp_path)
    
    # Matplotlib figürünü oluştur ve göster
    plt.figure(figsize=(12, 12))
    visualize_page(
        ocr_result.pages[page_idx].export(), 
        ocr_result.pages[page_idx].page, 
        interactive=True,
        add_labels=True
    )
    plt.tight_layout()
    plt.show()
    
    return temp_path

def main():
    """Ana uygulama"""
    # Geniş mod
    st.set_page_config(layout="wide")

    # Kayıtlı belge tiplerini yükle
    document_types = load_document_types()

    # Uygulama durumu
    if 'current_document_type' not in st.session_state:
        st.session_state.current_document_type = None
    if 'document_template' not in st.session_state:
        st.session_state.document_template = None
    if 'detected_document_type' not in st.session_state:
        st.session_state.detected_document_type = None
    if 'processed_document' not in st.session_state:
        st.session_state.processed_document = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Arayüz tasarımı
    st.title("DocuMind - Akıllı Belge Analiz Sistemi")
    
    try:
        # Sekme oluşturmayı dene
        tab_titles = ["Belge Analizi", "Belge Tipleri Yönetimi"]
        tabs = st.tabs(tab_titles)
        
        # Birinci sekme: Belge Analizi
        with tabs[0]:
            show_document_analysis(document_types)
            
        # İkinci sekme: Belge Tipleri Yönetimi - güvenli erişim
        try:
            with tabs[1]:
                show_document_type_management(document_types)
        except Exception as e:
            st.error(f"Belge Tipleri Yönetimi sekmesine erişilemiyor: {str(e)}")
            st.warning("Uygulamayı yeniden başlatmanız gerekebilir.")
    
    except Exception as e:
        st.error(f"Sekme oluşturma hatası: {str(e)}")
        st.info("Alternatif görünüm kullanılıyor.")
        
        # Alternatif görünüm - Radio button ile sekme benzeri görünüm
        selected_view = st.radio(
            "Görünüm Seçin:",
            ["Belge Analizi", "Belge Tipleri Yönetimi"],
            index=st.session_state.active_tab
        )
        
        st.session_state.active_tab = 0 if selected_view == "Belge Analizi" else 1
        
        if selected_view == "Belge Analizi":
            show_document_analysis(document_types)
        else:
            show_document_type_management(document_types)

def show_document_analysis(document_types):
    """Belge analizi işlemlerini göster"""
    # Kenar çubuğu
    # Dosya seçimi
    st.sidebar.title("Belge Seçimi")
    uploaded_file = st.sidebar.file_uploader("Dosya Yükle", type=["pdf", "png", "jpeg", "jpg"])
    
    # Model seçimi
    st.sidebar.title("Model Seçimi")
    det_arch = st.sidebar.selectbox("Metin Algılama Modeli", DET_ARCHS, index=DET_ARCHS.index("db_resnet50"))
    reco_arch = st.sidebar.selectbox("Metin Tanıma Modeli", RECO_ARCHS, index=RECO_ARCHS.index("crnn_vgg16_bn"))
    
    # LLM model seçimi
    st.sidebar.title("LLM Modeli Seçimi")
    
    # Model türü seçimi - varsayılan olarak Gemini'yi seç
    model_type = st.sidebar.radio(
        "Model Türü",
        ["Gemini (API)", "Ollama (Yerel)"]
    )
    
    if model_type == "Ollama (Yerel)":
        # Ollama modellerini al
        ollama_models = get_ollama_models()
        model_name = st.sidebar.selectbox("Ollama Modeli", ollama_models, index=0 if "llama2" in ollama_models else 0)
        selected_model_type = "ollama"
    else:
        # Gemini modelleri
        gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        model_name = st.sidebar.selectbox("Gemini Modeli", gemini_models, index=0)
        selected_model_type = "gemini"
    
    # Parametreler
    st.sidebar.title("Parametreler")
    assume_straight_pages = st.sidebar.checkbox("Düz Sayfalar Varsay", value=True)
    disable_page_orientation = st.sidebar.checkbox("Sayfa Yönü Algılamayı Devre Dışı Bırak", value=False)
    disable_crop_orientation = st.sidebar.checkbox("Kırpma Yönü Algılamayı Devre Dışı Bırak", value=False)
    straighten_pages = st.sidebar.checkbox("Sayfaları Düzleştir", value=False)
    export_straight_boxes = st.sidebar.checkbox("Düz Kutular Olarak Dışa Aktar", value=False)
    
    # Eşik değerleri
    bin_thresh = st.sidebar.slider("İkili Eşik Değeri", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    box_thresh = st.sidebar.slider("Kutu Eşik Değeri", min_value=0.1, max_value=0.9, value=0.1, step=0.1)
    
    # Belge tipi seçimi veya algılama
    st.sidebar.title("Belge Tipi")
    document_type_option = st.sidebar.radio(
        "Belge Tipi Seçimi",
        ["Otomatik Algıla", "Manuel Seç"]
    )
    
    if document_type_option == "Manuel Seç":
        selected_document_type = st.sidebar.selectbox(
            "Belge Tipini Seç", 
            [""] + document_types,
            index=0
        )
        if selected_document_type:
            st.session_state.current_document_type = selected_document_type
            st.session_state.document_template = load_document_template(selected_document_type)
    
    # LLM ile veri çıkarımı için metin kutusu
    st.sidebar.title("Veri Çıkarımı")
    
    # Varsayılan değer olarak "belgedeki tüm verileri şablona uygun olarak çıkar" kullan
    default_extraction_text = "Belgedeki tüm verileri verilen JSON şablonuna uygun olarak çıkar"
    
    # Eğer bir şablon seçilmişse, şablondaki alanları göster
    if st.session_state.document_template and 'fields_to_extract' in st.session_state.document_template:
        default_extraction_text = ", ".join(st.session_state.document_template['fields_to_extract'])
    
    extraction_fields = st.sidebar.text_area(
        "Çıkarmak istediğiniz verileri yazın",
        value=default_extraction_text
    )
    
    # Belge yüklendiyse işle
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        
        # Sayfa seçimi için seçenekler
        st.sidebar.title("Sayfa Seçimi")
        page_selection_mode = st.sidebar.radio(
            "Sayfa İşleme Modu",
            ["Tek Sayfa", "Tüm Sayfalar", "Sayfa Aralığı"]
        )
        
        total_pages = len(doc)
        
        if page_selection_mode == "Tek Sayfa":
            page_idx = st.sidebar.selectbox("Sayfa Seçimi", [idx + 1 for idx in range(total_pages)]) - 1
            selected_pages = [page_idx]
        elif page_selection_mode == "Tüm Sayfalar":
            selected_pages = list(range(total_pages))
            st.sidebar.info(f"Toplam {total_pages} sayfa işlenecek")
        else:  # Sayfa Aralığı
            col1, col2 = st.sidebar.columns(2)
            start_page = col1.number_input("Başlangıç", min_value=1, max_value=total_pages, value=1)
            end_page = col2.number_input("Bitiş", min_value=start_page, max_value=total_pages, value=min(start_page + 4, total_pages))
            selected_pages = list(range(start_page - 1, end_page))
            st.sidebar.info(f"{start_page} - {end_page} arası {len(selected_pages)} sayfa işlenecek")
        
        # Analiz butonu
        if st.sidebar.button("Belgeyi Analiz Et"):
            with st.spinner("Model yükleniyor..."):
                predictor = load_predictor(
                    det_arch=det_arch,
                    reco_arch=reco_arch,
                    assume_straight_pages=assume_straight_pages,
                    straighten_pages=straighten_pages,
                    export_as_straight_boxes=export_straight_boxes,
                    disable_page_orientation=disable_page_orientation,
                    disable_crop_orientation=disable_crop_orientation,
                    bin_thresh=bin_thresh,
                    box_thresh=box_thresh,
                    device=device,
                )
            
            # İlerleme çubuğu
            progress_bar = st.progress(0)
            
            # Tüm seçili sayfaları işle
            page_results = []
            
            for i, page_idx in enumerate(selected_pages):
                with st.spinner(f"Sayfa {page_idx + 1}/{total_pages} işleniyor..."):
                    page = doc[page_idx]
                    result = process_page(predictor, page, device)
                    page_results.append(result)
                
                # İlerleme çubuğunu güncelle
                progress_bar.progress((i + 1) / len(selected_pages))
            
            # İlerleme çubuğunu tamamla
            progress_bar.progress(100)
            
            # Tüm sayfalardan gelen metinleri birleştir
            combined_text = combine_texts(page_results)
            
            # LLM kurulumu
            llm = setup_llm(model_name, selected_model_type)
            
            # Belge tipini otomatik algıla
            if document_type_option == "Otomatik Algıla":
                with st.spinner("Belge tipi algılanıyor..."):
                    detected_type = detect_document_type(llm, combined_text, document_types)
                    if detected_type:
                        st.session_state.detected_document_type = detected_type
                        
                        # Belge adını ve mevcut tipleri yazdır
                        print(f"Algılanan belge tipi: {detected_type}")
                        print(f"Mevcut belge tipleri: {document_types}")
                        
                        # Belge tipi listesinde var mı kontrol et (büyük/küçük harf duyarsız)
                        # Tam metin karşılaştırması yap
                        exists_in_types = False
                        matching_type = None
                        
                        for dt in document_types:
                            dt_lower = dt.lower()
                            detected_lower = detected_type.lower()
                            
                            # Doğrudan karşılaştırma
                            if dt_lower == detected_lower:
                                exists_in_types = True
                                matching_type = dt
                                break
                                
                            # "invoice" ve "invoce" gibi benzerlikleri kontrol et
                            if dt_lower.replace("invoice", "invoce") == detected_lower or \
                               detected_lower.replace("invoice", "invoce") == dt_lower or \
                               dt_lower.replace("invoce", "invoice") == detected_lower or \
                               detected_lower.replace("invoce", "invoice") == dt_lower:
                                exists_in_types = True
                                matching_type = dt
                                print(f"Benzer eşleşme bulundu: {dt} ~ {detected_type}")
                                break
                        
                        if exists_in_types and matching_type:
                            st.session_state.current_document_type = matching_type
                            st.session_state.document_template = load_document_template(matching_type)
                            st.success(f"Bu belge daha önce '{matching_type}' olarak tanımlanmış.")
                        else:
                            # Dosyayı ayrıca doğrudan kontrol et
                            possible_filenames = [
                                f"{detected_type}.json",
                                f"{detected_type.replace('invoice', 'invoce')}.json",
                                f"{detected_type.replace('invoce', 'invoice')}.json"
                            ]
                            
                            found_file = None
                            for filename in possible_filenames:
                                full_path = os.path.join(DOCUMENT_TEMPLATES_DIR, filename)
                                if os.path.exists(full_path):
                                    found_file = filename
                                    matching_type = filename.replace('.json', '')
                                    print(f"Doğrudan dosya eşleşmesi bulundu: {filename}")
                                    break
                            
                            if found_file:
                                st.session_state.current_document_type = matching_type
                                st.session_state.document_template = load_document_template(matching_type)
                                st.success(f"Bu belge daha önce '{matching_type}' olarak tanımlanmış.")
                            else:
                                st.warning(f"Algılanan belge tipi '{detected_type}' sistemde tanımlı değil. Yeni bir belge tipi oluşturabilirsiniz.")
                                st.session_state.current_document_type = None
                                st.session_state.document_template = None
            
            # LLM ile veri çıkarımı
            extracted_data = None
            with st.spinner(f"{model_type} ile veri çıkarımı yapılıyor..."):
                extracted_data = extract_data_with_llm(
                    llm, 
                    combined_text, 
                    extraction_fields, 
                    selected_model_type,
                    st.session_state.document_template
                )
            
            # İşlenmiş belge durumunu güncelle
            st.session_state.processed_document = True
            
            # Belge tipi tanımlama seçeneği
            if st.session_state.detected_document_type and st.session_state.detected_document_type not in document_types:
                if st.button(f"'{st.session_state.detected_document_type}' Belge Tipini Kaydet"):
                    new_template = {
                        "name": st.session_state.detected_document_type,
                        "description": f"{st.session_state.detected_document_type} tipi belge",
                        "fields_to_extract": extraction_fields.split(","),
                        "created_at": datetime.now().isoformat(),
                        "example_data": extracted_data
                    }
                    save_document_template(st.session_state.detected_document_type, new_template)
                    st.success(f"'{st.session_state.detected_document_type}' belge tipi başarıyla kaydedildi!")
                    # Belge tiplerini yeniden yükle
                    document_types = load_document_types()
            
            # Sonuçları göster - Tek ekran düzeni
            st.subheader("Analiz Sonuçları")
            
            # Belge tipi bilgisi
            if st.session_state.current_document_type:
                st.info(f"Belge Tipi: {st.session_state.current_document_type}")
            elif st.session_state.detected_document_type:
                st.info(f"Algılanan Belge Tipi: {st.session_state.detected_document_type} (henüz tanımlanmamış)")
            
            # Sayfa sekmelerini oluştur
            tabs = st.tabs([f"Sayfa {selected_pages[i] + 1}" for i in range(len(page_results))])
            
            # Sayfa sekmeleri
            for i, (tab, result) in enumerate(zip(tabs, page_results)):
                with tab:
                    # Üç sütunlu düzen
                    col1, col2, col3 = st.columns(3)
                    
                    # Orijinal görüntü
                    col1.subheader("Orijinal Belge")
                    col1.image(doc[selected_pages[i]], caption=f"Sayfa {selected_pages[i] + 1}", use_container_width=True)
                    
                    # OCR çıktısı
                    col2.subheader("OCR Çıktısı")
                    # Doctr'ın orijinal görselleştirme fonksiyonunu kullan
                    fig = visualize_page(
                        result["page_export"], 
                        result["ocr_result"].pages[0].page, 
                        interactive=False,
                        add_labels=False
                    )
                    col2.pyplot(fig, use_container_width=True)
                    
                    # Segmentasyon haritası
                    col3.subheader("Segmentasyon Haritası")
                    fig, ax = plt.subplots()
                    ax.imshow(result["seg_map"])
                    ax.axis("off")
                    col3.pyplot(fig, use_container_width=True)
                    
                    # OCR Metin Kontrolü butonu
                    if st.button(f"Sayfa {selected_pages[i] + 1} için OCR Metin Kontrolü", key=f"ocr_check_{i}"):
                        st.info("OCR metin kontrolü için yeni bir pencere açılıyor. Lütfen bekleyin...")
                        save_interactive_visualization(result["ocr_result"], 0)
                    
                    # Sayfa metni
                    st.subheader("Sayfa Metni")
                    st.text_area(f"Sayfa {selected_pages[i] + 1} Metni", result["full_text"], height=150)
                    
                    # JSON çıktısı
                    with st.expander("JSON Formatında OCR Sonuçları"):
                        st.json(result["page_export"])
            
            # Veri çıkarımı sonuçları (en altta)
            st.subheader("Çıkartılmış Veri")
            if extracted_data:
                st.json(extracted_data)
            else:
                st.info("Veri çıkarımı yapılmadı. Lütfen çıkarmak istediğiniz alanları belirtin ve 'Belgeyi Analiz Et' düğmesine tıklayın.")
            
            # Birleştirilmiş metin (expander içinde)
            with st.expander("Tüm Sayfalardan Birleştirilmiş Metin"):
                st.text_area("Birleştirilmiş Metin", combined_text, height=300)

def show_document_type_management(document_types):
    """Belge tipleri yönetimini göster"""
    st.header("Belge Tipleri Yönetimi")
    
    # Mevcut belge tiplerini göster
    st.subheader("Mevcut Belge Tipleri")
    
    if not document_types:
        st.info("Henüz tanımlanmış belge tipi bulunmuyor.")
    else:
        # İçiçe expanderlar yerine daha düz bir yapı kullan
        for doc_type in document_types:
            st.markdown(f"### Belge Tipi: {doc_type}")
            
            template = load_document_template(doc_type)
            if template:
                st.write(f"**Açıklama:** {template.get('description', 'Açıklama yok')}")
                st.write(f"**Oluşturulma Tarihi:** {template.get('created_at', 'Bilinmiyor')}")
                st.write("**Çıkarılacak Alanlar:**")
                for field in template.get('fields_to_extract', []):
                    st.write(f"- {field}")
                
                # Özel JSON şeması gösterimi
                if 'json_schema' in template:
                    st.write("**Özel JSON Yapısı:**")
                    st.json(template['json_schema'])
                
                # Örnek veri (expander içinde)
                with st.expander("Örnek Veri"):
                    st.json(template.get('example_data', {}))
                
                # Belge tipini güncelleme bölümü
                st.markdown("#### Belge Tipini Güncelle")
                updated_description = st.text_area(
                    "Açıklama", 
                    value=template.get('description', ''),
                    key=f"desc_{doc_type}"
                )
                
                # Mevcut alanları virgülle ayrılmış şekilde göster
                current_fields = ", ".join(template.get('fields_to_extract', []))
                updated_fields = st.text_area(
                    "Çıkarılacak Alanlar (virgülle ayırın)",
                    value=current_fields,
                    key=f"fields_{doc_type}"
                )
                
                # Özel JSON şeması düzenleme
                st.write("**Özel JSON Yapısı (Şema):**")
                st.write("Bu bölümde, belge tipine özgü JSON yapısını tanımlayabilirsiniz.")
                st.write("Örnek: `{\"fatura\": {\"no\": \"\", \"tarih\": \"\", \"toplam\": 0}}`")
                
                json_schema = template.get('json_schema', {})
                json_schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2) if json_schema else ""
                
                updated_schema = st.text_area(
                    "JSON Şeması", 
                    value=json_schema_str, 
                    height=200,
                    key=f"schema_{doc_type}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Güncelle", key=f"update_{doc_type}"):
                        try:
                            # JSON şemasını kontrol et
                            if updated_schema.strip():
                                try:
                                    parsed_schema = json.loads(updated_schema)
                                    template['json_schema'] = parsed_schema
                                except json.JSONDecodeError:
                                    st.error("Geçersiz JSON formatı! Lütfen JSON yapısını kontrol edin.")
                                    continue
                            else:
                                template['json_schema'] = {}
                            
                            # Diğer alanları güncelle
                            template['description'] = updated_description
                            template['fields_to_extract'] = [f.strip() for f in updated_fields.split(",") if f.strip()]
                            template['updated_at'] = datetime.now().isoformat()
                            
                            # Şablonu kaydet
                            save_document_template(doc_type, template)
                            st.success(f"{doc_type} belge tipi başarıyla güncellendi!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Belge tipi güncellenirken hata oluştu: {str(e)}")
                
                # Belge tipini silme butonu
                with col2:
                    if st.button(f"{doc_type} Belge Tipini Sil", key=f"delete_{doc_type}"):
                        try:
                            os.remove(os.path.join(DOCUMENT_TEMPLATES_DIR, f"{doc_type}.json"))
                            st.success(f"{doc_type} belge tipi başarıyla silindi!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Belge tipi silinirken hata oluştu: {str(e)}")
            
            # Belge tipleri arasına ayırıcı ekle
            st.markdown("---")
    
    # Yeni belge tipi oluşturma
    st.subheader("Yeni Belge Tipi Oluştur")
    
    # Yeni belge tipi formu
    with st.form("new_document_type_form"):
        new_type_name = st.text_input("Belge Tipi Adı")
        new_type_description = st.text_area("Açıklama")
        new_type_fields = st.text_area("Çıkarılacak Alanlar (virgülle ayırın)", 
                                  "fatura numarası, tarih, toplam tutar, vergi numarası, müşteri adı, ürünler")
        
        # Özel JSON şeması tanımlama
        st.write("**Özel JSON Yapısı (isteğe bağlı):**")
        st.write("Bu bölümde, belge tipine özgü JSON yapısını tanımlayabilirsiniz.")
        st.write("Örnek: `{\"fatura\": {\"no\": \"\", \"tarih\": \"\", \"toplam\": 0}}`")
        
        new_json_schema = st.text_area("JSON Şeması", height=200)
        
        submit_button = st.form_submit_button("Yeni Belge Tipi Oluştur")
        
        if submit_button:
            if not new_type_name:
                st.error("Belge tipi adı boş olamaz!")
            elif new_type_name.lower() in [dt.lower() for dt in document_types]:
                st.error(f"'{new_type_name}' belge tipi zaten mevcut!")
            else:
                try:
                    # JSON şemasını kontrol et
                    json_schema = {}
                    if new_json_schema.strip():
                        try:
                            json_schema = json.loads(new_json_schema)
                        except json.JSONDecodeError:
                            st.error("Geçersiz JSON formatı! Lütfen JSON yapısını kontrol edin.")
                            return
                    
                    new_template = {
                        "name": new_type_name,
                        "description": new_type_description,
                        "fields_to_extract": [field.strip() for field in new_type_fields.split(",") if field.strip()],
                        "created_at": datetime.now().isoformat(),
                        "json_schema": json_schema,
                        "example_data": {}
                    }
                    
                    save_document_template(new_type_name.lower(), new_template)
                    st.success(f"'{new_type_name}' belge tipi başarıyla oluşturuldu!")
                    # Belge tiplerini yeniden yükle ve sayfayı yeniden yükle
                    st.rerun()
                except Exception as e:
                    st.error(f"Belge tipi oluşturulurken hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 