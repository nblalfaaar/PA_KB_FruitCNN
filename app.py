"""Aplikasi web klasifikasi buah (Streamlit) - Proyek Akhir KB.
Ubah bagian KONFIGURASI di bawah untuk menyesuaikan judul, info model, dan perilaku aplikasi.
Tampilan: palet wine, peach, dan blush; font Poppins; ikon Material Symbols.
Foto contoh dibaca dari folder assets/ (apple.jpg, avocado.jpg, grape.jpg).
"""
import base64
import html
from pathlib import Path

import streamlit as st
from PIL import Image

from inference import FruitClassifier

# ===================== KONFIGURASI (silakan ubah) =====================
JUDUL = "Klasifikasi Jenis Buah"
SUBJUDUL = "Proyek Akhir Kecerdasan Buatan - CNN buatan sendiri (tanpa transfer learning)"
INFO_MODEL = {                                # isi dengan angka hasil notebook-mu
    "Jumlah kelas": "50 jenis buah",
    "Akurasi uji": "79,76% (dengan TTA)",
    "Arsitektur": "Custom CNN residual + Swish",
}
NAMA_KELOMPOK = "Kelompok 5"
AMBANG_YAKIN = 0.40                           # di bawah ini, aplikasi memberi peringatan "kurang yakin"
AMBANG_TINGGI = 0.70                          # di atas ini, hasil ditandai "yakin tinggi"
JUMLAH_TOP = 5
# Foto contoh (kunci, label, nama file di folder assets/)
CONTOH = [("apple", "Apel", "apple.jpg"), ("avocado", "Alpukat", "avocado.jpg"), ("grape", "Anggur", "grape.jpg")]
# =====================================================================

ASSETS = Path(__file__).parent / "assets"
OK, MID, LOW = "#2FA37A", "#F29A4B", "#D7263D"

st.set_page_config(page_title=JUDUL, page_icon=":material/nutrition:", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0&display=swap');

:root{
  --plum:#3D0F25; --wine:#7A0B3A; --wine-2:#8F1046; --wine-dark:#45091F;
  --peach:#FFC28A; --peach-soft:#FFE3C7; --blush:#F8DAD5; --blush-2:#FBEBE7; --cream:#FFFBF8;
  --bg:#EADFDC; --mauve:#7A6670; --line:#F0DDD8;
  --shadow:0 2px 4px rgba(61,15,37,.05), 0 22px 44px -20px rgba(61,15,37,.28);
}

/* ---------- Font dan ikon ---------- */
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp button, .stApp input,
.stApp li, .stApp a, .stApp summary, .stApp small, .stApp b, .stApp strong, .stApp h1, .stApp h2, .stApp h3{
  font-family:'Poppins', sans-serif;
}
.stApp [data-testid*="Icon"], .stApp .mi{
  font-family:'Material Symbols Rounded' !important;
  font-weight:normal; font-style:normal; letter-spacing:normal; text-transform:none;
  line-height:1; white-space:nowrap; font-feature-settings:'liga'; -webkit-font-smoothing:antialiased;
  font-variation-settings:'FILL' 1,'wght' 500,'GRAD' 0,'opsz' 24;
}
.mi{display:inline-block; font-size:1.3rem; vertical-align:middle}

/* ---------- Kerangka ---------- */
.stApp{background:var(--bg); color:var(--plum)}
.stApp::before{content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(circle at 6% 14%, rgba(255,255,255,.75) 0 22px, transparent 23px),
    radial-gradient(circle at 93% 9%, rgba(255,255,255,.6) 0 14px, transparent 15px),
    radial-gradient(circle at 88% 78%, rgba(255,255,255,.55) 0 30px, transparent 31px),
    radial-gradient(circle at 3% 86%, rgba(255,255,255,.5) 0 12px, transparent 13px),
    radial-gradient(55% 45% at 85% 0%, rgba(255,255,255,.55), transparent 70%),
    radial-gradient(50% 40% at 0% 100%, rgba(255,255,255,.4), transparent 70%)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]{display:none}
.block-container{position:relative; z-index:1; max-width:1240px; padding:26px 28px 56px}

/* ---------- Top bar ---------- */
.topbar{display:flex; align-items:center; justify-content:space-between; margin-bottom:22px}
.brand{display:flex; align-items:center; gap:12px; font-size:.88rem; font-weight:500; color:var(--mauve)}
.logo{width:46px; height:46px; border-radius:50%; background:var(--wine); color:var(--peach); display:grid; place-items:center}
.who{display:flex; align-items:center; gap:12px; background:rgba(255,255,255,.75); border-radius:999px; padding:6px 22px 6px 6px; box-shadow:var(--shadow)}
.who .av{width:40px; height:40px; border-radius:50%; background:var(--peach); color:var(--wine); display:grid; place-items:center}
.who b{display:block; font-size:.86rem; font-weight:600; line-height:1.2; color:var(--plum)}
.who small{display:block; font-size:.72rem; color:var(--mauve); line-height:1.2}

/* ---------- Hero ---------- */
.hero{background:var(--blush); border-radius:44px; padding:44px 48px; display:grid; grid-template-columns:1.1fr .9fr; gap:30px;
  align-items:center; margin-bottom:26px; box-shadow:var(--shadow); position:relative; overflow:hidden}
.eyebrow{display:inline-flex; align-items:center; gap:8px; background:#fff; color:var(--wine); border-radius:999px; padding:7px 16px 7px 12px; font-size:.8rem; font-weight:600}
.eyebrow .mi{font-size:1.1rem}
.h1{font-size:3.1rem; line-height:1.1; font-weight:400; letter-spacing:-.025em; margin:20px 0 14px; color:var(--mauve)}
.h1 b{font-weight:700; color:var(--plum)}
.h-sub{font-size:.98rem; line-height:1.7; color:var(--mauve); max-width:31rem}
.stats{display:flex; flex-wrap:wrap; gap:10px; margin-top:24px}
.stat{display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,.8); border-radius:999px; padding:9px 16px 9px 11px; font-size:.82rem; font-weight:500}
.stat .mi{font-size:1.15rem; color:var(--wine)}
.collage{position:relative; width:100%; max-width:430px; margin-left:auto; aspect-ratio:1.3}
.blob{position:absolute; right:10%; top:-2%; width:66%; aspect-ratio:1; border-radius:50%; background:var(--peach)}
.dot{position:absolute; border-radius:50%; background:#fff; opacity:.8}
.ph{position:absolute; border:6px solid #fff; border-radius:28px; object-fit:cover; box-shadow:0 20px 30px -14px rgba(61,15,37,.45); background:#fff}
.p1{left:0; top:6%; width:57%; aspect-ratio:246/163; transform:rotate(-7deg)}
.p2{right:0; top:30%; width:52%; aspect-ratio:250/179; transform:rotate(5deg); z-index:2}
.p3{left:20%; bottom:-4%; width:32%; aspect-ratio:1; transform:rotate(-3deg); z-index:3; border-radius:50%}

/* ---------- Kartu input ---------- */
.st-key-input_card{background:var(--cream); border-radius:44px; padding:32px 32px 26px; box-shadow:var(--shadow)}
.sec{display:flex; align-items:center; gap:12px; margin:0 0 16px}
.sec .ico{width:42px; height:42px; border-radius:50%; display:grid; place-items:center; background:var(--wine); color:var(--peach)}
.sec .ico.peach{background:var(--peach); color:var(--wine)}
.sec-t{font-size:1.25rem; font-weight:600; line-height:1.2; color:var(--plum)}
.sec-s{font-size:.82rem; color:var(--mauve); margin-top:1px}

[role="tablist"]{gap:10px !important; background:transparent; padding:0; width:fit-content; border:0 !important; box-shadow:none !important; flex-wrap:wrap}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator, [role="tablist"]::after, [role="tablist"]::before{display:none !important}
[role="tab"]{border-radius:999px !important; padding:10px 20px !important; height:auto !important; background:var(--blush-2); border:0 !important; box-shadow:none !important;
  transition:background .2s}
[role="tab"]:hover{background:var(--blush)}
[role="tab"] p{font-size:.88rem; font-weight:500; color:var(--plum)}
[role="tab"][aria-selected="true"]{background:var(--wine) !important}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *{color:#fff !important}
[role="tabpanel"]{padding-top:18px}

[data-testid="stFileUploaderDropzone"]{background:#FFF3EC; border:2px dashed rgba(122,11,58,.35); border-radius:30px; padding:30px 22px; transition:border-color .2s, background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--wine); background:#FFEBDD}
[data-testid="stFileUploader"] label p, [data-testid="stCameraInput"] label p{font-size:.88rem; font-weight:500}
button[data-testid^="stBaseButton-secondary"]{background:var(--wine); color:#fff; border:0; border-radius:999px; padding:9px 22px; font-weight:500; transition:background .2s, color .2s}
button[data-testid^="stBaseButton-secondary"] *{color:inherit}
button[data-testid^="stBaseButton-secondary"]:hover{background:var(--peach); color:var(--plum)}
button[data-testid^="stBaseButton-secondary"]:focus-visible{outline:3px solid var(--peach); outline-offset:2px}
.stButton > button{width:100%}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{border-radius:28px}
[data-testid="stImage"] img{border-radius:28px; max-height:320px; width:auto !important; max-width:100%; object-fit:contain; margin:0 auto; display:block}
[data-testid="stImageCaption"]{color:var(--mauve); font-size:.8rem; text-align:center}
[data-testid="stToggle"] label p{font-size:.88rem; font-weight:500}

.sample{background:#fff; border-radius:26px; padding:10px 10px 14px; text-align:center; border:2px solid transparent; box-shadow:0 10px 24px -16px rgba(61,15,37,.4); margin-bottom:10px}
.sample.sel{border-color:var(--wine)}
.sample img{width:100%; aspect-ratio:1.2; object-fit:cover; border-radius:18px; display:block}
.sample b{display:block; font-size:.88rem; font-weight:600; margin-top:10px}

/* ---------- Sheet hasil ---------- */
.st-key-sheet{position:sticky; top:24px; background:linear-gradient(180deg, var(--wine) 0%, var(--wine-dark) 100%); border-radius:44px;
  padding:34px 28px 28px; box-shadow:0 30px 50px -24px rgba(69,9,31,.7); color:#fff}
.st-key-sheet::before{content:''; position:absolute; top:13px; left:50%; transform:translateX(-50%); width:64px; height:5px; border-radius:99px; background:rgba(255,255,255,.3)}
.sheet-h{display:flex; align-items:center; justify-content:space-between; margin:4px 0 20px}
.sheet-t{font-size:1.4rem; font-weight:600; color:#fff}
.badge-n{width:40px; height:40px; border-radius:50%; background:var(--peach); color:var(--wine); display:grid; place-items:center; font-weight:700; font-size:.95rem}

.mc{position:relative; background:linear-gradient(135deg, #FFE3C7 0%, #F8CFC8 100%); color:var(--plum); border-radius:32px; padding:22px 24px; margin-bottom:18px; overflow:hidden}
.mc-label{font-size:.8rem; font-weight:500; color:#6F4A57}
.mc-name{font-size:2.1rem; font-weight:700; line-height:1.12; letter-spacing:-.015em; margin:2px 0 14px; padding-right:84px}
.mc-meter{display:flex; align-items:center; gap:12px; margin-bottom:14px}
.mc-bar{flex:1; height:10px; border-radius:99px; background:rgba(61,15,37,.12); overflow:hidden}
.mc-bar i{display:block; height:100%; border-radius:99px; width:var(--w); animation:grow .8s ease-out}
.mc-meter b{font-size:1rem; font-weight:700}
.mc-img{position:absolute; right:20px; top:18px; width:64px; height:64px; border-radius:50%; object-fit:cover; border:4px solid #fff; box-shadow:0 10px 18px -8px rgba(61,15,37,.5)}
.status{display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:5px 14px 5px 9px; font-size:.78rem; font-weight:600}
.status .mi{font-size:1.05rem}
.s-high{background:rgba(47,163,122,.2); color:#145A41}
.s-mid{background:rgba(242,154,75,.3); color:#7A410A}
.s-low{background:rgba(215,38,61,.14); color:#9B1226}
.callout{display:flex; gap:12px; align-items:flex-start; background:rgba(255,194,138,.14); border-left:4px solid var(--peach); border-radius:16px;
  padding:13px 15px; font-size:.84rem; line-height:1.6; margin-bottom:18px; color:#fff}
.callout .mi{color:var(--peach); font-size:1.3rem}
.callout.light{background:#FFF3EC; color:var(--plum); border-left-color:var(--wine)}
.callout.light .mi{color:var(--wine)}

.list-t{font-size:.95rem; font-weight:600; margin:4px 0 6px; color:#fff}
.item{display:flex; align-items:center; gap:14px; padding:11px 0}
.rk{width:52px; height:52px; border-radius:50%; display:grid; place-items:center; font-weight:700; font-size:1.05rem; flex:none;
  background:rgba(255,255,255,.14); color:#fff; box-shadow:inset 0 0 0 1px rgba(255,255,255,.22)}
.item.top .rk{background:var(--peach); color:var(--wine); box-shadow:none}
.it-main{flex:1; min-width:0}
.it-name{font-size:.95rem; font-weight:600; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.it-sub{font-size:.76rem; color:rgba(255,255,255,.66); margin-top:2px}
.it-pill{font-size:.84rem; font-weight:700; background:#fff; color:var(--plum); border-radius:999px; padding:7px 16px; flex:none}
@keyframes grow{from{width:0}}

.paybar{display:flex; align-items:center; justify-content:space-between; background:#fff; color:var(--plum); border-radius:999px; padding:8px 8px 8px 26px;
  font-size:.92rem; font-weight:600; margin-top:18px}
.pay-c{width:50px; height:50px; border-radius:50%; background:var(--peach); color:var(--wine); display:grid; place-items:center}
.note{font-size:.76rem; color:rgba(255,255,255,.62); line-height:1.6; margin-top:14px}

.empty{text-align:center; padding:6px 2px 4px}
.emp-ico{width:84px; height:84px; border-radius:50%; background:var(--peach); color:var(--wine); display:grid; place-items:center; margin:6px auto 18px}
.emp-ico .mi{font-size:2.3rem}
.emp-t{font-size:1.1rem; font-weight:600; margin-bottom:6px; color:#fff}
.emp-d{font-size:.86rem; color:rgba(255,255,255,.74); margin-bottom:20px; line-height:1.6}
.tips{display:grid; gap:10px; text-align:left}
.tip{display:flex; gap:12px; align-items:center; font-size:.85rem; background:rgba(255,255,255,.1); border-radius:999px; padding:10px 18px 10px 12px}
.tip .mi{color:var(--peach); font-size:1.3rem}

/* ---------- Daftar kelas ---------- */
.st-key-band{background:var(--cream); border-radius:44px; padding:32px; margin-top:26px; box-shadow:var(--shadow)}
[data-testid="stTextInput"] label{display:none}
[data-testid="stTextInputRootElement"], [data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInput"] [data-baseweb="base-input"]{
  background:var(--blush-2) !important; border:0 !important; border-radius:999px !important}
[data-testid="stTextInputRootElement"]{position:relative; overflow:visible}
[data-testid="stTextInputRootElement"]::before{content:'search'; font-family:'Material Symbols Rounded'; font-feature-settings:'liga'; font-size:1.3rem; line-height:1;
  color:var(--wine); position:absolute; left:20px; top:50%; transform:translateY(-50%); pointer-events:none; z-index:2; font-variation-settings:'FILL' 1,'wght' 500}
[data-testid="stTextInput"] input{padding:14px 20px 14px 52px !important; font-size:.9rem; color:var(--plum); background:transparent !important; border-radius:999px}
[data-testid="stTextInput"] input::placeholder{color:#9A8791}
[data-testid="stTextInputRootElement"]:focus-within{outline:3px solid var(--peach)}
.chips{display:flex; flex-wrap:wrap; gap:10px; margin-top:6px}
.chip{background:var(--blush-2); color:var(--plum); border-radius:999px; padding:9px 18px; font-size:.82rem; font-weight:500; transition:background .2s}
.chip:hover{background:var(--peach)}
.none{font-size:.9rem; color:var(--mauve); background:var(--blush-2); border-radius:24px; padding:20px 24px; margin-top:6px}
.foot{text-align:center; font-size:.8rem; color:var(--mauve); margin-top:26px}

/* ---------- Responsif ---------- */
@media (max-width: 991px){
  [data-testid="stHorizontalBlock"]:has(.st-key-sheet){flex-wrap:wrap}
  [data-testid="stHorizontalBlock"]:has(.st-key-sheet) > [data-testid="stColumn"]{min-width:100%}
  .st-key-sheet{position:relative; top:0}
  .hero{grid-template-columns:1fr; padding:34px 28px}
  .collage{margin:0 auto}
}
@media (max-width: 640px){
  .block-container{padding:14px 12px 40px}
  .h1{font-size:2.2rem}
  .hero{border-radius:32px; padding:28px 20px}
  .st-key-input_card, .st-key-band{border-radius:32px; padding:24px 18px}
  .st-key-sheet{border-radius:32px; padding:30px 18px 22px}
  .who small{display:none}
}
@media (max-width: 640px){
  [data-testid="stHorizontalBlock"]:has(.sample):not(:has(.st-key-sheet)){flex-wrap:nowrap !important; gap:8px !important}
  [data-testid="stHorizontalBlock"]:has(.sample):not(:has(.st-key-sheet)) > [data-testid="stColumn"]{min-width:0 !important; flex:1 1 0 !important; width:auto !important}
  .sample{padding:6px 6px 10px; border-radius:20px} .sample b{font-size:.78rem}
  .stButton > button{padding:8px 6px}
}
@media (prefers-reduced-motion: reduce){ .mc-bar i{animation:none} }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def h(teks: str) -> str:
    """Rapikan HTML satu baris supaya tidak dibaca Markdown sebagai blok kode."""
    return "".join(baris.strip() for baris in teks.splitlines())


def ikon(nama: str) -> str:
    return f'<span class="mi">{nama}</span>'


@st.cache_data
def uri_gambar(nama_file: str) -> str:
    """Ubah foto di folder assets menjadi data URI. Kosong bila file tidak ada."""
    jalur = ASSETS / nama_file
    if not jalur.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(jalur.read_bytes()).decode()


def judul_bagian(ic: str, teks: str, sub: str = "", warna: str = "") -> str:
    s = f'<div class="sec-s">{html.escape(sub)}</div>' if sub else ""
    return h(f'<div class="sec"><span class="ico {warna}">{ikon(ic)}</span><div><div class="sec-t">{html.escape(teks)}</div>{s}</div></div>')


@st.cache_resource
def muat_model():
    return FruitClassifier("fruit_model.tflite", "class_names.json", "config.json")


def pilih_contoh(kunci: str):
    """Dipanggil saat tombol contoh ditekan: pilih contoh dan kosongkan unggahan/kamera."""
    st.session_state["contoh"] = kunci
    st.session_state["uid"] += 1


st.session_state.setdefault("contoh", None)
st.session_state.setdefault("uid", 0)

clf = muat_model()
peta_contoh = {k: f for k, _, f in CONTOH}

# ============================== TOP BAR ==============================
st.markdown(h(f"""
<div class="topbar">
  <div class="brand"><span class="logo">{ikon("nutrition")}</span><span>Proyek Akhir Kecerdasan Buatan</span></div>
  <div class="who"><span class="av">{ikon("groups")}</span><div><b>{html.escape(NAMA_KELOMPOK)}</b><small>Informatika Unmul</small></div></div>
</div>
"""), unsafe_allow_html=True)

# ============================== HERO ==============================
kata = JUDUL.split()
h1 = html.escape(" ".join(kata[:-2])) + f" <b>{html.escape(' '.join(kata[-2:]))}</b>" if len(kata) > 2 else f"<b>{html.escape(JUDUL)}</b>"
info = list(INFO_MODEL.items()) + [("Ukuran input", f"{clf.img_size} x {clf.img_size} piksel")]
ikon_stat = ["category", "verified", "hub", "photo_size_select_large"]
stat = "".join(f'<span class="stat">{ikon(ic)}{html.escape(str(v))}</span>' for (_, v), ic in zip(info[:4], ikon_stat))

foto_hero = ""
if uri_gambar("apple.jpg"):
    foto_hero = (
        '<span class="dot" style="width:16px;height:16px;left:44%;top:0"></span>'
        '<span class="dot" style="width:10px;height:10px;right:2%;top:8%"></span>'
        f'<img class="ph p1" src="{uri_gambar("apple.jpg")}" alt="Apel hijau dan merah">'
        f'<img class="ph p2" src="{uri_gambar("avocado.jpg")}" alt="Alpukat">'
        f'<img class="ph p3" src="{uri_gambar("grape.jpg")}" alt="Anggur">'
    )
st.markdown(h(f"""
<div class="hero">
  <div>
    <span class="eyebrow">{ikon("auto_awesome")}Dikenali dalam sekali foto</span>
    <div class="h1">{h1}</div>
    <div class="h-sub">{html.escape(SUBJUDUL)}</div>
    <div class="stats">{stat}</div>
  </div>
  <div class="collage" aria-hidden="true"><span class="blob"></span>{foto_hero}</div>
</div>
"""), unsafe_allow_html=True)

# ============================== INPUT DAN HASIL ==============================
kiri, kanan = st.columns([1.08, 1], gap="large")

gambar = None
with kiri:
    with st.container(key="input_card"):
        st.markdown(judul_bagian("add_photo_alternate", "Foto buah", "Gunakan satu jenis buah per foto."), unsafe_allow_html=True)
        tab_unggah, tab_kamera, tab_contoh = st.tabs([":material/upload: Unggah", ":material/photo_camera: Kamera", ":material/photo_library: Contoh"])
        with tab_unggah:
            berkas = st.file_uploader("Pilih foto buah", type=["jpg", "jpeg", "png", "webp", "bmp"], key=f"up_{st.session_state['uid']}")
        with tab_kamera:
            foto = st.camera_input("Ambil foto buah", key=f"cam_{st.session_state['uid']}")
        with tab_contoh:
            if not uri_gambar(CONTOH[0][2]):
                st.markdown(h(f'<div class="callout light">{ikon("info")}<div>Folder <b>assets</b> belum berisi foto contoh.</div></div>'), unsafe_allow_html=True)
            else:
                kolom = st.columns(len(CONTOH))
                for kol, (kunci, label, nama_file) in zip(kolom, CONTOH):
                    with kol:
                        pilih = " sel" if st.session_state["contoh"] == kunci else ""
                        st.markdown(h(f'<div class="sample{pilih}"><img src="{uri_gambar(nama_file)}" alt="{html.escape(label)}"><b>{html.escape(label)}</b></div>'),
                                    unsafe_allow_html=True)
                        st.button("Coba", key=f"btn_{kunci}", icon=":material/play_arrow:", on_click=pilih_contoh, args=(kunci,))

        tta = st.toggle(":material/tune: Mode akurat (TTA, 4 sudut pandang)", value=True,
                        help="Prediksi dirata-ratakan dari 4 versi gambar (asli, flip, zoom tengah, flip zoom). Sedikit lebih lambat, biasanya lebih akurat.")

        sumber = foto or berkas
        if sumber is None and st.session_state["contoh"] in peta_contoh and (ASSETS / peta_contoh[st.session_state["contoh"]]).exists():
            sumber = ASSETS / peta_contoh[st.session_state["contoh"]]
        if sumber is not None:
            try:
                gambar = Image.open(sumber)
                gambar.load()
                st.image(gambar, caption="Gambar masukan", width="stretch")
            except Exception:
                gambar = None
                st.markdown(h(f'<div class="callout light">{ikon("error")}<div>File tidak dapat dibaca sebagai gambar. Coba file lain.</div></div>'),
                            unsafe_allow_html=True)

with kanan:
    with st.container(key="sheet"):
        st.markdown(h(f'<div class="sheet-h"><div class="sheet-t">Hasil prediksi</div><span class="badge-n" title="Jumlah prediksi teratas">{JUMLAH_TOP}</span></div>'),
                    unsafe_allow_html=True)

        if gambar is None:
            st.markdown(h(f"""
            <div class="empty">
              <div class="emp-ico">{ikon("image_search")}</div>
              <div class="emp-t">Belum ada foto</div>
              <div class="emp-d">Unggah foto, ambil foto, atau coba salah satu contoh. Hasilnya muncul di sini.</div>
              <div class="tips">
                <div class="tip">{ikon("check_circle")}Foto jelas dan tidak buram</div>
                <div class="tip">{ikon("check_circle")}Satu jenis buah di dalam frame</div>
                <div class="tip">{ikon("check_circle")}Latar polos lebih baik</div>
              </div>
            </div>
            """), unsafe_allow_html=True)
        else:
            with st.spinner("Memprediksi..."):
                hasil = clf.predict_topk(gambar, k=JUMLAH_TOP, tta=tta)
            nama, prob = hasil[0]
            nama_tampil = nama.replace("_", " ").title()

            if prob >= AMBANG_TINGGI:
                warna, kelas_s, ic_s, label_s = OK, "s-high", "verified", "Yakin tinggi"
            elif prob >= AMBANG_YAKIN:
                warna, kelas_s, ic_s, label_s = MID, "s-mid", "info", "Cukup yakin"
            else:
                warna, kelas_s, ic_s, label_s = LOW, "s-low", "warning", "Kurang yakin"

            gambar_ref = ""
            if nama.lower() in peta_contoh and uri_gambar(peta_contoh[nama.lower()]):
                gambar_ref = f'<img class="mc-img" src="{uri_gambar(peta_contoh[nama.lower()])}" alt="">'

            st.markdown(h(f"""
            <div class="mc">
              {gambar_ref}
              <div class="mc-label">Prediksi utama</div>
              <div class="mc-name">{html.escape(nama_tampil)}</div>
              <div class="mc-meter"><div class="mc-bar"><i style="--w:{prob * 100:.1f}%; background:{warna}"></i></div><b>{prob * 100:.1f}%</b></div>
              <span class="status {kelas_s}">{ikon(ic_s)}{label_s}</span>
            </div>
            """), unsafe_allow_html=True)

            if prob < AMBANG_YAKIN:
                st.markdown(h(f'<div class="callout">{ikon("warning")}<div>Model kurang yakin. Pastikan foto jelas, satu jenis buah, dan buahnya termasuk dalam daftar kelas.</div></div>'),
                            unsafe_allow_html=True)

            baris = ""
            for i, (n, p) in enumerate(hasil):
                baris += (
                    f'<div class="item{" top" if i == 0 else ""}"><span class="rk">{i + 1}</span>'
                    f'<div class="it-main"><div class="it-name">{html.escape(n.replace("_", " ").title())}</div><div class="it-sub">Peluang</div></div>'
                    f'<span class="it-pill">{p * 100:.1f}%</span></div>'
                )
            st.markdown(h(f'<div class="list-t">Top-{JUMLAH_TOP} prediksi</div>{baris}'), unsafe_allow_html=True)
            st.markdown(h(f'<div class="paybar"><span>{"Mode akurat aktif" if tta else "Mode cepat aktif"}</span><span class="pay-c">{ikon("bolt")}</span></div>'),
                        unsafe_allow_html=True)
            st.markdown('<div class="note">Catatan: model hanya mengenal kelas yang terdaftar. Buah di luar daftar tetap akan dipetakan ke salah satu kelas.</div>',
                        unsafe_allow_html=True)

# ============================== DAFTAR KELAS ==============================
with st.container(key="band"):
    st.markdown(judul_bagian("grid_view", "Kelas yang dikenali", f"{len(clf.class_names)} jenis buah yang bisa dikenali.", "peach"), unsafe_allow_html=True)
    cari = st.text_input("Cari jenis buah", placeholder="Cari jenis buah...", label_visibility="collapsed")
    q = cari.strip().lower()
    cocok = [c for c in clf.class_names if q in c.replace("_", " ").lower()] if q else list(clf.class_names)
    if cocok:
        chips = "".join(f'<span class="chip">{html.escape(c.replace("_", " ").title())}</span>' for c in cocok)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="none">Tidak ada kelas yang cocok. Coba kata kunci lain, misalnya nama buah dalam bahasa Inggris.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="foot">Proyek Akhir Kecerdasan Buatan, {html.escape(NAMA_KELOMPOK)}</div>', unsafe_allow_html=True)
