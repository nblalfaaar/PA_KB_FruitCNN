"""Aplikasi web klasifikasi buah (Streamlit) - Proyek Akhir KB.
Ubah bagian KONFIGURASI di bawah untuk menyesuaikan judul, info model, dan perilaku aplikasi.
Tampilan: dashboard modern dengan palet Deep Blueberry, Strawberry Delight, Lime Zest, Golden Honey;
font Poppins; ikon Material Symbols.
"""
import html

import pandas as pd
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
# =====================================================================

# Palet warna
BLUEBERRY = "#2D3E56"
STRAWBERRY = "#FF3131"
LIME = "#9ACD32"
HONEY = "#FFD166"

st.set_page_config(page_title=JUDUL, page_icon=":material/nutrition:", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0&display=swap');

:root{
  --blue:#2D3E56; --blue-2:#3B5070; --blue-soft:#E3E8EF; --blue-mid:#B7C1CF;
  --red:#FF3131; --red-dark:#C71F1F;
  --lime:#9ACD32; --lime-dark:#4A6B05;
  --honey:#FFD166; --honey-dark:#8A5F00;
  --outer:#DCE3EC; --shell:#FAF8F2; --card:#FFFFFF; --line:#E8E5DC; --muted:#6B7889;
  --shadow:0 1px 2px rgba(45,62,86,.06), 0 14px 34px -14px rgba(45,62,86,.2);
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
html{scroll-behavior:smooth}

/* ---------- Kerangka ---------- */
.stApp{background:var(--outer); color:var(--blue)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]{display:none}
.block-container{max-width:1380px; padding:28px 28px 28px 118px}
.st-key-shell{background:var(--shell); border-radius:38px; padding:30px 30px 34px;
  box-shadow:0 30px 60px -30px rgba(45,62,86,.35)}
.anchor{position:relative; top:-90px; height:0}

/* ---------- Rail navigasi ---------- */
.rail{position:fixed; left:28px; top:28px; bottom:28px; width:76px; background:var(--blue); border-radius:32px;
  display:flex; flex-direction:column; align-items:center; padding:18px 0; gap:10px; z-index:1000;
  box-shadow:0 20px 40px -20px rgba(45,62,86,.6)}
.rail-logo{width:46px; height:46px; border-radius:16px; background:var(--honey); color:var(--blue);
  display:grid; place-items:center; margin-bottom:14px}
.rail-logo .mi{font-size:1.5rem}
.rail a.rb{width:46px; height:46px; border-radius:16px; display:grid; place-items:center; color:rgba(255,255,255,.72);
  text-decoration:none; transition:background .2s, color .2s}
.rail a.rb:hover{background:rgba(255,255,255,.14); color:#fff}
.rail a.rb.on{background:rgba(255,255,255,.18); color:#fff}
.rail a.rb:focus-visible{outline:3px solid var(--honey); outline-offset:2px}
.rail .sp{flex:1}

/* ---------- Topbar ---------- */
[data-testid="stTextInput"] label{display:none}
[data-testid="stTextInputRootElement"], [data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInput"] [data-baseweb="base-input"]{
  background:#fff !important; border:0 !important; border-radius:999px !important}
[data-testid="stTextInputRootElement"]{position:relative; box-shadow:var(--shadow); overflow:visible}
[data-testid="stTextInputRootElement"]::before{
  content:'search'; font-family:'Material Symbols Rounded'; font-feature-settings:'liga'; font-size:1.3rem; line-height:1;
  color:var(--muted); position:absolute; left:20px; top:50%; transform:translateY(-50%); pointer-events:none; z-index:2;
  font-variation-settings:'FILL' 1,'wght' 500}
[data-testid="stTextInput"] input{padding:14px 20px 14px 52px !important; font-size:.9rem; color:var(--blue); background:transparent !important; border-radius:999px}
[data-testid="stTextInput"] input::placeholder{color:#8794A6}
[data-testid="stTextInputRootElement"]:focus-within{outline:3px solid var(--honey)}
.who{display:flex; align-items:center; gap:12px; background:#fff; border-radius:999px; padding:7px 22px 7px 8px; box-shadow:var(--shadow); width:fit-content; margin-left:auto}
.who .av{width:42px; height:42px; border-radius:50%; background:var(--honey); display:grid; place-items:center}
.who b{display:block; font-size:.88rem; font-weight:600; line-height:1.2}
.who small{display:block; font-size:.74rem; color:var(--muted); line-height:1.2}
.hint{font-size:.82rem; margin:10px 0 0 6px; color:var(--muted)}
.hint a{color:var(--red-dark); font-weight:600; text-decoration:none}

/* ---------- Banner ---------- */
.banner{background:var(--blue-soft); border-radius:32px; padding:26px 30px 24px; margin:22px 0 28px; overflow:hidden}
.art{display:grid; grid-template-columns:repeat(4, 1fr); gap:16px}
.ft{aspect-ratio:1; border-radius:28px; display:grid; place-items:center; transform:rotate(var(--r)); box-shadow:0 14px 24px -14px rgba(0,0,0,.45)}
.ft:nth-child(even){margin-top:14px}
.ft svg{width:68%; height:68%; filter:drop-shadow(0 8px 6px rgba(0,0,0,.28))}
.f-red{background:var(--red)} .f-blue{background:var(--blue)} .f-lime{background:var(--lime)} .f-honey{background:var(--honey)}
.bar{position:relative; margin:-14px 0 0; background:var(--blue); color:#fff; border-radius:26px; padding:20px 28px;
  display:flex; align-items:center; justify-content:space-between; gap:20px; box-shadow:0 0 0 7px var(--blue-soft)}
.bar-t{font-size:1.9rem; font-weight:700; line-height:1.15; letter-spacing:-.015em}
.bar-s{font-size:.88rem; color:rgba(255,255,255,.78); margin-top:4px; line-height:1.55}
.pill-h{background:var(--honey); color:var(--blue); border-radius:999px; padding:8px 18px; font-size:.82rem; font-weight:600; white-space:nowrap}

/* ---------- Judul bagian ---------- */
.sec{display:flex; align-items:center; gap:12px; margin:0 0 14px}
.sec .ico{width:40px; height:40px; border-radius:14px; display:grid; place-items:center; background:var(--blue); color:#fff}
.sec .ico.honey{background:var(--honey); color:var(--blue)}
.sec .ico.lime{background:var(--lime); color:var(--blue)}
.sec-t{font-size:1.2rem; font-weight:600; line-height:1.2}
.sec-s{font-size:.82rem; color:var(--muted); margin-top:1px}

/* ---------- Kartu input ---------- */
.st-key-input_card{background:var(--card); border-radius:28px; padding:26px 26px 22px; box-shadow:var(--shadow)}
[role="tablist"]{gap:8px !important; background:#F1EFE8; padding:5px; border-radius:999px; width:fit-content; border:0 !important; box-shadow:none !important}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator{display:none !important}
[role="tab"]{border-radius:999px !important; padding:8px 18px !important; height:auto !important; background:transparent; border:0 !important; box-shadow:none !important}
[role="tab"] p{font-size:.88rem; font-weight:500; color:var(--blue)}
[role="tab"][aria-selected="true"]{background:var(--blue) !important}
[role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] *{color:#fff !important}
[role="tabpanel"]{padding-top:16px}
[data-testid="stFileUploaderDropzone"]{background:#F8F7F3; border:2px dashed rgba(45,62,86,.35); border-radius:22px; padding:28px 20px;
  transition:border-color .2s, background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--red); background:#FFF6F5}
[data-testid="stFileUploader"] label p, [data-testid="stCameraInput"] label p{font-size:.88rem; font-weight:500}
button[data-testid^="stBaseButton-secondary"]{background:var(--blue); color:#fff; border:0; border-radius:999px; padding:8px 20px; font-weight:500;
  transition:background .2s, color .2s}
button[data-testid^="stBaseButton-secondary"] *{color:inherit}
button[data-testid^="stBaseButton-secondary"]:hover{background:var(--honey); color:var(--blue)}
button[data-testid^="stBaseButton-secondary"]:focus-visible{outline:3px solid var(--honey); outline-offset:2px}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img, [data-testid="stImage"] img{border-radius:22px}
[data-testid="stImageCaption"]{color:var(--muted); font-size:.8rem; text-align:center}
[data-testid="stImage"] img{max-height:340px; width:auto !important; max-width:100%; object-fit:contain; margin:0 auto; display:block}
[data-testid="stToggle"] label p{font-size:.88rem; font-weight:500}

/* ---------- Kartu info model ---------- */
.cards{display:grid; grid-template-columns:repeat(4, 1fr); gap:16px}
.card{display:block}
.card-tile{aspect-ratio:2.2; border-radius:26px; display:grid; place-items:center}
.card-tile .mi{font-size:2.6rem}
.card-v{font-size:.95rem; font-weight:600; margin-top:12px; line-height:1.35}
.card-k{font-size:.78rem; color:var(--muted); margin-top:1px}
.c-lime{background:var(--lime); color:var(--blue)} .c-honey{background:var(--honey); color:var(--blue)}
.c-red{background:var(--red); color:#fff} .c-blue{background:var(--blue); color:#fff}

/* ---------- Panel samping ---------- */
.st-key-side_col{position:sticky; top:28px; background:var(--card); border-radius:32px; padding:28px 26px 26px; box-shadow:var(--shadow)}
.prof{display:flex; align-items:center; gap:14px; margin-bottom:22px}
.prof .av{width:62px; height:62px; border-radius:20px; background:var(--honey); display:grid; place-items:center; flex:none}
.prof .av .mi{font-size:1.9rem}
.prof-n{font-size:1.1rem; font-weight:600; line-height:1.2}
.prof-s{font-size:.8rem; color:var(--muted); margin:2px 0 8px}
.pill{display:inline-block; background:var(--red); color:#fff; border-radius:999px; padding:4px 14px; font-size:.72rem; font-weight:600}
.side-t{font-size:1.1rem; font-weight:600; margin:0 0 14px}

.res-head{display:flex; align-items:center; gap:20px; margin:0 0 18px; flex-wrap:wrap}
.ring{width:118px; height:118px; border-radius:50%; flex:none; display:grid; place-items:center;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), #ECEAE4 0)}
.ring-in{width:90px; height:90px; border-radius:50%; background:#fff; display:flex; align-items:baseline;
  justify-content:center; padding-top:30px; box-sizing:border-box}
.ring-in b{font-size:1.4rem; font-weight:700; line-height:1}
.ring-in small{font-size:.74rem; font-weight:600; margin-left:2px}
.res-label{font-size:.8rem; color:var(--muted); font-weight:500}
.res-name{font-size:1.65rem; font-weight:700; line-height:1.15; letter-spacing:-.01em; margin:2px 0 10px}
.status{display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:5px 13px 5px 9px; font-size:.78rem; font-weight:600}
.status .mi{font-size:1.05rem}
.s-high{background:rgba(154,205,50,.25); color:var(--lime-dark)}
.s-mid{background:rgba(255,209,102,.4); color:var(--honey-dark)}
.s-low{background:rgba(255,49,49,.12); color:var(--red-dark)}
.callout{display:flex; gap:12px; align-items:flex-start; background:rgba(255,49,49,.08); border-left:4px solid var(--red);
  border-radius:14px; padding:13px 15px; font-size:.84rem; line-height:1.6; margin-bottom:18px}
.callout .mi{color:var(--red); font-size:1.3rem}

.item{display:flex; align-items:center; gap:14px; padding:12px 0; border-top:1px solid var(--line)}
.item:first-of-type{border-top:0}
.rk{width:44px; height:44px; border-radius:15px; display:grid; place-items:center; font-weight:700; font-size:1rem;
  background:var(--blue-soft); color:var(--blue); flex:none}
.rk.k-high{background:var(--lime)} .rk.k-mid{background:var(--honey)} .rk.k-low{background:var(--red); color:#fff}
.it-main{flex:1; min-width:0}
.it-name{font-size:.9rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.it-bar{height:7px; background:#ECEAE4; border-radius:999px; margin-top:8px; overflow:hidden}
.it-bar i{display:block; height:100%; border-radius:999px; width:var(--w); animation:grow .8s ease-out}
.it-pill{font-size:.76rem; font-weight:600; background:var(--blue-soft); color:var(--blue); border-radius:999px; padding:4px 12px; flex:none}
.item.top .it-pill{background:var(--blue); color:#fff}
@keyframes grow{from{width:0}}
.note{font-size:.76rem; color:var(--muted); line-height:1.6; margin-top:12px}

.empty{text-align:center; padding:6px 4px 4px}
.empty .ft{width:150px; margin:6px auto 20px; --r:-6deg}
.empty-t{font-size:1.05rem; font-weight:600; margin-bottom:6px}
.empty-d{font-size:.85rem; color:var(--muted); margin-bottom:20px; line-height:1.6}
.tips{display:grid; gap:10px; text-align:left}
.tip{display:flex; gap:12px; align-items:center; font-size:.85rem; background:#F8F7F3; border-radius:16px; padding:10px 14px}
.tip .mi{color:var(--lime-dark); font-size:1.3rem}

[data-testid="stExpander"]{border:0; border-radius:18px; background:#F8F7F3}
[data-testid="stExpander"] summary p{font-weight:600; font-size:.88rem}

/* ---------- Daftar kelas ---------- */
.chips{display:flex; flex-wrap:wrap; gap:10px}
.chip{background:#fff; color:var(--blue); border-radius:999px; padding:8px 18px; font-size:.82rem; font-weight:500; box-shadow:var(--shadow);
  transition:background .2s}
.chip:hover{background:var(--honey)}
.none{font-size:.9rem; color:var(--muted); background:#fff; border-radius:20px; padding:20px 24px}
.foot{text-align:center; font-size:.8rem; color:var(--muted); margin-top:26px}

[data-testid="stColumn"]:has(.st-key-side_col), [data-testid="stColumn"]:has(.st-key-side_col) > [data-testid="stVerticalBlock"]{height:100%}

/* ---------- Responsif ---------- */
@media (max-width: 991px){
  .cards{grid-template-columns:repeat(2, 1fr)}
  .card-tile{aspect-ratio:1.6}
  [data-testid="stHorizontalBlock"]:has(.st-key-side_col){flex-wrap:wrap}
  [data-testid="stHorizontalBlock"]:has(.st-key-side_col) > [data-testid="stColumn"]{min-width:100%}
}
@media (max-width: 760px){
  .block-container{padding:14px 12px 96px}
  .st-key-shell{padding:18px 14px 24px; border-radius:28px}
  .rail{top:auto; left:12px; right:12px; bottom:12px; width:auto; height:64px; flex-direction:row; justify-content:space-around;
    padding:0 10px; border-radius:24px}
  .rail-logo, .rail .sp{display:none}
  .art{gap:10px} .ft{border-radius:20px}
  .banner{padding:16px 14px 16px}
  .bar{flex-direction:column; align-items:flex-start; padding:18px 20px; margin:-12px 0 0}
  .bar-t{font-size:1.45rem}
  .cards{grid-template-columns:repeat(2, 1fr)}
  .card-tile{aspect-ratio:1.5}
  .who{margin:10px 0 0}
}
@media (prefers-reduced-motion: reduce){ .it-bar i{animation:none} html{scroll-behavior:auto} }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------- Ilustrasi buah (SVG buatan sendiri, selaras palet) ----------
SVG_STRAWBERRY = (
    '<svg viewBox="0 0 100 100" aria-hidden="true">'
    '<path d="M50 93C22 76 10 52 16 36C21 24 38 24 50 31C62 24 79 24 84 36C90 52 78 76 50 93Z" fill="#D61B22"/>'
    '<path d="M50 31C44 17 32 16 27 21C35 22 42 27 50 31ZM50 31C56 17 68 16 73 21C65 22 58 27 50 31ZM50 31C47 21 49 14 50 9C52 15 53 23 50 31Z" fill="#4A6B05"/>'
    '<g fill="#FFD166"><ellipse cx="36" cy="46" rx="2" ry="3"/><ellipse cx="50" cy="44" rx="2" ry="3"/><ellipse cx="64" cy="46" rx="2" ry="3"/>'
    '<ellipse cx="30" cy="60" rx="2" ry="3"/><ellipse cx="43" cy="58" rx="2" ry="3"/><ellipse cx="57" cy="58" rx="2" ry="3"/>'
    '<ellipse cx="70" cy="60" rx="2" ry="3"/><ellipse cx="38" cy="73" rx="2" ry="3"/><ellipse cx="50" cy="71" rx="2" ry="3"/>'
    '<ellipse cx="62" cy="73" rx="2" ry="3"/></g>'
    '<ellipse cx="30" cy="44" rx="4" ry="9" fill="#fff" opacity=".25" transform="rotate(20 30 44)"/></svg>'
)
SVG_BLUEBERRY = (
    '<svg viewBox="0 0 100 100" aria-hidden="true">'
    '<circle cx="50" cy="54" r="34" fill="#4E6A94"/><circle cx="50" cy="54" r="34" fill="none" stroke="#1F2C40" stroke-width="3" opacity=".5"/>'
    '<polygon points="50,42 53.5,49.5 61.5,50 55.5,55.5 57,63.5 50,59.5 43,63.5 44.5,55.5 38.5,50 46.5,49.5" fill="#2A3A52"/>'
    '<ellipse cx="34" cy="38" rx="9" ry="5" fill="#fff" opacity=".25" transform="rotate(-35 34 38)"/></svg>'
)
SVG_LIME = (
    '<svg viewBox="0 0 100 100" aria-hidden="true">'
    '<circle cx="50" cy="50" r="38" fill="#4A6B05"/><circle cx="50" cy="50" r="33" fill="#EAF6B8"/><circle cx="50" cy="50" r="29" fill="#C5E468"/>'
    '<g stroke="#EAF6B8" stroke-width="3" stroke-linecap="round"><line x1="50" y1="21" x2="50" y2="79"/><line x1="21" y1="50" x2="79" y2="50"/>'
    '<line x1="29.5" y1="29.5" x2="70.5" y2="70.5"/><line x1="70.5" y1="29.5" x2="29.5" y2="70.5"/></g>'
    '<circle cx="50" cy="50" r="4.5" fill="#EAF6B8"/></svg>'
)
SVG_HONEY = (
    '<svg viewBox="0 0 100 100" aria-hidden="true">'
    '<path d="M50 9C50 9 21 45 21 63C21 80 34 91 50 91C66 91 79 80 79 63C79 45 50 9 50 9Z" fill="#F0A81C"/>'
    '<path d="M50 9C50 9 21 45 21 63C21 80 34 91 50 91C66 91 79 80 79 63C79 45 50 9 50 9Z" fill="none" stroke="#B87A06" stroke-width="2.5" opacity=".55"/>'
    '<ellipse cx="38" cy="66" rx="5" ry="10" fill="#FFE7A3" opacity=".85" transform="rotate(12 38 66)"/></svg>'
)


def h(teks: str) -> str:
    """Rapikan HTML satu baris supaya tidak dibaca Markdown sebagai blok kode."""
    return "".join(baris.strip() for baris in teks.splitlines())


def ikon(nama: str) -> str:
    return f'<span class="mi">{nama}</span>'


def judul_bagian(ic: str, teks: str, sub: str = "", warna: str = "") -> str:
    s = f'<div class="sec-s">{html.escape(sub)}</div>' if sub else ""
    return h(f'<div class="sec"><span class="ico {warna}">{ikon(ic)}</span><div><div class="sec-t">{html.escape(teks)}</div>{s}</div></div>')


@st.cache_resource
def muat_model():
    return FruitClassifier("fruit_model.tflite", "class_names.json", "config.json")


clf = muat_model()

# ============================== RAIL NAVIGASI ==============================
st.markdown(h(f"""
<nav class="rail" aria-label="Navigasi halaman">
  <div class="rail-logo">{ikon("nutrition")}</div>
  <a class="rb on" href="#atas" title="Beranda" aria-label="Beranda">{ikon("home")}</a>
  <a class="rb" href="#foto" title="Foto buah" aria-label="Foto buah">{ikon("add_photo_alternate")}</a>
  <a class="rb" href="#hasil" title="Hasil prediksi" aria-label="Hasil prediksi">{ikon("insights")}</a>
  <a class="rb" href="#model" title="Tentang model" aria-label="Tentang model">{ikon("memory")}</a>
  <a class="rb" href="#kelas" title="Daftar kelas" aria-label="Daftar kelas">{ikon("grid_view")}</a>
  <div class="sp"></div>
  <a class="rb" href="#model" title="Bantuan" aria-label="Bantuan">{ikon("help")}</a>
</nav>
"""), unsafe_allow_html=True)

# ============================== KERANGKA UTAMA ==============================
gambar = None
with st.container(key="shell"):
    utama, samping = st.columns([2.15, 1], gap="large")

    # ---------------------------- KOLOM UTAMA ----------------------------
    with utama:
        st.markdown('<div class="anchor" id="atas"></div>', unsafe_allow_html=True)
        kolom_cari, kolom_who = st.columns([3, 1.3], vertical_alignment="center")
        with kolom_cari:
            cari = st.text_input("Cari jenis buah", placeholder="Cari jenis buah yang dikenali...", label_visibility="collapsed")
        with kolom_who:
            st.markdown(h(f"""
            <div class="who"><span class="av">{ikon("groups")}</span>
            <div><b>{html.escape(NAMA_KELOMPOK)}</b><small>Proyek Akhir KB</small></div></div>
            """), unsafe_allow_html=True)

        q = cari.strip().lower()
        cocok = [c for c in clf.class_names if q in c.replace("_", " ").lower()] if q else list(clf.class_names)
        if q:
            st.markdown(f'<div class="hint">{len(cocok)} kelas cocok dengan pencarianmu. '
                        f'<a href="#kelas">Lihat daftar</a></div>', unsafe_allow_html=True)

        # Banner
        st.markdown(h(f"""
        <div class="banner">
          <div class="art" aria-hidden="true">
            <div class="ft f-red" style="--r:-4deg">{SVG_STRAWBERRY}</div>
            <div class="ft f-blue" style="--r:3deg">{SVG_BLUEBERRY}</div>
            <div class="ft f-lime" style="--r:-3deg">{SVG_LIME}</div>
            <div class="ft f-honey" style="--r:4deg">{SVG_HONEY}</div>
          </div>
          <div class="bar">
            <div>
              <div class="bar-t">{html.escape(JUDUL)}</div>
              <div class="bar-s">{html.escape(SUBJUDUL)}</div>
            </div>
            <span class="pill-h">{len(clf.class_names)} jenis buah</span>
          </div>
        </div>
        """), unsafe_allow_html=True)

        # Input foto
        st.markdown('<div class="anchor" id="foto"></div>', unsafe_allow_html=True)
        with st.container(key="input_card"):
            st.markdown(judul_bagian("add_photo_alternate", "Foto buah", "Gunakan satu jenis buah per foto."), unsafe_allow_html=True)
            tab_unggah, tab_kamera = st.tabs([":material/upload: Unggah gambar", ":material/photo_camera: Kamera"])
            with tab_unggah:
                berkas = st.file_uploader("Pilih foto buah", type=["jpg", "jpeg", "png", "webp", "bmp"])
            with tab_kamera:
                foto = st.camera_input("Ambil foto buah")

            tta = st.toggle(":material/tune: Mode akurat (TTA, 4 sudut pandang)", value=True,
                            help="Prediksi dirata-ratakan dari 4 versi gambar (asli, flip, zoom tengah, flip zoom). Sedikit lebih lambat, biasanya lebih akurat.")

            sumber = foto or berkas
            if sumber is not None:
                try:
                    gambar = Image.open(sumber)
                    gambar.load()
                    st.image(gambar, caption="Gambar masukan", width="stretch")
                except Exception:
                    gambar = None
                    st.markdown(h(f"""
                    <div class="callout">{ikon("error")}<div>File tidak dapat dibaca sebagai gambar. Coba file lain.</div></div>
                    """), unsafe_allow_html=True)


    # ---------------------------- PANEL SAMPING ----------------------------
    with samping:
        with st.container(key="side_col"):
            st.markdown(h(f"""
            <div class="prof">
              <div class="av">{ikon("nutrition")}</div>
              <div><div class="prof-n">Model klasifikasi</div>
              <div class="prof-s">CNN buatan sendiri</div>
              <span class="pill">{len(clf.class_names)} kelas</span></div>
            </div>
            <div class="anchor" id="hasil"></div>
            <div class="side-t">Hasil prediksi</div>
            """), unsafe_allow_html=True)

            if gambar is None:
                st.markdown(h(f"""
                <div class="empty">
                  <div class="ft f-honey">{SVG_HONEY}</div>
                  <div class="empty-t">Belum ada foto</div>
                  <div class="empty-d">Unggah foto atau ambil foto buah, hasilnya muncul di sini.</div>
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
                    warna, kelas_s, kelas_k, ic_s, label_s = LIME, "s-high", "k-high", "verified", "Yakin tinggi"
                elif prob >= AMBANG_YAKIN:
                    warna, kelas_s, kelas_k, ic_s, label_s = HONEY, "s-mid", "k-mid", "info", "Cukup yakin"
                else:
                    warna, kelas_s, kelas_k, ic_s, label_s = STRAWBERRY, "s-low", "k-low", "warning", "Kurang yakin"

                st.markdown(h(f"""
                <div class="res-head">
                  <div class="ring" style="--p:{prob * 100:.1f}; --c:{warna}">
                    <div class="ring-in"><b>{prob * 100:.1f}</b><small>%</small></div>
                  </div>
                  <div>
                    <div class="res-label">Prediksi utama</div>
                    <div class="res-name">{html.escape(nama_tampil)}</div>
                    <span class="status {kelas_s}">{ikon(ic_s)}{label_s}</span>
                  </div>
                </div>
                """), unsafe_allow_html=True)

                if prob < AMBANG_YAKIN:
                    st.markdown(h(f"""
                    <div class="callout">{ikon("warning")}<div>Model kurang yakin. Pastikan foto jelas, satu jenis buah,
                    dan buahnya termasuk dalam daftar kelas.</div></div>
                    """), unsafe_allow_html=True)

                baris = ""
                for i, (n, p) in enumerate(hasil):
                    isi = warna if i == 0 else BLUEBERRY
                    baris += (
                        f'<div class="item{" top" if i == 0 else ""}">'
                        f'<span class="rk {kelas_k if i == 0 else ""}">{i + 1}</span>'
                        f'<div class="it-main"><div class="it-name">{html.escape(n.replace("_", " ").title())}</div>'
                        f'<div class="it-bar"><i style="--w:{max(p * 100, 1.5):.1f}%; background:{isi}"></i></div></div>'
                        f'<span class="it-pill">{p * 100:.1f}%</span></div>'
                    )
                st.markdown(h(f'<div class="side-t" style="margin-top:6px">Top-{JUMLAH_TOP} prediksi</div>{baris}'), unsafe_allow_html=True)

                with st.expander("Lihat tabel probabilitas", icon=":material/table_chart:"):
                    df = pd.DataFrame({"Kelas": [n.replace("_", " ") for n, _ in hasil],
                                       "Probabilitas (%)": [round(p * 100, 2) for _, p in hasil]})
                    st.dataframe(df, hide_index=True, width="stretch")

                st.markdown('<div class="note">Catatan: model hanya mengenal kelas yang terdaftar. '
                            'Buah di luar daftar tetap akan dipetakan ke salah satu kelas.</div>', unsafe_allow_html=True)

    # Info model
    st.markdown('<div class="anchor" id="model"></div><div style="height:28px"></div>', unsafe_allow_html=True)
    st.markdown(judul_bagian("memory", "Tentang model", "Ringkasan hasil pelatihan.", "honey"), unsafe_allow_html=True)
    info = list(INFO_MODEL.items()) + [("Ukuran input", f"{clf.img_size} x {clf.img_size} piksel")]
    gaya = [("c-lime", "category"), ("c-honey", "verified"), ("c-red", "hub"), ("c-blue", "photo_size_select_large")]
    kartu = "".join(
        f'<div class="card"><div class="card-tile {kelas}">{ikon(ic)}</div>'
        f'<div class="card-v">{html.escape(str(v))}</div><div class="card-k">{html.escape(k)}</div></div>'
        for (k, v), (kelas, ic) in zip(info[:4], gaya)
    )
    st.markdown(f'<div class="cards">{kartu}</div>', unsafe_allow_html=True)

    # Daftar kelas
    st.markdown('<div class="anchor" id="kelas"></div><div style="height:30px"></div>', unsafe_allow_html=True)
    sub = f"{len(cocok)} dari {len(clf.class_names)} kelas" if q else f"{len(clf.class_names)} jenis buah yang bisa dikenali."
    st.markdown(judul_bagian("grid_view", "Kelas yang dikenali", sub, "lime"), unsafe_allow_html=True)
    if cocok:
        chips = "".join(f'<span class="chip">{html.escape(c.replace("_", " ").title())}</span>' for c in cocok)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="none">Tidak ada kelas yang cocok. Coba kata kunci lain, misalnya nama buah dalam bahasa Inggris.</div>',
                    unsafe_allow_html=True)


st.markdown(f'<div class="foot">Proyek Akhir Kecerdasan Buatan, {html.escape(NAMA_KELOMPOK)}</div>', unsafe_allow_html=True)
