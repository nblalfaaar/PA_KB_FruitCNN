"""Aplikasi web klasifikasi buah (Streamlit) - Proyek Akhir KB.
Ubah bagian KONFIGURASI di bawah untuk menyesuaikan judul, info model, dan perilaku aplikasi.
Tampilan: palet Deep Blueberry, Strawberry Delight, Lime Zest, Golden Honey; font Poppins; ikon Material Symbols.
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
  --blue:#2D3E56; --blue-2:#3B5070; --blue-soft:#E3E8EF;
  --red:#FF3131; --red-dark:#C71F1F;
  --lime:#9ACD32; --lime-dark:#4A6B05;
  --honey:#FFD166; --honey-dark:#8A5F00;
  --bg:#F1F0EC; --card:#FFFFFF; --line:#E4E2DB; --muted:#6B7889;
}

/* ---------- Font ---------- */
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp button, .stApp input,
.stApp li, .stApp a, .stApp summary, .stApp small, .stApp b, .stApp strong, .stApp h1, .stApp h2, .stApp h3{
  font-family:'Poppins', sans-serif;
}
.stApp [data-testid="stIconMaterial"], .stApp .mi{
  font-family:'Material Symbols Rounded' !important;
  font-weight:normal; font-style:normal; letter-spacing:normal; text-transform:none;
  line-height:1; white-space:nowrap; font-feature-settings:'liga'; -webkit-font-smoothing:antialiased;
  font-variation-settings:'FILL' 1,'wght' 500,'GRAD' 0,'opsz' 24;
}
.mi{display:inline-block; font-size:1.3rem; vertical-align:middle}

/* ---------- Kerangka halaman ---------- */
.stApp{background:var(--bg); color:var(--blue)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]{display:none}
.block-container{max-width:1180px; padding:2.2rem 1.5rem 3rem}

/* ---------- Hero ---------- */
.hero{
  background:var(--blue); color:#fff; border-radius:32px; padding:44px 48px;
  display:grid; grid-template-columns:1.1fr .9fr; gap:40px; align-items:center; margin-bottom:28px;
}
.badge{
  display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,.12);
  color:#fff; border-radius:999px; padding:7px 16px 7px 12px; font-size:.82rem; font-weight:500;
}
.badge .mi{font-size:1.1rem; color:var(--honey)}
.hero-title{font-size:3rem; font-weight:700; line-height:1.12; margin:22px 0 14px; letter-spacing:-.02em; color:#fff}
.hero-sub{font-size:1rem; line-height:1.7; color:rgba(255,255,255,.78); max-width:30rem}
.tiles{display:grid; grid-template-columns:1fr 1fr; gap:14px}
.tile{border-radius:22px; padding:18px 18px 20px; min-height:132px; display:flex; flex-direction:column; justify-content:space-between}
.tile .mi{font-size:1.6rem}
.tile-val{font-size:1.08rem; font-weight:600; line-height:1.3; margin-top:12px}
.tile-key{font-size:.76rem; font-weight:500; opacity:.75; margin-top:2px}
.t-lime{background:var(--lime); color:var(--blue)}
.t-honey{background:var(--honey); color:var(--blue)}
.t-red{background:var(--red); color:#fff}
.t-paper{background:#F6F5F1; color:var(--blue)}

/* ---------- Panel ---------- */
.st-key-panel_input, .st-key-panel_result{
  background:var(--card); border-radius:28px; padding:28px 28px 26px;
  box-shadow:0 1px 2px rgba(45,62,86,.06), 0 12px 32px -12px rgba(45,62,86,.14);
}
.panel-title{display:flex; align-items:center; gap:10px; font-size:1.15rem; font-weight:600; margin-bottom:4px}
.panel-title .ico{width:38px; height:38px; border-radius:12px; display:grid; place-items:center; background:var(--blue); color:#fff}
.panel-title .ico .mi{font-size:1.25rem}
.panel-title .ico.honey{background:var(--honey); color:var(--blue)}
.panel-sub{font-size:.86rem; color:var(--muted); margin:0 0 14px 48px}

/* ---------- Tab ---------- */
[data-baseweb="tab-list"]{gap:8px; background:#F1F0EC; padding:5px; border-radius:999px; width:fit-content}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none}
button[data-baseweb="tab"]{border-radius:999px; padding:8px 18px; height:auto; background:transparent}
button[data-baseweb="tab"] p{font-size:.88rem; font-weight:500; color:var(--blue)}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--blue)}
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] *{color:#fff !important}
[data-baseweb="tab-panel"]{padding-top:16px}

/* ---------- Unggah dan kamera ---------- */
[data-testid="stFileUploaderDropzone"]{
  background:#F8F7F3; border:2px dashed rgba(45,62,86,.35); border-radius:20px; padding:26px 20px;
  transition:border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--red); background:#FFF6F5}
[data-testid="stFileUploader"] label p, [data-testid="stCameraInput"] label p{font-size:.88rem; font-weight:500}
button[data-testid^="stBaseButton-secondary"]{
  background:var(--blue); color:#fff; border:0; border-radius:999px; padding:8px 20px; font-weight:500;
  transition:background .2s, color .2s;
}
button[data-testid^="stBaseButton-secondary"] *{color:inherit}
button[data-testid^="stBaseButton-secondary"]:hover{background:var(--honey); color:var(--blue)}
button[data-testid^="stBaseButton-secondary"]:focus-visible{outline:3px solid var(--honey); outline-offset:2px}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{border-radius:20px}
[data-testid="stImage"] img{border-radius:20px}
[data-testid="stImageCaption"]{color:var(--muted); font-size:.8rem}
[data-testid="stToggle"] label p{font-size:.88rem; font-weight:500}

/* ---------- Hasil ---------- */
.res-head{display:flex; align-items:center; gap:26px; margin:10px 0 24px; flex-wrap:wrap}
.ring{width:132px; height:132px; border-radius:50%; flex:none; display:grid; place-items:center;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), #ECEAE4 0)}
.ring-in{width:102px; height:102px; border-radius:50%; background:#fff; display:flex; align-items:baseline;
  justify-content:center; padding-top:34px; box-sizing:border-box}
.ring-in b{font-size:1.65rem; font-weight:700; line-height:1}
.ring-in small{font-size:.8rem; font-weight:600; margin-left:2px}
.res-label{font-size:.82rem; color:var(--muted); font-weight:500}
.res-name{font-size:2.1rem; font-weight:700; line-height:1.15; letter-spacing:-.01em; margin:2px 0 12px}
.status{display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:6px 14px 6px 10px; font-size:.82rem; font-weight:600}
.status .mi{font-size:1.1rem}
.s-high{background:rgba(154,205,50,.25); color:var(--lime-dark)}
.s-mid{background:rgba(255,209,102,.4); color:var(--honey-dark)}
.s-low{background:rgba(255,49,49,.12); color:var(--red-dark)}
.callout{display:flex; gap:12px; align-items:flex-start; background:rgba(255,49,49,.08);
  border-left:4px solid var(--red); border-radius:14px; padding:14px 16px; font-size:.86rem; line-height:1.6; margin-bottom:22px; color:var(--blue)}
.callout .mi{color:var(--red); font-size:1.3rem}

.list-title{font-size:.95rem; font-weight:600; margin:6px 0 12px}
.row{display:grid; grid-template-columns:minmax(90px,150px) 1fr 58px; gap:14px; align-items:center; margin-bottom:12px}
.row-name{font-size:.88rem; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.track{height:12px; background:#ECEAE4; border-radius:999px; overflow:hidden}
.fill{height:100%; border-radius:999px; width:var(--w); animation:grow .8s ease-out}
.row-val{font-size:.85rem; font-weight:600; text-align:right}
@keyframes grow{from{width:0}}
.row.top .row-name{font-weight:700}
.note{font-size:.78rem; color:var(--muted); line-height:1.6; margin-top:10px}

/* ---------- Keadaan kosong ---------- */
.empty{border:2px dashed var(--line); border-radius:22px; padding:34px 28px; text-align:center}
.empty-ico{width:68px; height:68px; border-radius:20px; background:var(--honey); color:var(--blue);
  display:grid; place-items:center; margin:0 auto 16px}
.empty-ico .mi{font-size:2rem}
.empty-t{font-size:1.1rem; font-weight:600; margin-bottom:6px}
.empty-d{font-size:.88rem; color:var(--muted); margin-bottom:20px}
.tips{display:grid; gap:10px; text-align:left; max-width:340px; margin:0 auto}
.tip{display:flex; gap:10px; align-items:center; font-size:.86rem}
.tip .mi{color:var(--lime); font-size:1.3rem}

/* ---------- Daftar kelas ---------- */
[data-testid="stExpander"]{background:var(--card); border:0; border-radius:22px; margin-top:24px;
  box-shadow:0 1px 2px rgba(45,62,86,.06), 0 12px 32px -12px rgba(45,62,86,.14)}
[data-testid="stExpander"] summary{padding:16px 22px}
[data-testid="stExpander"] summary p{font-weight:600; font-size:.95rem}
.chips{display:flex; flex-wrap:wrap; gap:8px; padding:2px 4px 8px}
.chip{background:var(--blue-soft); color:var(--blue); border-radius:999px; padding:6px 14px; font-size:.8rem; font-weight:500}
.chip:hover{background:var(--honey)}

.foot{text-align:center; font-size:.8rem; color:var(--muted); margin-top:30px}

/* ---------- Responsif ---------- */
@media (max-width: 860px){
  .hero{grid-template-columns:1fr; padding:30px 24px; gap:28px}
  .hero-title{font-size:2.2rem}
  .res-name{font-size:1.7rem}
  .block-container{padding:1.2rem .9rem 2rem}
  .st-key-panel_input, .st-key-panel_result{padding:22px 18px}
}
@media (prefers-reduced-motion: reduce){ .fill{animation:none} }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def h(teks: str) -> str:
    """Rapikan HTML satu baris supaya tidak dibaca Markdown sebagai blok kode."""
    return "".join(baris.strip() for baris in teks.splitlines())


def ikon(nama: str) -> str:
    return f'<span class="mi">{nama}</span>'


@st.cache_resource
def muat_model():
    return FruitClassifier("fruit_model.tflite", "class_names.json", "config.json")


clf = muat_model()

# ============================== HERO ==============================
tiles = list(INFO_MODEL.items()) + [("Ukuran input", f"{clf.img_size} x {clf.img_size} piksel")]
gaya_tile = [("t-lime", "category"), ("t-honey", "verified"), ("t-red", "memory"), ("t-paper", "photo_size_select_large")]
html_tile = "".join(
    f'<div class="tile {kelas}">{ikon(ic)}<div><div class="tile-val">{html.escape(str(v))}</div>'
    f'<div class="tile-key">{html.escape(k)}</div></div></div>'
    for (k, v), (kelas, ic) in zip(tiles[:4], gaya_tile)
)
st.markdown(h(f"""
<div class="hero">
  <div>
    <span class="badge">{ikon("nutrition")}{html.escape(NAMA_KELOMPOK)}</span>
    <div class="hero-title">{html.escape(JUDUL)}</div>
    <div class="hero-sub">{html.escape(SUBJUDUL)}</div>
  </div>
  <div class="tiles">{html_tile}</div>
</div>
"""), unsafe_allow_html=True)

# ============================== KONTEN ==============================
kiri, kanan = st.columns(2, gap="large")

gambar = None
with kiri:
    with st.container(key="panel_input"):
        st.markdown(h(f"""
        <div class="panel-title"><span class="ico">{ikon("add_photo_alternate")}</span>Foto buah</div>
        <div class="panel-sub">Gunakan satu jenis buah per foto.</div>
        """), unsafe_allow_html=True)

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

with kanan:
    with st.container(key="panel_result"):
        st.markdown(h(f"""
        <div class="panel-title"><span class="ico honey">{ikon("insights")}</span>Hasil prediksi</div>
        <div class="panel-sub">Lima kelas dengan peluang tertinggi.</div>
        """), unsafe_allow_html=True)

        if gambar is None:
            st.markdown(h(f"""
            <div class="empty">
              <div class="empty-ico">{ikon("image_search")}</div>
              <div class="empty-t">Belum ada foto</div>
              <div class="empty-d">Unggah foto atau ambil foto buah, hasil prediksi muncul di sini.</div>
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
                warna, kelas_s, ic_s, label_s = LIME, "s-high", "verified", "Yakin tinggi"
            elif prob >= AMBANG_YAKIN:
                warna, kelas_s, ic_s, label_s = HONEY, "s-mid", "info", "Cukup yakin"
            else:
                warna, kelas_s, ic_s, label_s = STRAWBERRY, "s-low", "warning", "Kurang yakin"

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
                isi = warna if i == 0 else "#B7C1CF"
                baris += (
                    f'<div class="row{" top" if i == 0 else ""}">'
                    f'<div class="row-name">{html.escape(n.replace("_", " ").title())}</div>'
                    f'<div class="track"><div class="fill" style="--w:{max(p * 100, 1.5):.1f}%; background:{isi}"></div></div>'
                    f'<div class="row-val">{p * 100:.1f}%</div></div>'
                )
            st.markdown(h(f'<div class="list-title">Top-{JUMLAH_TOP} prediksi</div>{baris}'), unsafe_allow_html=True)

            with st.expander("Lihat tabel probabilitas", icon=":material/table_chart:"):
                df = pd.DataFrame({"Kelas": [n.replace("_", " ") for n, _ in hasil],
                                   "Probabilitas (%)": [round(p * 100, 2) for _, p in hasil]})
                st.dataframe(df, hide_index=True, width="stretch")

            st.markdown('<div class="note">Catatan: model hanya mengenal kelas yang terdaftar. '
                        'Buah di luar daftar tetap akan dipetakan ke salah satu kelas.</div>', unsafe_allow_html=True)

# ============================== DAFTAR KELAS ==============================
with st.expander(f"Daftar {len(clf.class_names)} kelas yang dikenali", icon=":material/grid_view:"):
    chips = "".join(f'<span class="chip">{html.escape(c.replace("_", " ").title())}</span>' for c in clf.class_names)
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

st.markdown(f'<div class="foot">Proyek Akhir Kecerdasan Buatan, {html.escape(NAMA_KELOMPOK)}</div>', unsafe_allow_html=True)
