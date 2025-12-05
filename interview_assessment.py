import streamlit as st
import requests
import re

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

HF_TOKEN = st.secrets["HF_TOKEN"]
HF_WHISPER_MODEL = "NbAiLab/nb-whisper-medium"  # nama model saja
HF_MISTRAL_MODEL = "nndayoow/mistral-interview-lora"  # nama model saja

INTERVIEW_QUESTIONS = [
    "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
]

CRITERIA_TEXT = (
    "Kriteria Penilaian:\n"
    "0 - Tidak menjawab\n"
    "1 - Tidak relevan\n"
    "2 - Paham general\n"
    "3 - Pemahaman cukup\n"
    "4 - Menguasai materi\n"
)

# -----------------------
# FUNCTIONS
# -----------------------
def transcribe_via_hf(video_bytes):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"file": ("video.mp4", video_bytes)}
    url = f"https://api-inference.huggingface.co/models/{HF_WHISPER_MODEL}"
    try:
        response = requests.post(url, headers=headers, files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "") or data.get("error", "")
    except requests.exceptions.RequestException as e:
        return f"ERROR: {str(e)}"

def mistral_lora_api(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_MISTRAL_MODEL}", 
                                 headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        elif isinstance(result, dict):
            return result.get("generated_text", str(result))
        else:
            return str(result)
    except requests.exceptions.RequestException as e:
        return f"ERROR: {str(e)}"

def prompt_for_classification(question, answer):
    return (
        f"Anda adalah HRD profesional. Nilailah jawaban kandidat dengan skala 0–5.\n\n"
        f"{CRITERIA_TEXT}\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban Kandidat: {answer}\n\n"
        "Format Output:\nKLASIFIKASI: <angka>\nALASAN: <teks>\n"
    )

def parse_model_output(text):
    score_match = re.search(r"\b([0-5])\b", text)
    score = int(score_match.group(1)) if score_match else None
    reason_match = re.search(r"(ALASAN|REASON)[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(2).strip() if reason_match else text.strip()
    return score, reason

# -----------------------
# SESSION STATE INIT
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "input"
if "results" not in st.session_state:
    st.session_state.results = []
if "nama" not in st.session_state:
    st.session_state.nama = ""
if "rerun_done" not in st.session_state:
    st.session_state.rerun_done = False

# -----------------------
# PAGE: INPUT
# -----------------------

if st.session_state.page == "input":
    st.title("🎥 AI-Powered Interview Assessment System")
    st.write("Upload **5 video interview** lalu klik mulai analisis.")

    with st.form("input_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 Video (1 → 5)", 
            type=["mp4", "mov", "mkv", "webm"], 
            accept_multiple_files=True
        )
        submit = st.form_submit_button("Mulai Proses Analisis")

    if submit:
        error_flag = False
        if not nama:
            st.error("Nama pelamar wajib diisi.")
            error_flag = True
        if not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
            error_flag = True

        if not error_flag:
            st.session_state.nama = nama
            st.session_state.results = []
            st.session_state.processing_done = True  # Flag untuk pindah ke result
            st.session_state.page = "result"

# -----------------------
# PAGE: RESULT
# -----------------------
if st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.write(f"**Nama Pelamar:** {st.session_state.nama}")

    valid_scores = [r["score"] for r in st.session_state.results if r["score"] is not None]
    if len(valid_scores) == 5:
        final_score = sum(valid_scores) / 5
        st.markdown(f"### Skor Akhir: **{final_score:.2f} / 5**")
    else:
        st.error("Skor tidak lengkap.")

    st.markdown("---")
    for i, r in enumerate(st.session_state.results):
        st.subheader(f"🎬 Video {i+1}")
        st.write(f"**Pertanyaan:** {r['question']}")
        st.write(f"**Transkrip:** {r['transcript']}")
        st.write(f"**Skor:** {r['score']}")
        st.write(f"**Alasan:** {r['reason']}")
        with st.expander("Raw Output Model"):
            st.code(r["raw_model"])
        st.markdown("---")

    if st.button("Kembali ke Input"):
        st.session_state.page = "input"
        st.session_state.results = []
        st.session_state.nama = ""
        st.session_state.rerun_done = False
        st.experimental_rerun()

