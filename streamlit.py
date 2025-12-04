import streamlit as st
import tempfile
import os
import re
import requests
import time

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

HF_API_URL = "https://api-inference.huggingface.co/models/nndayoow/mistral-interview-lora"
HF_TOKEN = st.secrets["HF_TOKEN"]
OPENAI_KEY = st.secrets["OPENAI_KEY"]

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
    "5 - Sangat menguasai\n"
)

# -----------------------
# API FUNCTIONS
# -----------------------

def whisper_api_transcribe(video_path):
    """Transcribe audio/video using OpenAI Whisper API."""
    url = "https://api.openai.com/v2/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}

    files = {"file": open(video_path, "rb")}
    data = {"model": "whisper-1"}

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        return f"ERROR Whisper: {response.text}"

    return response.json().get("text", "")


def mistral_lora_api(prompt):
    """Call fine-tuned Mistral model on HuggingFace Inference API."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }

    response = requests.post(HF_API_URL, json=payload, headers=headers)

    try:
        result = response.json()
    except:
        return "ERROR: Response tidak bisa dibaca."

    if isinstance(result, list):
        return result[0].get("generated_text", "")

    return result.get("generated_text", str(result))


# -----------------------
# HELPERS
# -----------------------

def prompt_for_classification(question, answer):
    return (
        "Anda adalah HRD profesional. Nilailah jawaban kandidat dengan skala 0–5.\n\n"
        f"{CRITERIA_TEXT}\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban Kandidat: {answer}\n\n"
        "Format Output:\n"
        "KLASIFIKASI: <angka>\n"
        "ALASAN: <teks>\n"
    )


def parse_model_output(text):
    """Extract score (0–5) and reason."""
    score = None

    score_match = re.search(r"\b([0-5])\b", text)
    if score_match:
        score = int(score_match.group(1))

    reason_match = re.search(r"(ALASAN|REASON)[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(2).strip() if reason_match else text

    return score, reason


# -----------------------
# SESSION STATE INIT
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

if "results" not in st.session_state:
    st.session_state.results = []


# -----------------------
# PAGE: INPUT
# -----------------------
if st.session_state.page == "input":

    st.title("🎥 AI-Powered Interview Assessment System")
    st.write("Upload **5 video interview** lalu klik mulai analisis.")

    with st.form("input_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader("Upload 5 Video (1 → 5)", type=["mp4", "mov", "mkv", "webm"], accept_multiple_files=True)
        submit = st.form_submit_button("Mulai Proses Analisis")

    if submit:

        if not nama:
            st.error("Nama pelamar wajib diisi.")
            st.stop()

        if not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
            st.stop()

        st.session_state.results = []

        progress = st.empty()

        # PROCESS EACH VIDEO
        for idx, vid in enumerate(uploaded):
            progress.info(f"Memproses Video {idx+1}...")

            # Simpan sementara
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vid.read())
            tmp.close()

            video_path = tmp.name

            # Transkripsi
            transcript = whisper_api_transcribe(video_path)

            # Klasifikasi jawaban
            prompt = prompt_for_classification(INTERVIEW_QUESTIONS[idx], transcript)
            raw_output = mistral_lora_api(prompt)
            score, reason = parse_model_output(raw_output)

            # Simpan hasil
            st.session_state.results.append({
                "question": INTERVIEW_QUESTIONS[idx],
                "transcript": transcript,
                "score": score,
                "reason": reason,
                "raw_model": raw_output
            })

            os.remove(video_path)

            progress.success(f"Video {idx+1} selesai ✔")

        st.session_state.page = "result"
        st.rerun()


# -----------------------
# PAGE: RESULT
# -----------------------
if st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.write(f"**Nama Pelamar:** {nama}")

    # Hitung skor akhir
    valid_scores = [r["score"] for r in st.session_state.results if r["score"] is not None]
    if len(valid_scores) == 5:
        final_score = sum(valid_scores) / 5
        st.markdown(f"### Skor Akhir: **{final_score:.2f} / 5**")
    else:
        st.error("Skor tidak lengkap.")

    st.markdown("---")

    # Detail
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
        st.rerun()
