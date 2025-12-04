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

HF_API_URL = st.secrets["HF_API_URL"]
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
    "Kriteria:\n"
    "0 - Tidak menjawab\n"
    "1 - Tidak relevan\n"
    "2 - Paham general\n"
    "3 - Pemahaman cukup\n"
    "4 - Menguasai materi\n"
)

# -----------------------
# API FUNCTIONS
# -----------------------

def whisper_api_transcribe(video_path):
    """Send audio/video to Whisper API (OpenAI)."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}

    files = {"file": open(video_path, "rb")}
    data = {"model": "whisper-1"}

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        return f"ERROR Whisper: {response.text}"

    return response.json()["text"]


def mistral_lora_api(prompt):
    """Send classification prompt to HuggingFace LoRA Endpoint."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }

    response = requests.post(HF_API_URL, json=payload, headers=headers)

    try:
        result = response.json()
    except Exception:
        return "ERROR: Bad JSON response"

    # HF inference API returns list sometimes
    if isinstance(result, list):
        return result[0]["generated_text"]

    if "generated_text" in result:
        return result["generated_text"]

    return str(result)


# -----------------------
# HELPERS
# -----------------------
def prompt_for_classification(question, answer):
    return (
        "Anda adalah penilai HRD. Klasifikasikan jawaban kandidat dengan skala 0-5.\n\n"
        f"{CRITERIA_TEXT}\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban Kandidat: {answer}\n\n"
        "Format output:\n"
        "KLASIFIKASI: <angka>\n"
        "ALASAN: <penjelasan singkat>\n"
    )


def parse_model_output(text):
    score_match = re.search(r"\b([0-5])\b", text)
    score = int(score_match.group(1)) if score_match else None

    reason_match = re.search(r"(ALASAN|REASON)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(2).strip() if reason_match else text.strip()

    return score, reason


# -----------------------
# SESSION STATE
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "input"
if "results" not in st.session_state:
    st.session_state.results = []


# -----------------------
# INPUT PAGE
# -----------------------
st.title("🎥 AI-Powered Interview Assessment System")

if st.session_state.page == "input":
    st.write("Upload 5 video lalu tekan **Mulai Proses Analisis**")

    with st.form("input_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 video interview (urut dari 1 → 5)",
            type=["mp4", "mov", "mkv", "webm"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("Mulai Proses Analisis")

    if submitted:
        if not nama:
            st.error("Nama harus diisi!")
            st.stop()

        if not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
            st.stop()

        st.session_state.results = []
        progress = st.empty()

        # PROCESS EACH VIDEO
        for idx, vid in enumerate(uploaded):
            progress.info(f"⏳ Memproses video {idx+1}...")

            # Save temporarily
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vid.read())
            tmp.close()
            video_path = tmp.name

            # 1 → TRANSCRIBE (WHISPER API)
            try:
                transcript = whisper_api_transcribe(video_path)
            except Exception as e:
                transcript = ""
                progress.error(f"❌ Error Whisper Video {idx+1}: {e}")

            # 2 → CLASSIFY (MISTRAL LORA API)
            prompt = prompt_for_classification(INTERVIEW_QUESTIONS[idx], transcript)
            raw_output = mistral_lora_api(prompt)
            score, reason = parse_model_output(raw_output)

            st.session_state.results.append({
                "question": INTERVIEW_QUESTIONS[idx],
                "transcript": transcript,
                "score": score,
                "reason": reason,
                "raw_model": raw_output
            })

            # cleanup
            try:
                os.remove(video_path)
            except:
                pass

            progress.success(f"Video {idx+1} selesai ✔")

        st.session_state.page = "result"
        st.rerun()



# -----------------------
# RESULT PAGE
# -----------------------
if st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.subheader(f"Nama Pelamar: {nama}")

    scores = [r["score"] for r in st.session_state.results if r["score"] is not None]
    if len(scores) == 5:
        overall = sum(scores) / 5
        st.markdown(f"### Skor Akhir: **{overall:.2f} / 5**")
    else:
        st.markdown("### Skor tidak lengkap")

    st.markdown("---")
    st.markdown("## Detail Penilaian")

    for i, r in enumerate(st.session_state.results):
        st.markdown(f"### 🎬 Video {i+1}")
        st.write(f"**Pertanyaan:** {r['question']}")
        st.write(f"**Transkrip:** {r['transcript']}")
        st.write(f"**Skor:** {r['score']}")
        st.write(f"**Alasan:** {r['reason']}")
        with st.expander("Raw Output Model"):
            st.code(r["raw_model"])

        st.markdown("---")

    if st.button("Kembali ke Halaman Input"):
        st.session_state.page = "input"
        st.session_state.results = []
        st.rerun()
