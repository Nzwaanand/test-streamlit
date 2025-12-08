import streamlit as st
import re
import tempfile
import os
from transformers import pipeline
import subprocess

# ========================= CONFIG =========================
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

# Model yang benar & aman untuk CPU
HF_PHI3_MODEL = "microsoft/Phi-3.5-mini-instruct"
HF_WHISPER_MODEL = "openai/whisper-large-v3-turbo"

INTERVIEW_QUESTIONS = [
    "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
]

CRITERIA = (
    "Kriteria Penilaian:\n"
    "0 - not answer the question\n"
    "1 - the answer is not relevan for question\n"
    "2 - Understand for general question\n"
    "3 - Understand with practice solution\n"
    "4 - Deep understanding with inovative solution\n"
)

# ========================= SAFE MODEL LOADING =========================
@st.cache_resource
def get_asr_pipeline():
    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=HF_WHISPER_MODEL
        )
    except Exception as e:
        st.error(f"Gagal memuat model Whisper: {e}")
        return None


@st.cache_resource
def get_llm_pipeline():
    try:
        return pipeline(
            task="text-generation",
            model=HF_PHI3_MODEL
        )
    except Exception as e:
        st.error(f"Gagal memuat model Phi-3: {e}")
        return None


# ========================= FUNCTIONS =========================
def transcribe_via_hf(video_bytes):
    asr = get_asr_pipeline()
    if asr is None:
        return "ERROR: Model Whisper gagal dimuat."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in.flush()
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tmp_out_path = tmp_out.name

    # Cek ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return "ERROR: ffmpeg tidak terpasang di server."

    # Convert ke WAV mono 16k
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ac", "1", "-ar", "16000", tmp_out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except Exception as e:
        return f"ERROR FFMPEG: {e}"

    # Transcribe
    try:
        result = asr(tmp_out_path)
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result)
    except Exception as e:
        return f"ERROR TRANSCRIBE: {e}"
    finally:
        try: os.remove(tmp_in_path)
        except: pass
        try: os.remove(tmp_out_path)
        except: pass


def phi3_api(prompt):
    llm = get_llm_pipeline()
    if llm is None:
        return "ERROR: Model Phi-3 gagal dimuat."

    try:
        out = llm(prompt, max_new_tokens=200, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0]["generated_text"]
        return str(out)
    except Exception as e:
        return f"ERROR LLM: {e}"


def prompt_for_classification(question, answer):
    return (
        "You are an expert HR interviewer and technical evaluator. Your task is to objectively assess the "
        "candidate's response based solely on the provided transcript. You must classify the answer using a strict "
        "0 until 4 scoring rubric.\n\n"
        f"{CRITERIA}\n\n"
        "Evaluation Rules:\n"
        "- Evaluate ONLY based on the candidate's answer.\n"
        "- Do NOT add missing information, assumptions, or corrections.\n"
        "- Judge relevance, accuracy, clarity, and depth based on the rubric.\n"
        "- Your explanation must be concise and directly tied to the rubric.\n"
        "- You MUST follow the output format exactly.\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate Answer (Transcript):\n{answer}\n\n"
        "Required Output Format:\n"
        "KLASIFIKASI: <angka>\n"
        "ALASAN: <teks>\n"
    )


def parse_model_output(text):
    score_match = re.search(r"KLASIFIKASI[:\- ]*([0-4])", text, re.IGNORECASE)
    if not score_match:
        score_match = re.search(r"\b([0-4])\b", text)
    score = int(score_match.group(1)) if score_match else None

    reason_match = re.search(r"ALASAN[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text

    return score, reason


# ========================= SESSION STATE =========================
for key, default in {
    "page": "input",
    "results": [],
    "nama": "",
    "processing_done": False
}.items():
    st.session_state.setdefault(key, default)


# ========================= PAGE INPUT =========================
if st.session_state.page == "input":
    st.title("🎥 AI-Powered Interview Assessment System")
    st.write("Upload **5 video interview** lalu klik mulai analisis.")

    with st.form("upload_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 Video (1 → 5)",
            type=["mp4", "mov", "mkv", "webm"],
            accept_multiple_files=True
        )
        submit = st.form_submit_button("Mulai Proses Analisis")

    if submit:
        if not nama:
            st.error("Nama wajib diisi.")
        elif not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
        else:
            st.session_state.nama = nama
            st.session_state.uploaded = uploaded
            st.session_state.results = []
            st.session_state.page = "result"
            st.session_state.processing_done = True
            st.rerun()


# ========================= PAGE RESULT =========================
if st.session_state.processing_done and st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.write(f"**Nama Pelamar:** {st.session_state.nama}")

    progress = st.empty()

    if len(st.session_state.results) == 0:
        for idx, vid in enumerate(st.session_state.uploaded):
            progress.info(f"Memproses Video {idx+1}...")

            bytes_data = vid.read()
            transcript = transcribe_via_hf(bytes_data)
            prompt = prompt_for_classification(INTERVIEW_QUESTIONS[idx], transcript)
            raw_output = phi3_api(prompt)
            score, reason = parse_model_output(raw_output)

            st.session_state.results.append({
                "question": INTERVIEW_QUESTIONS[idx],
                "transcript": transcript,
                "score": score,
                "reason": reason,
                "raw_model": raw_output
            })

            progress.success(f"Video {idx+1} selesai ✔")

    scores = [r["score"] for r in st.session_state.results if r["score"] is not None]
    if len(scores) == 5:
        final_score = sum(scores) / 5
        st.markdown(f"### ⭐ Skor Akhir: **{final_score:.2f} / 4**")
    else:
        st.error("Skor tidak semua berhasil diproses. Cek raw output model.")

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

    if st.button("🔙 Kembali"):
        st.session_state.page = "input"
        st.session_state.processing_done = False
        st.session_state.results = []
        st.session_state.nama = ""
        st.rerun()
