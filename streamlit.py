import streamlit as st
import tempfile
import os
import whisper
import torch
import re
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "nndayoow/mistral-interview-lora"

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
# GPU CLEANER
# -----------------------
def clear_gpu():
    """Force clear VRAM so models can be loaded again safely."""
    try:
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect may not exist on very old torch versions, guard it
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass

# -----------------------
# CACHED LOADER (Whisper - medium)
# -----------------------
@st.cache_resource
def load_whisper():
    # Using 'medium' to reduce VRAM usage on Colab
    return whisper.load_model("medium")

# -----------------------
# CLASSIFIER LOADER (NO CACHE) - load on CPU to avoid OOM
# -----------------------
def load_classifier():
    clear_gpu()  # try to free GPU before heavy load
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Force CPU to avoid CUDA OOM on limited GPUs
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cpu"  # load on CPU (slower but safe)
    )

    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    try:
        ft_model = ft_model.to("cpu")
    except Exception:
        pass
    return tokenizer, ft_model

# -----------------------
# HELPERS
# -----------------------
def extract_audio(video_path):
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    (
        ffmpeg.input(video_path)
        .output(out_wav, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return out_wav

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
    m = re.search(r"\b([0-5])\b", text)
    score = int(m.group(1)) if m else None

    m2 = re.search(r"(ALASAN|REASON)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    reason = m2.group(2).strip() if m2 else text.strip()

    return score, reason

def classify_with_model(tokenizer, model, question, answer):
    prompt = prompt_for_classification(question, answer)
    chat = [{"role":"user", "content": prompt}]
    decoded = ""
    try:
        # Prefer apply_chat_template if the tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(model.device)
        else:
            # fallback: basic tokenization
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            input_ids["input_ids"] if isinstance(input_ids, dict) else input_ids,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        decoded = f"ERROR DURING GENERATION: {e}"

    score, reason = parse_model_output(decoded)
    return decoded, score, reason

# -----------------------
# SESSION STATE
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "input"
if "results" not in st.session_state:
    st.session_state.results = []
if "progress" not in st.session_state:
    st.session_state.progress = {
        "load_whisper": False,
        "load_classifier": False,
        "process_videos": [False]*len(INTERVIEW_QUESTIONS),
        "completed": False
    }

# -----------------------
# INPUT PAGE
# -----------------------
st.title("🎥 AI-Powered Interview Assessment System")

if st.session_state.page == "input":
    st.write("Upload 5 video lalu tekan **Mulai Proses Analisis**")

    with st.form("input_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 video interview (urut dari pertanyaan 1→5)",
            type=["mp4","mov","mkv","webm"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("Mulai Proses Analisis")

    if submitted:
        if not nama:
            st.error("Harap isi Nama Pelamar.")
            st.stop()

        if not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
            st.stop()

        # Reset
        st.session_state.results = []
        st.session_state.progress = {
            "load_whisper": False,
            "load_classifier": False,
            "process_videos": [False]*5,
            "completed": False
        }

        status_container = st.empty()
        with status_container.container():
            st.markdown("### ▶️ Proses dimulai...")

            # Load whisper
            step1 = st.empty()
            step1.info("⏳ Loading Whisper (medium)...")
            try:
                whisper_model = load_whisper()
                st.session_state.progress["load_whisper"] = True
                step1.success("✅ Whisper loaded.")
            except Exception as e:
                step1.error(f"❌ Error loading Whisper: {e}")
                st.stop()

            # Load classifier
            step2 = st.empty()
            step2.info("⏳ Loading classifier model (on CPU)...")
            try:
                tokenizer, ft_model = load_classifier()
                st.session_state.progress["load_classifier"] = True
                step2.success("✅ Classifier loaded (CPU).")
            except Exception as e:
                step2.error(f"❌ Gagal load classifier: {e}")
                st.stop()

            time.sleep(0.3)

            # Process each video
            for idx, vid in enumerate(uploaded):
                vid_box = st.empty()
                vid_box.info(f"⏳ Memproses Video {idx+1}...")

                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmpf.write(vid.read())
                tmpf.close()
                tmp_path = tmpf.name

                wav = None
                trans = ""
                try:
                    wav = extract_audio(tmp_path)
                    trans = whisper_model.transcribe(wav, language="en", verbose=False)["text"]
                except Exception as e:
                    trans = ""
                    vid_box.error(f"❌ Gagal memproses (transcribe) video {idx+1}: {e}")

                try:
                    raw_out, score, reason = classify_with_model(
                        tokenizer, ft_model, INTERVIEW_QUESTIONS[idx], trans
                    )
                except Exception as e:
                    raw_out, score, reason = (f"Error: {e}", None, f"Error: {e}")

                st.session_state.results.append({
                    "question": INTERVIEW_QUESTIONS[idx],
                    "transcript": trans,
                    "score": score,
                    "reason": reason,
                    "raw_model": raw_out
                })

                st.session_state.progress["process_videos"][idx] = True
                vid_box.success(f"✅ Video {idx+1} selesai.")

                # cleanup temps safely
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                try:
                    if wav and os.path.exists(wav):
                        os.remove(wav)
                except Exception:
                    pass

            st.session_state.progress["completed"] = True
            status_container.success("🎉 Semua video selesai.")

            # 🧹 CLEAR GPU setelah selesai
            clear_gpu()

            st.session_state.page = "result"
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

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
        st.write(f"**Transkrip:** {r['transcript'] or '(kosong)'}")
        st.write(f"**Skor:** {r['score']}")
        st.write(f"**Alasan:** {r['reason']}")
        with st.expander("Raw Output Model"):
            st.code(r["raw_model"][:800])
        st.markdown("---")

    if st.button("Kembali ke Halaman Input"):
        st.session_state.page = "input"
        st.session_state.results = []
        # 🧹 Clear VRAM juga saat kembali
        clear_gpu()
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass
