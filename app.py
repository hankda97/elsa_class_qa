"""Streamlit FAQ assistant backed by semantic search over QA.csv."""

import csv
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CSV_PATH = Path(os.environ.get("QA_CSV_PATH", BASE_DIR / "QA.csv"))
MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.30"))


@st.cache_resource(show_spinner="正在載入語意模型（首次啟動可能需要一些時間）…")
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner="正在讀取客服知識庫並建立搜尋索引…")
def get_knowledge_base(csv_path: str, model_name: str):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到知識庫檔案：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        faqs = list(csv.DictReader(file))

    required = {"question", "category", "product", "symptoms", "answer", "urgency"}
    if not faqs or not required.issubset(faqs[0]):
        raise ValueError("QA.csv 必須包含 question、category、product、symptoms、answer、urgency 欄位。")

    texts = [
        f"問題：{row['question']} 分類：{row['category']} 產品：{row['product']} 症狀：{row['symptoms']}"
        for row in faqs
    ]
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return faqs, vectors


def answer_question(question: str, faqs: list[dict[str, str]], vectors: np.ndarray) -> dict:
    model = get_model()
    query_vector = model.encode(question, normalize_embeddings=True, convert_to_numpy=True)
    scores = vectors @ query_vector
    best_index = int(np.argmax(scores))
    return faqs[best_index], float(scores[best_index])


st.set_page_config(page_title="電腦產品問題小幫手", page_icon="💻")
st.title("💻 電腦產品問題小幫手")
st.caption("描述你遇到的電腦或周邊設備問題，我會從客服知識庫搜尋相近案例並提供處理建議。")

try:
    MODEL = get_model()
    FAQS, FAQ_VECTORS = get_knowledge_base(str(CSV_PATH), MODEL_NAME)
except Exception as exc:
    st.error(f"知識庫或語意模型載入失敗：{exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

with st.sidebar:
    st.subheader("試著詢問")
    examples = [
        "電腦按下電源完全沒反應",
        "筆電連上 Wi-Fi 但不能上網",
        "螢幕一直閃爍怎麼辦？",
        "硬碟有喀喀聲，檔案也打不開",
    ]
    for index, example in enumerate(examples):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            st.session_state.pending_question = example
    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("請描述你遇到的問題…")
question = question or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在搜尋相近問題…"):
            faq, similarity = answer_question(question, FAQS, FAQ_VECTORS)
        if similarity < MIN_SIMILARITY:
            response = (
                "目前知識庫找不到足夠相近的問題。請補充產品型號、作業系統、錯誤訊息，"
                "以及問題發生前做過的操作；若有焦味、冒煙或異常高溫，請先停止使用並拔除電源。"
            )
            st.markdown(response)
        else:
            st.markdown(faq["answer"])
            st.markdown(f"**參考問題：** {faq['question']}")
            st.caption(
                f"分類：{faq['category']}｜產品：{faq['product']}｜"
                f"緊急程度：{faq['urgency']}｜相似度：{similarity:.0%}"
            )
            response = (
                f"{faq['answer']}\n\n**參考問題：** {faq['question']}\n\n"
                f"分類：{faq['category']}｜產品：{faq['product']}｜"
                f"緊急程度：{faq['urgency']}｜相似度：{similarity:.0%}"
            )
    st.session_state.messages.append({"role": "assistant", "content": response})
