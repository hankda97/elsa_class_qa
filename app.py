"""Streamlit FAQ assistant using Supabase vector search."""

import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.30"))


def get_setting(name: str) -> str | None:
    """Read local .env values or Streamlit Community Cloud secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value) if value else None


@st.cache_resource(show_spinner="正在載入語意模型（首次啟動可能需要一些時間）…")
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner=False)
def get_supabase_client(url: str, anon_key: str) -> Client:
    return create_client(url, anon_key)


def search_faq(question: str, model: SentenceTransformer, client: Client) -> tuple[dict | None, float]:
    query_vector = model.encode(question, normalize_embeddings=True, convert_to_numpy=True)
    if query_vector.shape[0] != 384:
        raise ValueError(
            f"目前模型輸出 {query_vector.shape[0]} 維向量；qa_semantic 表需要 384 維。"
        )

    result = client.rpc(
        "match_qa_semantic",
        {
            "query_embedding": query_vector.astype(np.float32).tolist(),
            "match_count": 5,
            "filter_category": None,
        },
    ).execute()
    rows = result.data or []
    if not rows:
        return None, 0.0
    best = rows[0]
    return best, float(best.get("similarity", 0.0))


st.set_page_config(page_title="電腦產品問題小幫手", page_icon="💻")
st.title("💻 電腦產品問題小幫手")
st.caption("描述你遇到的電腦或周邊設備問題，我會從 Supabase 客服知識庫搜尋相近案例並提供處理建議。")

supabase_url = get_setting("SUPABASE_URL")
supabase_key = get_setting("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    st.error(
        "找不到 Supabase 連線設定。請在本機 .env 設定 SUPABASE_URL 和 SUPABASE_ANON_KEY；"
        "部署到 Streamlit Community Cloud 時，請在 App Settings → Secrets 設定相同名稱。"
    )
    st.stop()

try:
    MODEL = get_model(MODEL_NAME)
    SUPABASE = get_supabase_client(supabase_url, supabase_key)
except Exception as exc:
    st.error(f"模型或 Supabase 初始化失敗：{exc}")
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
        with st.spinner("正在搜尋 Supabase 知識庫…"):
            try:
                faq, similarity = search_faq(question, MODEL, SUPABASE)
            except Exception as exc:
                # Keep server-side details out of the public chat interface.
                print(f"Supabase FAQ search failed: {type(exc).__name__}: {exc}")
                st.error(
                    "Supabase 搜尋失敗。請確認 qa_semantic 資料表已有資料、"
                    "match_qa_semantic RPC 函式已建立，且 anon 權限可讀取 FAQ。"
                )
                st.stop()

        if faq is None or similarity < MIN_SIMILARITY:
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
