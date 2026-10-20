# 電腦產品問題小幫手

使用 Streamlit 建立的客服問答網站。系統會用本機語意模型將使用者問題轉成向量，再呼叫 Supabase `match_qa_semantic` 函式搜尋 FAQ 並顯示對應答案。

## 本機啟動

```powershell
uv sync
uv run streamlit run app.py
```

在 `.env` 設定 `SUPABASE_URL` 和 `SUPABASE_ANON_KEY`。首次啟動會下載 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 模型；完成後模型會使用本機快取。

## 部署到 Streamlit Community Cloud

1. 將專案（包含 `app.py`、`requirements.txt`、`README.md` 和 `.streamlit/config.toml`）推送到 GitHub。
2. 在 Streamlit Community Cloud 建立新 app，選擇該 GitHub repository，主程式指定為 `app.py`。
3. 在 App Settings → Secrets 設定 `SUPABASE_URL` 與 `SUPABASE_ANON_KEY`。
4. 部署後首次啟動需要下載語意模型，請預留啟動時間。

`.env` 已排除在 Git 版本控制之外，不要將 API Key 提交到 repository。請使用 Supabase anon/publishable key，並確保資料表的 RLS 政策只允許讀取 FAQ；不要把 `service_role` key 放進前端或部署 secrets。
