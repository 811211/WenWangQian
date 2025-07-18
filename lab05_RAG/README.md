文王籤智慧解籤系統 - 完整指南
📁 專案概述
本專案實現了完整的 RAG (Retrieval-Augmented Generation) 系統，專門用於查詢與解讀「文王籤」，提供多種使用方式：

🔄 系統架構
資料處理階段：文王籤原始資料處理與向量化

命令列查詢：基礎 AI Agent 查詢工具

Streamlit Web UI：單頁式智慧解籤查詢介面

前後端分離：FastAPI + React 完整現代化架構

🎯 核心功能
文王籤資料讀取與預處理

智慧文本分割與結構化

多執行緒 Embedding 向量生成

PostgreSQL 向量資料庫儲存

AI Agent 智慧解籤

繁體中文 Reranker 提升語意準確性

語義向量檢索與即時互動

🗂️ 檔案結構
bash
複製
編輯
lab05_RAG/
├── 文王籤.txt                        # 原始文王籤資料
├── utils/
│   ├── database_config.py
│   ├── ai_client.py
│   └── tracking_utils.py
├── process_data.py                   # 文王籤資料處理
├── query_test.py                     # 命令列智慧查詢
├── streamlit_app.py                  # Streamlit Web UI
├── api_server.py                     # FastAPI API Server
├── frontend/                         # React 前端程式碼
└── README.md
🚀 快速開始
1️⃣ 安裝依賴
bash
複製
編輯
pip install -r requirements-api.txt
2️⃣ 設定環境變數
建立 .env 並填入：

env
複製
編輯
AOAI_KEY=your_azure_openai_api_key
AOAI_URL=https://your-azure-endpoint/
EMBEDDING_API_KEY=your_azure_openai_api_key
EMBEDDING_URL=https://your-azure-endpoint/
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=wenwang_qian_rag
PG_USER=postgres
PG_PASSWORD=your_password
🔧 功能特性
🔎 語義檢索文王籤

📝 自動查詢改寫優化（針對籤號問題強化）

🎯 繁體中文專用 Reranker 排序

🖥️ Streamlit 美觀 UI 與 React 前端選擇

📊 Token 使用統計 / 技術細節展示

🔄 多輪互動查詢（完整上下文保持）

🌟 使用方式
📥 資料處理：
bash
複製
編輯
python process_data.py
💬 命令列智慧查詢：
bash
複製
編輯
python query_test.py
🎨 Streamlit UI：
bash
複製
編輯
streamlit run streamlit_app.py
🔗 API Server：
bash
複製
編輯
python api_server.py
📖 使用範例
常見問題查詢

複製
編輯
請給我第10籤解籤
結果示例

複製
編輯
🤖 AI Agent 回答：
第10籤（中吉）
卦意：守正安分，忍耐等待時機，終有好轉。
宜謹慎，勿躁進。
🧠 技術特色
BAAI/bge-reranker-base 繁體中文 Reranker

支援語義檢索與資料庫結合

多執行緒 embedding 優化效能

即時互動，支援 WebSocket 與 RESTful API
