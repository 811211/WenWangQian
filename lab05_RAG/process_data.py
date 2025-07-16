"""
勞動基準法PDF處理與RAG資料準備程式

此程式實現以下功能：
1. 讀取PDF內容
2. 智能分割文本內容
3. 生成embedding向量（多執行緒並行處理）
4. 儲存至PostgreSQL資料庫

主要依賴套件：
- PyPDF2: PDF讀取
- langchain: 文本分割
- psycopg2: PostgreSQL連接
- numpy: 數值計算
- threading: 多執行緒處理
"""

import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from utils.database_config import get_database_config
from utils.ai_client import get_embedding_for_content

# PDF處理相關
try:
    import PyPDF2
except ImportError:
    print("請安裝PyPDF2: pip install PyPDF2")

# 文本分割相關
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("請安裝langchain: pip install langchain")

# 載入環境變數
load_dotenv()

class LaborLawProcessor:
    """日本東京淺草觀音寺一百籤"""
    
    def __init__(self):
        """初始化處理器"""
        # 文本分割器配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,         # 每段文字約400字符
            chunk_overlap=200,       # 段落間重疊200字符，確保上下文連貫
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]  # 按優先順序分割
        )
        
        # PostgreSQL連接配置
        self.db_config = get_database_config()
        
        # 多執行緒配置
        self.max_workers = 4  # 使用4個執行緒
        self.lock = threading.Lock()  # 執行緒鎖用於安全輸出
    
    def read_pdf(self, pdf_path: str) -> str:
        """
        讀取PDF文件內容
        
        Args:
            pdf_path (str): PDF檔案路徑
            
        Returns:
            str: PDF文本內容
        """
        print(f"正在讀取PDF檔案: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 提取所有頁面的文字
                text_content = ""
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text_content += page_text + "\n"
                    
                    # 顯示進度
                    print(f"已讀取 {page_num + 1}/{total_pages} 頁")
                
                print(f"PDF讀取完成，共 {total_pages} 頁，總字數: {len(text_content)}")
                return text_content
                
        except Exception as e:
            print(f"讀取PDF時發生錯誤: {e}")
            return ""
    
    def query_aoai_embedding(self, content: str) -> list[float]:
        """從 Azure OpenAI 服務獲取文本的 embedding 向量

        Args:
            content (str): 要進行 embedding 的文本內容

        Returns:
            list[float]: 返回 embedding 向量，如果發生錯誤則返回空列表
        """
        return get_embedding_for_content(content)


    def preprocess_text(self, text: str) -> str:
        """
        預處理文本內容
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 預處理後的文本
        """
        print("正在進行文本預處理...")
        
        # 移除多餘的空白和換行
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # 移除頁碼和頁眉頁腳
        text = re.sub(r'第\s*\d+\s*頁', '', text)
        text = re.sub(r'- \d+ -', '', text)
        
        # 正規化標點符號
        text = text.replace('　', ' ')  # 將全形空格替換為半形空格
        
        print("文本預處理完成")
        return text.strip()
    
    def split_text_intelligently(self, text: str) -> List[Dict]:
        """
        智能分割文本，保留法條結構
        
        Args:
            text (str): 預處理後的文本
            
        Returns:
            List[Dict]: 分割後的文本段落列表，包含元數據
        """
        print("正在進行智能文本分割...")
        
        # 使用RecursiveCharacterTextSplitter進行基本分割
        chunks = self.text_splitter.split_text(text)
        
        # 為每個chunk添加元數據
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 檢測是否為法條
            article_match = re.search(r'第\s*(\d+)\s*條', chunk)
            chapter_match = re.search(r'第\s*([一二三四五六七八九十]+)\s*章', chunk)
            
            # 生成內容context資訊
            context = f"段落{i+1}"
            if article_match:
                context += f" | 第{article_match.group(1)}條"
            if chapter_match:
                context += f" | 第{chapter_match.group(1)}章"
            
            chunk_data = {
                'content': chunk.strip(),
                'context': context,
                'chunk_index': i,
                'article_number': article_match.group(1) if article_match else None,
                'chapter_info': chapter_match.group(1) if chapter_match else None,
                'char_count': len(chunk)
            }
            
            processed_chunks.append(chunk_data)
        
        print(f"文本分割完成，共生成 {len(processed_chunks)} 個段落")
        return processed_chunks
    
    def process_embedding_batch(self, chunks_batch: List[Dict], batch_id: int, total_batches: int) -> List[Dict]:
        """
        處理一批embedding（單一執行緒任務）
        
        Args:
            chunks_batch (List[Dict]): 文本段落批次
            batch_id (int): 批次編號
            total_batches (int): 總批次數
            
        Returns:
            List[Dict]: 處理完成的段落列表
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks_batch):
            try:
                # 生成embedding
                embedding = self.query_aoai_embedding(chunk['content'])
                chunk['embedding'] = embedding
                
                # 執行緒安全的進度輸出
                with self.lock:
                    print(f"🧵 執行緒 {batch_id}/{total_batches} | 段落 {i+1}/{len(chunks_batch)}")
                
                processed_chunks.append(chunk)
                
                # 避免API請求過於頻繁
                time.sleep(0.1)
                
            except Exception as e:
                with self.lock:
                    print(f"❌ 執行緒 {batch_id} 處理段落 {i+1} 時發生錯誤: {e}")
                chunk['embedding'] = None
                processed_chunks.append(chunk)
        
        with self.lock:
            print(f"✅ 執行緒 {batch_id} 完成，處理了 {len(chunks_batch)} 個段落")
        
        return processed_chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        使用多執行緒為文本段落生成embedding向量
        
        Args:
            chunks (List[Dict]): 文本段落列表
            
        Returns:
            List[Dict]: 包含embedding的段落列表
        """
        print(f"🚀 開始使用 {self.max_workers} 個執行緒生成embedding向量...")
        print(f"📊 總共需要處理 {len(chunks)} 個段落")
        
        # 將chunks分割成4個批次
        batch_size = len(chunks) // self.max_workers
        if len(chunks) % self.max_workers != 0:
            batch_size += 1
        
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        
        print(f"📦 分割成 {len(batches)} 個批次，每批次約 {batch_size} 個段落")
        
        # 記錄開始時間
        start_time = time.time()
        
        # 使用ThreadPoolExecutor進行多執行緒處理
        all_processed_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任務
            future_to_batch = {
                executor.submit(self.process_embedding_batch, batch, i+1, len(batches)): i 
                for i, batch in enumerate(batches)
            }
            
            # 收集結果
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_processed_chunks.extend(batch_results)
                except Exception as e:
                    print(f"❌ 批次 {batch_id + 1} 處理失敗: {e}")
        
        # 按原始順序排序
        all_processed_chunks.sort(key=lambda x: x['chunk_index'])
        
        # 計算處理時間
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 統計成功/失敗數量
        successful_embeddings = sum(1 for chunk in all_processed_chunks if chunk.get('embedding') is not None)
        failed_embeddings = len(all_processed_chunks) - successful_embeddings
        
        print("=" * 60)
        print(f"🎉 多執行緒Embedding生成完成！")
        print(f"⏱️  總處理時間: {processing_time:.2f} 秒")
        print(f"✅ 成功生成: {successful_embeddings} 個embedding")
        print(f"❌ 失敗: {failed_embeddings} 個embedding")
        print(f"🚀 平均速度: {successful_embeddings/processing_time:.2f} 個embedding/秒")
        print("=" * 60)
        
        return all_processed_chunks
    
    def create_database_tables(self):
        """創建PostgreSQL資料表"""
        print("正在創建資料庫表格...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            print("✅ 使用embeddings表格結構")
            
            embeddings_table = """
            CREATE TABLE IF NOT EXISTS public.embeddings
            (
                id SERIAL NOT NULL,
                embedding_vector double precision[],
                created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
                content text COLLATE pg_catalog."default",
                context text NOT NULL,
                CONSTRAINT embeddings_pkey PRIMARY KEY (id)
            );
            """
            
            # 創建基本索引以提高查詢效能
            indexes = """
            CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings USING gin(to_tsvector('chinese', content));
            CREATE INDEX IF NOT EXISTS idx_embeddings_context ON embeddings(context);
            """
            
            # 創建表格和索引
            cur.execute(embeddings_table)
            cur.execute(indexes)
            
            conn.commit()
            cur.close()
            conn.close()
            
            print("✅ 資料庫表格創建完成")
            print("🔍 支援全文檢索和向量相似度搜索功能")
            
        except Exception as e:
            print(f"❌ 創建資料庫表格時發生錯誤: {e}")
            print("\n🔧 可能的解決方案：")
            print("1. 確認PostgreSQL服務正在運行")
            print("2. 檢查資料庫連接設定")
            print("3. 確認資料庫權限")
    
    def save_to_database(self, chunks: List[Dict]):
        """
        將處理後的資料儲存到PostgreSQL
        
        Args:
            chunks (List[Dict]): 包含embedding的文本段落列表
        """
        print("正在儲存資料到PostgreSQL...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # 準備批量插入資料
            embedding_data = []
            
            for chunk in chunks:
                if chunk.get('embedding'):
                    embedding_data.append((
                        chunk['embedding'],          # embedding_vector
                        chunk['content'],           # content
                        chunk['context']            # context
                    ))
            
            # 批量插入embedding資料
            execute_values(cur, """
                INSERT INTO \"2500567RAG\" (embedding_vector, content, context)
                VALUES %s
            """, embedding_data)
            
            conn.commit()
            cur.close()
            conn.close()

            print(f"✅ 資料儲存完成！共儲存 {len(embedding_data)} 筆記錄到\"2500567RAG\"表格")

        except Exception as e:
            print(f"❌ 儲存資料時發生錯誤: {e}")
    
    def check_existing_data(self) -> int:
        """
        檢查資料庫中現有的資料數量
        
        Returns:
            int: 現有記錄數量
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("SELECT COUNT(*) FROM \"2500567RAG\"")
            count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return count
            
        except Exception as e:
            print(f"檢查現有資料時發生錯誤: {e}")
            return 0
    
    def process_pdf(self, pdf_path: str):
        """
        完整的PDF處理流程
        
        Args:
            pdf_path (str): PDF檔案路徑
        """
        print("=" * 50)
        print("開始處理勞動基準法PDF")
        print("=" * 50)
        
        # 步驟1：創建資料庫表格
        self.create_database_tables()
        
        # 步驟2：檢查現有資料
        existing_count = self.check_existing_data()
        if existing_count > 0:
            print(f"⚠️  資料庫中已有 {existing_count} 筆記錄")
            response = input("是否要清除現有資料並重新處理？(y/N): ")
            if response.lower() == 'y':
                try:
                    conn = psycopg2.connect(**self.db_config)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM \"2500567RAG\"")
                    conn.commit()
                    cur.close()
                    conn.close()
                    print("✅ 已清除現有資料")
                except Exception as e:
                    print(f"❌ 清除資料時發生錯誤: {e}")
                    return
            else:
                print("取消處理")
                return
        
        # 步驟3：讀取PDF內容
        raw_text = self.read_pdf(pdf_path)
        if not raw_text:
            print("PDF讀取失敗，程式終止")
            return
        
        # 步驟4：預處理文本
        cleaned_text = self.preprocess_text(raw_text)
        
        # 步驟5：智能分割文本
        chunks = self.split_text_intelligently(cleaned_text)
        
        # 步驟6：生成embedding向量
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # 步驟7：儲存到資料庫
        self.save_to_database(chunks_with_embeddings)
        
        print("=" * 50)
        print("PDF處理完成！")
        print("=" * 50)
        
        # 顯示最終統計
        final_count = self.check_existing_data()
        print(f"📊 資料庫中共有 {final_count} 筆embedding記錄")

def main():
    """主程式入口"""
    # 初始化處理器
    processor = LaborLawProcessor()
    
    # PDF檔案路徑
    pdf_path = "日本東京淺草觀音寺一百籤.pdf"
    
    # 檢查檔案是否存在
    if not os.path.exists(pdf_path):
        print(f"找不到PDF檔案: {pdf_path}")
        return
    
    # 開始處理
    processor.process_pdf(pdf_path)

if __name__ == "__main__":
    main()