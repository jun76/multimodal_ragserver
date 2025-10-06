# LlamaIndex 移行ガイド (v2)

このドキュメントは、`multimodal_ragserver` の LangChain から LlamaIndex への完全移行について説明します。

## 概要

`llama_v2` ブランチでは、プロジェクトを LangChain から LlamaIndex に完全移行しました。既存の抽象化レイヤーを削除し、LlamaIndex の設計思想に沿った新しいアーキテクチャを採用しています。

## 主な変更点

### 1. アーキテクチャの刷新

**移行前（LangChain）**:
```
複数の抽象化レイヤー
├── EmbeddingsManager
│   ├── OpenAIEmbeddingsManager
│   ├── CohereEmbeddingsManager
│   └── HFCLIPEmbeddingsManager
├── VectorStoreManager
│   ├── ChromaManager
│   └── PgVectorManager
├── RerankManager
│   ├── CohereRerankManager
│   └── HFRerankManager
├── FileLoader / HTMLLoader
├── Ingest
└── Retriever
```

**移行後（LlamaIndex）**:
```
シンプルな構造
├── MultiModalVectorStoreIndex (コア)
├── Settings (グローバル設定)
│   ├── embed_model
│   └── text_splitter
├── カスタムコンポーネント
│   ├── HFCLIPEmbedding
│   └── HFRerank
└── 標準コンポーネント
    ├── OpenAIEmbedding
    ├── CohereEmbedding
    ├── CohereRerank
    ├── ChromaVectorStore
    ├── PGVectorStore
    └── SimpleDirectoryReader
```

### 2. 削除されたディレクトリ

以下のディレクトリとその中のすべてのファイルが削除されました：

- `ragserver/embed/` - 埋め込み管理の抽象化レイヤー
- `ragserver/ingest/` - インジェスト処理
- `ragserver/retrieval/` - 検索処理
- `ragserver/rerank/` - リランク処理
- `ragserver/store/` - ベクトルストア管理

これらの機能は、LlamaIndex の標準コンポーネントとカスタムコンポーネントで実現されます。

### 3. 新しいファイル構成

```
ragserver/
├── __init__.py
├── config.py                      # 環境変数読み込み（変更なし）
├── core/                          # コアユーティリティ（変更なし）
├── logger.py                      # ロガー（変更なし）
├── hfclip_embedding.py           # カスタムHFCLIP埋め込み
├── hf_rerank.py                  # カスタムHFリランク
├── main.py                       # LlamaIndexベースの実装
├── main_langchain_backup.py     # LangChain実装のバックアップ
└── run.sh
```

### 4. コードの大幅な簡潔化

#### 削減されたコード量

- **削除**: 約4,000行（LangChain関連の抽象化レイヤー）
- **追加**: 約1,500行（LlamaIndexベースの実装）
- **正味削減**: 約2,500行（**60%削減**）

#### インジェスト処理の簡潔化

**移行前**: 約200行（複数ファイル）
```python
# file_loader.py, html_loader.py, ingest.py の合計
```

**移行後**: 約30行
```python
reader = SimpleDirectoryReader(input_dir=path, recursive=True)
documents = reader.load_data()
for doc in documents:
    _index.insert(doc)
```

#### 検索処理の簡潔化

**移行前**: 約150行（retriever.py）
```python
# 複雑な space_key 管理とマルチモーダル処理
```

**移行後**: 約10行
```python
retriever = _index.as_retriever(similarity_top_k=topk)
nodes = retriever.retrieve(query)
if _rerank:
    nodes = _rerank.postprocess_nodes(nodes, query_bundle)
```

## 機能の対応表

| 機能 | LangChain 実装 | LlamaIndex 実装 | 状態 |
|------|---------------|----------------|------|
| **埋め込み: OpenAI** | `OpenAIEmbeddingsManager` | `OpenAIEmbedding` | ✅ 完了 |
| **埋め込み: Cohere** | `CohereEmbeddingsManager` | `CohereEmbedding` | ✅ 完了 |
| **埋め込み: HFCLIP** | `HFCLIPEmbeddingsManager` | `HFCLIPEmbedding` (カスタム) | ✅ 完了 |
| **ベクトルストア: Chroma** | `ChromaManager` | `ChromaVectorStore` | ✅ 完了 |
| **ベクトルストア: PgVector** | `PgVectorManager` | `PGVectorStore` | ✅ 完了 |
| **リランク: Cohere** | `CohereRerankManager` | `CohereRerank` | ✅ 完了 |
| **リランク: HF** | `HFRerankManager` | `HFRerank` (カスタム) | ✅ 完了 |
| **テキスト検索** | `/v1/query/text` | `/v1/query/text` | ✅ 完了 |
| **マルチモーダル検索** | `/v1/query/text_multi` | `/v1/query/text_multi` | ✅ 完了 |
| **画像検索** | `/v1/query/image` | `/v1/query/image` | ✅ 完了 |
| **ローカルファイル取り込み** | `/v1/ingest/path` | `/v1/ingest/path` | ✅ 完了 |
| **ファイルリスト取り込み** | `/v1/ingest/path_list` | `/v1/ingest/path_list` | ✅ 完了 |
| **URL取り込み** | `/v1/ingest/url` | `/v1/ingest/url` | ✅ 完了 |
| **URLリスト取り込み** | `/v1/ingest/url_list` | `/v1/ingest/url_list` | ✅ 完了 |

**すべての機能が実装されています。**

## FastAPI エンドポイント仕様

### 維持されている仕様

すべてのエンドポイントは既存の仕様を維持しています：

- **`GET /v1/health`**: ヘルスチェック
- **`POST /v1/reload`**: プロバイダのリロード
- **`POST /v1/upload`**: ファイルアップロード
- **`POST /v1/query/text`**: テキスト検索
- **`POST /v1/query/text_multi`**: マルチモーダル検索（テキストで画像を検索）
- **`POST /v1/query/image`**: 画像検索（画像で画像を検索）
- **`POST /v1/ingest/path`**: ローカルファイル取り込み
- **`POST /v1/ingest/path_list`**: ファイルリスト取り込み
- **`POST /v1/ingest/url`**: URL取り込み
- **`POST /v1/ingest/url_list`**: URLリスト取り込み

### リクエスト/レスポンス形式

既存のクライアントとの互換性を維持しています。

**例: テキスト検索**
```json
// Request
{
  "query": "検索クエリ",
  "topk": 5
}

// Response
{
  "documents": [
    {
      "page_content": "...",
      "metadata": {...},
      "score": 0.95
    }
  ]
}
```

## サポートするプロバイダ

### 埋め込みプロバイダ

1. **OpenAI** (`OPENAI_EMBED_NAME`)
   - 環境変数: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBED_MODEL_TEXT`
   - テキストのみ対応

2. **Cohere** (`COHERE_EMBED_NAME`)
   - 環境変数: `COHERE_API_KEY`, `COHERE_EMBED_MODEL_TEXT`
   - Cohere v4 はマルチモーダル対応

3. **HFCLIP** (`HFCLIP_EMBED_NAME`)
   - 環境変数: `HFCLIP_EMBED_BASE_URL`, `HFCLIP_EMBED_MODEL_TEXT`, `HFCLIP_EMBED_MODEL_IMAGE`
   - マルチモーダル対応（カスタム実装）

### ベクトルストアプロバイダ

1. **Chroma** (`CHROMA_STORE_NAME`)
   - 環境変数: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIR`
   - ローカルモードとリモートモードをサポート

2. **PgVector** (`PGVECTOR_STORE_NAME`)
   - 環境変数: `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`

### リランクプロバイダ

1. **Cohere** (`COHERE_RERANK_NAME`)
   - 環境変数: `COHERE_API_KEY`, `COHERE_RERANK_MODEL`

2. **HF** (`HF_RERANK_NAME`)
   - 環境変数: `HF_RERANK_BASE_URL`, `HF_RERANK_MODEL`
   - カスタム実装

## 動作確認

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**URL取り込み機能を使用する場合**:
```bash
pip install llama-index-readers-web
```

### 2. 環境変数の設定

`.env` ファイルは既存のものをそのまま使用できます。設定項目に変更はありません。

### 3. サーバーの起動

```bash
cd ragserver
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. ヘルスチェック

```bash
curl http://localhost:8000/v1/health
```

レスポンス例:
```json
{
  "status": "ok",
  "store": "chroma",
  "embed": "hfclip",
  "rerank": "hf",
  "framework": "llamaindex"
}
```

### 5. ファイルの取り込み

```bash
curl -X POST http://localhost:8000/v1/ingest/path \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documents"}'
```

### 6. 検索

```bash
# テキスト検索
curl -X POST http://localhost:8000/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "検索クエリ", "topk": 5}'

# マルチモーダル検索（テキストで画像を検索）
curl -X POST http://localhost:8000/v1/query/text_multi \
  -H "Content-Type: application/json" \
  -d '{"query": "検索クエリ", "topk": 5}'

# 画像検索（画像で画像を検索）
curl -X POST http://localhost:8000/v1/query/image \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/image.jpg", "topk": 5}'
```

## 主な改善点

### 1. コードの簡潔化

- 約60%のコード削減
- 抽象化レイヤーの削除により、コードの見通しが向上
- LlamaIndexの標準APIを使用することで、保守性が向上

### 2. マルチモーダル対応の改善

- `MultiModalVectorStoreIndex` による統一的な処理
- `space_key` の手動管理が不要に
- テキストと画像の自動分離

### 3. パイプラインの統一

- `ingest → embed → retrieval → rerank` が統一されたAPIで実現
- 設定の一元管理（`Settings`）

### 4. プロバイダサポートの維持

- すべての既存プロバイダをサポート
- 実行時の切り替えが可能

## コードスタイル

移行後も、元のコードスタイルを維持しています:

- **docstring**: すべての関数・クラスに詳細な docstring を記述
- **logger.debug("trace")**: 各関数の先頭でトレースログを出力
- **コメント行**: 重要な処理にはコメントを追加

## トラブルシューティング

### インポートエラー

```
ModuleNotFoundError: No module named 'llama_index'
```

**解決方法**: 依存パッケージをインストールしてください。
```bash
pip install -r requirements.txt
```

### URL取り込みエラー

```
HTTPException: 501 URL ingestion requires llama-index-readers-web
```

**解決方法**: Web ローダーをインストールしてください。
```bash
pip install llama-index-readers-web
```

### ベクトルストアの初期化エラー

```
RuntimeError: Failed to initialize index
```

**解決方法**: 
1. `.env` ファイルの設定を確認してください。
2. Chroma または PgVector が正しく起動しているか確認してください。
3. ログを確認して詳細なエラーメッセージを確認してください。

### HFCLIP サーバーとの接続エラー

```
RuntimeError: Failed to get text embedding from http://localhost:8001/v1/embeddings
```

**解決方法**: 
1. HFCLIP Embedding Server (`embed_server`) が起動しているか確認してください。
2. `.env` の `HFCLIP_EMBED_BASE_URL` が正しいか確認してください。

## 今後の改善点

1. **パフォーマンス最適化**: 大量ドキュメントの取り込み時の並列処理
2. **エラーハンドリングの強化**: より詳細なエラーメッセージとリトライ機構
3. **テストの追加**: ユニットテストと統合テストの整備
4. **ドキュメントの充実**: API仕様書の作成

## 参考資料

- [LlamaIndex Documentation](https://developers.llamaindex.ai/)
- [LlamaIndex Multi-modal Guide](https://developers.llamaindex.ai/python/framework/use_cases/multimodal/)
- [LlamaHub](https://llamahub.ai/)

## 質問・フィードバック

移行に関する質問やフィードバックは、GitHub Issues でお願いします。
