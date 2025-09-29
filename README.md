# Multimodal RAG Server

## 1. 概要

ragserver は、テキストおよびマルチモーダル（テキスト + 画像）検索のための Retrieval-Augmented Generation (RAG) 基盤サーバです。ragserver 本体が REST API としての検索・投入処理を担い、周辺の補助サーバ（ベクトルストア、埋め込み生成、リランカー）が協調することでドキュメントを取り込み、関連度の高い情報を検索できる環境を提供します。標準構成ではローカル実行に必要なコンポーネントを含み、環境変数を書き換えるだけでクラウドサービスや外部 API に切り替えられます。

## 2. システム構成

### 2.1 デフォルト構成

| アクター       | 役割                                              | 実装ディレクトリ                                   | 備考                                                                    |
| -------------- | ------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| ragserver 本体 | REST API、MCP サーバ、検索・投入制御              | `ragserver/`                                       | `main.py` がエントリポイント                                            |
| store          | ベクトルストア（既定: Chroma）                    | `ragserver/store/`、`chroma_db/`、`chroma_server/` | `ChromaManager` がローカル永続 DB (`chroma_db/`) を使用                 |
| embed          | テキスト/画像埋め込み生成（既定: ローカル CLIP）  | `ragserver/embed/`、`embed_server/`                | `LocalEmbeddingsManager` が `embed_server/local_embed_server.py` に接続 |
| rerank         | 検索結果のリランク（既定: ローカル BGE reranker） | `ragserver/rerank/`、`rerank_server/`              | `LocalRerankManager` が `rerank_server/local_rerank_server.py` に接続   |

- ragserver 本体 (`ragserver/`): FastAPI アプリ (`main.py`) が REST API と MCP の窓口。`core/`, `store/`, `embed/`, `ingest/`, `retrieval/`, `rerank/` がモジュール群です。
- store: `ragserver/store/` で抽象化し、既定の `ChromaManager` が `chroma_db/` を永続ディレクトリとして使用。別途 `chroma_server/run.sh` でリモート Chroma にも切り替え可能。
- embed: `ragserver/embed/` に埋め込みのインタフェース。既定は `embed_server/local_embed_server.py`（ローカル CLIP）を叩く構成で、OpenAI/Cohere などへの切り替えも環境変数で対応。
- rerank: `ragserver/rerank/` にリランク機構。既定はローカル BGE (`rerank_server/local_rerank_server.py`) を使用。

### 2.2 カスタマイズ例

`.env` に設定を記述し、ベクトルストア／埋め込み／リランクを差し替えます。以下に 3 つの例を示します。

#### (A) 既定ローカル構成（Chroma + ローカル CLIP + ローカル BGE）

```env
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=chroma_db
EMBED_PROVIDER=local
LOCAL_EMBED_BASE_URL=http://localhost:8001/v1
RERANK_PROVIDER=local
LOCAL_RERANK_BASE_URL=http://localhost:8002/v1
TOPK=10
TOPK_RERANK_SCALE=5
```

`embed_server/run.sh` と `rerank_server/run.sh` を立ち上げ、`ragserver/run.sh` を実行するとローカル RAG が動作します。

#### (B) OpenAI Embeddings + Cohere Rerank（Chroma 継続利用）

```env
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=chroma_db
EMBED_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
RERANK_PROVIDER=cohere
COHERE_API_KEY=...
COHERE_RERANK_MODEL=rerank-multilingual-v3.0
TOPK=10
TOPK_RERANK_SCALE=3
```

埋め込みは OpenAI Embeddings API、リランクは Cohere Rerank API に差し替え。`embed_server`/`rerank_server` は不要。

#### (C) PgVector + Cohere Embeddings + リランク無し

```env
VECTOR_STORE=pgvector
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=ragserver
PG_USER=ragserver
PG_PASSWORD=ragserver
EMBED_PROVIDER=cohere
COHERE_API_KEY=...
COHERE_EMBED_MODEL_TEXT=embed-v4.0
COHERE_EMBED_MODEL_IMAGE=embed-v4.0
RERANK_PROVIDER=none
TOPK=20
TOPK_RERANK_SCALE=1
```

`pgvector_server/init_pgdb.sh` で DB を初期化後、Cohere API による埋め込みと PgVector を組み合わせます。リランクは無効化されます。

## 3. 起動停止手順
### 3.1 サーバ群
- `ragserver/run.sh`
  - FastAPI アプリ（`ragserver.main:app`）のみ起動。
- `run_all.sh`
  - embed → rerank → ragserver の順で起動。Chroma サーバ接続が必要であればコメントアウト部分を有効化します。
- `stop_all.sh`
  - 8000/8001/8002/8003 ポートを LISTEN しているプロセスへ SIGINT → SIGTERM を送って順次停止。

各スクリプトはバックグラウンド起動 (`&`) を使用します。必要に応じて `systemd` 等で管理下さい。
また、起動中はログレベル INFO で標準出力にログが出力されます。必要に応じて `ragserver/logger.py` を修正下さい。

### 3.2 ragclient
ragserver の API を利用するデモクライアントとして、ragclient を用意しています。
- `ragclient/run.sh`
  - デフォルトで 8004 ボートを使用し、streamlit サーバが起動します。

## 4. REST API 利用手順

以下は `http://localhost:8000/v1` に API が公開されている場合の `curl` 例です。

```bash
# /reload : store を再ロード
curl -X POST http://localhost:8000/v1/reload \
  -H "Content-Type: application/json" \
  -d '{"target": "store"}'

# /upload : ファイルをアップロード
curl -X POST http://localhost:8000/v1/upload \
  -F "files=@hoge.png" \
  -F "files=@fuga.txt" \
  -F "files=@piyo.pdf"

# /query/text : テキスト検索
curl -X POST http://localhost:8000/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ragserver?", "topk": 5}'

# /query/text_multi : テキストクエリによるマルチモーダル検索
curl -X POST http://localhost:8000/v1/query/text_multi \
  -H "Content-Type: application/json" \
  -d '{"query": "Find related images", "topk": 5}'

# /query/image : 画像クエリ検索
curl -X POST http://localhost:8000/v1/query/image \
  -H "Content-Type: application/json" \
  -d '{"path": "./image.png", "topk": 5}'

# /ingest/path : ローカルパスから取り込み
curl -X POST http://localhost:8000/v1/ingest/path \
  -H "Content-Type: application/json" \
  -d '{"path": "./documents"}'

# /ingest/path_list : パスリスト（テキストファイル）を指定
curl -X POST http://localhost:8000/v1/ingest/path_list \
  -H "Content-Type: application/json" \
  -d '{"path": "./path_list.txt"}'

# /ingest/url : 単一 URL を取り込み
curl -X POST http://localhost:8000/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# /ingest/url_list : URL リストを取り込み
curl -X POST http://localhost:8000/v1/ingest/url_list \
  -H "Content-Type: application/json" \
  -d '{"path": "./url_list.txt"}'
```

各 API は `HTTP 500` でエラーメッセージを返す場合があります。詳細はログを参照してください。

## 5. MCP サーバ利用手順

`FastApiMCP` を通じて MCP (Model Context Protocol) サーバとして動作します。`ragserver/main.py` の末尾で `FastApiMCP(app, name=PROJECT_NAME)` を初期化し、`mount_http()` で HTTP エンドポイント（/mcp）を公開しています。

MCP クライアント側に ragserver を登録すると、`query`, `ingest` など FastAPI エンドポイントと同等の操作が可能です。クライアント側設定方法は利用するツールの詳細を参照下さい。以下は LM Studio での設定ファイル（json）記述例です。

```
{
  "mcpServers": {
    "my_mcp_server": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## 6. その他詳細仕様

### 6.1 使用する環境変数

`ragserver/config.py` が読み込む環境変数と値域は以下のとおりです。

| 変数名                     | 用途                                   | 値域/例                       |
| -------------------------- | -------------------------------------- | ----------------------------- |
| `VECTOR_STORE`             | ベクトルストア選択                     | `chroma` / `pgvector`         |
| `LOAD_LIMIT`               | メタ情報ロード上限件数                 | 正の整数                      |
| `CHECK_UPDATE`             | 既存ソースの更新チェックを有効化するか | `true/false` 等の真偽文字列   |
| `CHROMA_PERSIST_DIR`       | Chroma 永続ディレクトリ                | パス文字列                    |
| `CHROMA_HOST`              | Chroma サーバホスト                    | ホスト名/URL                  |
| `CHROMA_PORT`              | Chroma サーバポート                    | 正の整数                      |
| `CHROMA_API_KEY`           | Chroma Cloud API キー                  | 文字列                        |
| `CHROMA_TENANT`            | Chroma Cloud テナント名                | 文字列                        |
| `CHROMA_DATABASE`          | Chroma Cloud データベース名            | 文字列                        |
| `PG_HOST`                  | PgVector ホスト                        | 文字列                        |
| `PG_PORT`                  | PgVector ポート                        | 正の整数                      |
| `PG_DATABASE`              | PgVector DB 名                         | 文字列                        |
| `PG_USER`                  | PgVector ユーザ名                      | 文字列                        |
| `PG_PASSWORD`              | PgVector パスワード                    | 文字列                        |
| `EMBED_PROVIDER`           | 埋め込みプロバイダ選択                 | `local` / `openai` / `cohere` |
| `OPENAI_EMBED_MODEL_TEXT`  | OpenAI テキストモデル名                | 文字列                        |
| `OPENAI_API_KEY`           | OpenAI API キー                        | 文字列                        |
| `OPENAI_BASE_URL`          | OpenAI API Base URL                    | URL                           |
| `COHERE_EMBED_MODEL_TEXT`  | Cohere テキストモデル名                | 文字列                        |
| `COHERE_EMBED_MODEL_IMAGE` | Cohere 画像モデル名                    | 文字列                        |
| `COHERE_API_KEY`           | Cohere API キー                        | 文字列                        |
| `LOCAL_EMBED_MODEL_TEXT`   | ローカル埋め込みテキストモデル         | 文字列                        |
| `LOCAL_EMBED_MODEL_IMAGE`  | ローカル埋め込み画像モデル             | 文字列                        |
| `LOCAL_EMBED_BASE_URL`     | ローカル埋め込み API の URL            | URL                           |
| `CHUNK_SIZE`               | テキスト分割チャンクサイズ             | 正の整数                      |
| `CHUNK_OVERLAP`            | テキスト分割オーバーラップ             | `0 <= value < CHUNK_SIZE`     |
| `USER_AGENT`               | HTML 取得時の User-Agent               | 文字列                        |
| `RERANK_PROVIDER`          | リランクプロバイダ                     | `local` / `cohere` / `none`   |
| `LOCAL_RERANK_MODEL`       | ローカルリランカーモデル名             | 文字列                        |
| `LOCAL_RERANK_BASE_URL`    | ローカルリランカー API の URL          | URL                           |
| `COHERE_RERANK_MODEL`      | Cohere リランカーモデル名              | 文字列                        |
| `TOPK`                     | 取得件数                               | 正の整数                      |
| `TOPK_RERANK_SCALE`        | リランキング前の取得倍率               | 正の整数                      |
| `UPLOAD_DIR`               | ファイルアップロード用ディレクトリ名   | `upload`                      |

### 6.2 収集対象ファイル

`ragserver/ingest/loader.Exts` で許可された拡張子：

- テキスト: `.txt`
- マークダウン: `.md`
- 画像: `.jpg`, `.jpeg`, `.png`, `.gif`
- PDF: `.pdf`

URL ingest（`HTMLLoader`）では、HTML に含まれるリンクから上記拡張子のアセットを取得します。

### 6.3 空間キーについて

本プロジェクトでは、空間キーという概念でベクトルストアの管理を行います。\
空間キー (`space_key`) は「埋め込み種別 × モデル × 用途」を識別するキーで、ベクトルストア内のコレクション/テーブル命名に使用します。

- 形式: `[embed provider]__[model name]__[text|image]`
  - 例: `local__openai/clip-vit-base-patch32__text`
- 埋め込みやストア構成を切り替えると異なる空間キーが生成され、既存データとは別コレクションとして扱われます。モデルを切り替えた場合は再インジェストが必要です。
- マルチモーダル構成ではテキスト用 `space_key` と画像用 `space_key_multi` の 2 種が存在し、それぞれ別コレクションに格納されます。

### 6.4 ベクトル埋め込み時の正規化ポリシーについて

本プロジェクトではコサイン類似度を採用しており、以下を考慮の上、埋め込みベクトルは既定で L2 正規化（単位長化）します。

- 距離関数の一貫性：コサイン類似度 ≒ 正規化後の内積になり、DB や検索器の違いを吸収
- モデル差し替え耐性：埋め込みモデル（OpenAI/Cohere/CLIP 等）を変えても挙動が安定
- 数値安定性：極端なベクトル長によるスコアの偏りを抑制
