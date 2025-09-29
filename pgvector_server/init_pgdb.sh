#!/usr/bin/env bash

# set -e: 途中で失敗したら中断 / -u: 未定義変数をエラー / -o pipefail: パイプの途中失敗を検知
set -euo pipefail

# =========================
# 設定（必要に応じて変更してください）
# =========================
PG_SUPERUSER="postgres"        # 管理者（sudo -u で切替可能なOSユーザ）
DB_NAME="ragserver"            # アプリ用データベース名
APP_USER="ragserver"           # アプリ用DBユーザ
APP_PASSWORD="ragserver"       # アプリ用DBユーザのパスワード

# =========================
# ヘルパ関数
# =========================
run_psql() {
  # 管理者でDB非指定のコマンドを実行
  local sql="$1"
  sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 -c "$sql"
}

run_psql_db() {
  # 管理者で対象DBに対してSQL実行
  local db="$1"
  local sql="$2"
  sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 -d "$db" -c "$sql"
}

usage() {
  cat <<EOF
Usage:
  ./init_pgdb.sh          初期インストール直後のPostgreSQLに対し、ragserver利用に必要な初期化を実施
  ./init_pgdb.sh --reset  既存のDBを空（再作成）にして再初期化

現在の設定:
  PG_SUPERUSER=${PG_SUPERUSER}
  DB_NAME=${DB_NAME}
  APP_USER=${APP_USER}
EOF
}

# =========================
# メイン処理
# =========================
ARG="${1:-}"

if [[ "$ARG" == "-h" || "$ARG" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$ARG" == "--reset" ]]; then
  echo "[reset] 既存DB(${DB_NAME})をドロップして再作成します..."
  # 接続中セッションを切断してからDROP
  run_psql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();" || true
  run_psql "DROP DATABASE IF EXISTS \"${DB_NAME}\";"
  run_psql "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}'; END IF; END \$\$;"
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"
  run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"
  run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"
  echo "[reset] 完了"
  exit 0
fi

echo "[init] ragserver用ユーザ/DB/権限を初期化します..."
# ユーザ作成（存在しなければ）
run_psql "DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN
    CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}';
  END IF;
END \$\$;"

# DB作成（存在しなければ）
if ! sudo -u "$PG_SUPERUSER" psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"
fi

# vector拡張を作成（存在しなければ）
run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"

# スキーマ権限
run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"

echo "[init] 完了"
