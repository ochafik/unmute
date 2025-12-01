#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

cd frontend
pnpm install
pnpm dev
