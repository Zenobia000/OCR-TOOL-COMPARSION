# olmOCR PDF 處理工具

## 簡介

olmOCR (Open Language Model OCR) 是 AllenAI 開發的 OCR 工具，特別適合處理掃描版 PDF 文檔。它使用大型語言模型進行 OCR 識別，準確率可達 90%。olmOCR 是一個用於將 PDF 線性化以用於 LLM 數據集/訓練的工具包。

參考：[olmOCR GitHub](https://github.com/allenai/olmocr)

## 特點

- ✅ **掃描文檔準確率高**：對掃描版 PDF 的準確率可達 90%
- ✅ **模型預下載支持**：支持預先下載模型，避免處理時超時
- ✅ **GPU 記憶體優化**：可配置 GPU 記憶體使用率
- ✅ **批量處理支持**：支持處理大量 PDF 文件
- ⚠️ **配置較複雜**：需要安裝多個依賴和模型
- ⚠️ **處理時間較長**：平均處理時間約 15-23 秒/文件

## 安裝要求

### 系統要求

- **Python 版本**：Python 3.10.12（NVIDIA 官方推薦）或 Python 3.11+（兼容）
- **作業系統**：Ubuntu/Debian（或其他 Linux 發行版）
- **GPU（可選）**：如需使用 GPU 加速，需安裝 NVIDIA 驅動與 CUDA
  - **NVIDIA 官方推薦配置**：
    - CUDA 版本：12.6.2.004
    - CUDA Driver 版本：560.35.03 或更新
    - vLLM 版本：0.5.5
  - **其他兼容配置**：CUDA 12.4 或 12.8（向後兼容）
- **記憶體**：至少 1.5GB 可用記憶體
- **網絡連接**：首次運行需要下載模型

### 步驟 1: 安裝 `uv`

```bash
pip install uv
```

驗證安裝：
```bash
uv --version
```

### 步驟 2: 使用公用 OCR 環境

**本項目使用統一的 OCR 虛擬環境** (`.venv`)，位於 `03-advanced-tools/` 目錄。

#### 啟動環境

```bash
# 進入項目目錄
cd 03-advanced-tools

# 啟動公用環境
source .venv/bin/activate
```

驗證虛擬環境：
```bash
which python  # 應指向 .venv/bin/python
python --version  # 應為 Python 3.10.12（NVIDIA 推薦）或 3.11.x 或更新版本
```

### 步驟 3: 安裝系統套件（Ubuntu/Debian）

```bash
sudo apt-get update
sudo apt-get install -y \
    poppler-utils \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools
```

驗證系統套件：
```bash
pdftoppm -v  # 檢查 poppler-utils
```

### 步驟 4: 安裝 olmOCR 和依賴

#### 推薦安裝方式（GPU 模式）

**首先確認您的 CUDA 版本**：
```bash
nvcc --version  # 查看 CUDA 版本
nvidia-smi  # 查看 CUDA Driver 版本
```

**NVIDIA 官方推薦配置**（最佳兼容性）：
- CUDA 版本：12.6.2.004
- CUDA Driver 版本：560.35.03 或更新
- vLLM 版本：0.5.5
- Python 版本：3.10.12

**根據 CUDA 版本選擇安裝命令**：

**CUDA 12.6（NVIDIA 官方推薦）**：
```bash
# 確保公用環境已啟動
source .venv/bin/activate

# 1. 安裝 olmOCR（GPU 模式）
# ⚠️ 注意：olmocr[gpu] 會自動安裝所有必需的依賴，包括 vLLM 推理引擎
# 安裝時間可能較長（10-20分鐘），請耐心等待
# 對於 CUDA 12.6，可以使用 cu128 版本（向後兼容）
uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# 2. （可選）驗證 vLLM 版本（應為 0.5.5 或兼容版本）
python -c "import vllm; print(f'vLLM 版本: {vllm.__version__}')"

# 3. （可選）安裝 FlashInfer 以加速推理
# ⚠️ 警告：安裝 FlashInfer 可能會降級 PyTorch 版本，但 CUDA 支持通常會保留
# 如果不想降級 PyTorch，可以跳過此步驟
# uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

**CUDA 12.8**：
```bash
# 確保公用環境已啟動
source .venv/bin/activate

# 1. 安裝 olmOCR（GPU 模式）
# ⚠️ 注意：olmocr[gpu] 會自動安裝所有必需的依賴，包括 vLLM 推理引擎
# 安裝時間可能較長（10-20分鐘），請耐心等待
uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# 2. （可選）安裝 FlashInfer 以加速推理
# ⚠️ 警告：安裝 FlashInfer 可能會降級 PyTorch 版本，但 CUDA 支持通常會保留
# 如果不想降級 PyTorch，可以跳過此步驟
uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl

# 3. （可選）如果 FlashInfer 降級了 PyTorch，可以重新安裝所需版本
# python -c "import torch; print(torch.__version__)"  # 檢查當前版本
# ⚠️ 重要：必須明確指定版本號，否則可能安裝 cu126 版本
# uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.4**：
```bash
# 確保公用環境已啟動
source .venv/bin/activate

# 1. 安裝 olmOCR（GPU 模式）
# ⚠️ 注意：PyTorch cu128 版本通常可以在 CUDA 12.4 系統上運行（向後兼容）
# 如果遇到兼容性問題，可以嘗試使用 cu124 版本
uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

# 如果上述命令失敗或出現 CUDA 版本錯誤，嘗試使用 cu124：
# uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu124

# 2. （可選）安裝 FlashInfer 以加速推理
# ⚠️ 警告：安裝 FlashInfer 可能會降級 PyTorch 版本，但 CUDA 支持通常會保留
# ⚠️ 注意：FlashInfer 可能沒有 cu124 的官方 wheel
# CUDA 12.4 用戶可以安全跳過此步驟，olmOCR 仍可正常運行
# 如果安裝後 PyTorch 被降級，可以重新安裝所需版本：
# ⚠️ 重要：必須明確指定版本號，否則可能安裝 cu126 版本
# uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
```

**重要說明**：
- `olmocr[gpu]` 已經包含了所有必需的依賴（vLLM、HuggingFace Hub 等）
- **不需要**單獨安裝 sglang、fastapi、uvicorn 等
- FlashInfer 是可選的，但推薦安裝以獲得更好的性能
- **⚠️ FlashInfer 警告**：安裝 FlashInfer 可能會降級 PyTorch 版本（例如從 2.9.1+cu128 到 2.7.1+cu126），但 CUDA 支持通常會保留。如果不想降級，可以跳過 FlashInfer 安裝
- 從 v0.1.75 開始，olmOCR 使用 vLLM 而不是 sglang 作為推理引擎
- **NVIDIA 官方推薦配置**：CUDA 12.6.2.004 + CUDA Driver 560.35.03 + vLLM 0.5.5 + Python 3.10.12（最佳兼容性）
- **CUDA 兼容性說明**（基於 NVIDIA 官方文檔）：
  - **Minor Version Compatibility（向後兼容）**：
    - 較新的 CUDA Driver 可以運行較舊的 CUDA Runtime 構建的應用程序
    - 例如：Driver 560（支持 CUDA 12.6）可以運行 CUDA 12.4 構建的應用程序
    - PyTorch cu128 版本（為 CUDA 12.8 構建）可以在 CUDA 12.6/12.4 Driver 上運行（向後兼容）
  - **Forward Compatibility（向前兼容）**：
    - 在較舊的 Driver 上運行較新的 CUDA Toolkit 構建的應用程序
    - 需要安裝 `cuda-compat-<major>-<minor>` 包（例如：`cuda-compat-12-8`）
    - 僅適用於 NVIDIA Data Center GPUs、部分 RTX 卡和 Jetson 板
    - 例如：Driver 535 可以通過安裝 `cuda-compat-12-8` 來運行 CUDA 12.8 構建的應用程序
  - **建議**：
    - 優先使用 NVIDIA 官方推薦的配置以獲得最佳性能和穩定性
    - 如果 Driver 版本較舊，可以考慮安裝對應的 CUDA Forward Compatibility 包

#### 驗證安裝

```bash
python -c "import olmocr; print('olmOCR 安裝成功')"
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 版本: {torch.version.cuda}')"
python -c "import vllm; print(f'vLLM 版本: {vllm.__version__}')"  # 如果安裝成功，NVIDIA 推薦 0.5.5
```

### 步驟 5: 模型下載

程序會自動下載模型，但也可以手動預下載：

```bash
# 手動下載模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/olmOCR-2-7B-1025-FP8')"
```

如果遇到網絡問題，可以使用 HuggingFace 鏡像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/olmOCR-2-7B-1025-FP8')"
```

---

## 使用方法

### 基本使用

```bash
# 進入項目目錄
cd 03-advanced-tools

# 啟動公用環境
source .venv/bin/activate

# 運行 olmocr
cd olmocr
python demo.py
```

### 功能說明

1. **GPU 檢查**：自動檢查 GPU 記憶體使用狀況
2. **依賴檢查**：自動檢查 olmOCR 及其依賴是否已安裝
3. **模型預下載**：自動下載或檢查模型是否已下載
4. **批量處理**：自動處理父目錄 `../test_pdfs/` 下的所有 PDF 文件
5. **結果分析**：處理完成後會顯示統計信息和文件預覽
6. **結果保存**：處理結果會保存到 `output/olmocr_results.json`

### 輸出說明

- **處理結果 JSON**：`output/olmocr_results.json` 包含每個文件的處理詳情
- **轉換文件**：處理後的文檔會保存在 `output/` 目錄（JSON 或 Markdown 格式）

### 配置選項

可以在 `demo.py` 中調整以下參數：

```python
# 指定使用的 GPU 設備（文件開頭）
GPU_DEVICE = 1  # None 表示使用所有 GPU，0 表示 GPU0，1 表示 GPU1，依此類推

# 處理超時時間（秒，在 convert_pdf 函數中）
timeout=900

# GPU 記憶體使用率（在 convert_pdf 函數中）
--gpu-memory-utilization 0.80  # 80% GPU 記憶體

# 最大模型長度（在 convert_pdf 函數中）
--max_model_len 16384
```

**指定 GPU 設備**：
- 設置 `GPU_DEVICE = 1` 使用 GPU1
- 設置 `GPU_DEVICE = 0` 使用 GPU0
- 設置 `GPU_DEVICE = None` 使用所有可用 GPU

---

## 命令格式

olmOCR 使用以下命令格式：

```bash
python -m olmocr.pipeline <workspace> [OPTIONS]
```

### 基本命令

```bash
# 處理單個 PDF
python -m olmocr.pipeline /path/to/workspace --pdfs /path/to/document.pdf

# 處理多個 PDF
python -m olmocr.pipeline /path/to/workspace --pdfs pdf1.pdf pdf2.pdf pdf3.pdf

# 生成 Markdown 輸出
python -m olmocr.pipeline /path/to/workspace --markdown --pdfs /path/to/document.pdf
```

### 常用選項

- `workspace`：工作目錄路徑（必需的位置參數）
- `--pdfs [PDFS ...]`：要處理的 PDF 文件路徑（可多個）
- `--markdown`：同時生成 Markdown 格式輸出
- `--model MODEL`：模型路徑，默認為 `allenai/olmOCR-2-7B-1025-FP8`
- `--workers WORKERS`：並行工作進程數
- `--gpu-memory-utilization GPU_MEMORY_UTILIZATION`：GPU 記憶體使用率（0.0-1.0）
- `--max_model_len MAX_MODEL_LEN`：模型最大長度（tokens）
- `--pages_per_group PAGES_PER_GROUP`：每組處理的頁數

### 完整幫助

查看所有可用選項：

```bash
python -m olmocr.pipeline --help
```

---

## 性能指標

根據測試數據（掃描版 PDF，25 個樣本）：
- **平均處理時間**：15.2 秒/文件
- **準確率**：90%
- **記憶體消耗**：約 1.5GB
- **GPU 需求**：可選（有 GPU 會更快）

---

## 適用場景

✅ **推薦使用**：
- 掃描版 PDF 文檔
- 圖片轉文字
- 需要高準確率的 OCR 場景
- 多語言文檔識別

❌ **不推薦使用**：
- 純文本 PDF（浪費資源）
- 簡單格式文檔（處理速度慢）
- 對處理速度要求極高的場景

---

## 常見問題

### Q: `uv` 命令未找到

**A:** 安裝 uv：
```bash
pip install uv
# 或使用官方安裝腳本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Q: 虛擬環境啟動失敗

**A:** 檢查 Python 版本：
```bash
cd 03-advanced-tools
python3 --version  # 應為 3.10.12（NVIDIA 推薦）或 3.11+
source .venv/bin/activate
```

### Q: GPU 不可用

**A:** 檢查 NVIDIA 驅動和 CUDA：
```bash
nvidia-smi  # 檢查驅動和 CUDA Driver 版本（NVIDIA 推薦 560.35.03+）
nvcc --version  # 檢查 CUDA 版本（NVIDIA 推薦 12.6.2.004）
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA 版本: {torch.version.cuda}')"  # 檢查 PyTorch 使用的 CUDA 版本
python -c "import vllm; print(f'vLLM 版本: {vllm.__version__}')"  # NVIDIA 推薦 0.5.5
```

如果返回 False 或版本不匹配，重新安裝 GPU 版本的 PyTorch：

**CUDA 12.6（NVIDIA 官方推薦）**：
```bash
# ⚠️ 重要：必須明確指定版本號，否則可能安裝 cu126 版本
# CUDA 12.6 可以使用 cu128 版本（向後兼容）
uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# 或者安裝最新版本（如果可用）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.8**：
```bash
# ⚠️ 重要：必須明確指定版本號，否則可能安裝 cu126 版本
uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# 或者安裝最新版本（如果可用）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.4**：
```bash
# 嘗試 cu128（向後兼容）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 如果失敗，使用 cu124
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Q: CUDA 版本不匹配

**A:** NVIDIA 官方推薦配置：
- CUDA 版本：12.6.2.004
- CUDA Driver 版本：560.35.03 或更新
- vLLM 版本：0.5.5
- Python 版本：3.10.12

如果您的系統 CUDA 版本與推薦配置不同：

1. **CUDA 12.6（NVIDIA 推薦）**：
   - 可以使用 cu128 版本的 PyTorch（Minor Version Compatibility - 向後兼容）
   - 建議使用 NVIDIA 官方推薦的完整配置

2. **CUDA 12.4 或 12.8**：
   - PyTorch cu128 版本通常可以在 CUDA 12.4/12.6 系統上運行（Minor Version Compatibility - 向後兼容）
   - 如果遇到問題，檢查 `torch.version.cuda` 是否與系統 CUDA 版本匹配
   - 可以嘗試安裝匹配的 PyTorch 版本，或升級系統 CUDA 到 12.6（NVIDIA 推薦）

3. **Driver 版本較舊（例如 535）但需要運行 CUDA 12.8 構建的應用程序**：
   - 可以使用 **Forward Compatibility（向前兼容）**
   - 安裝對應的 CUDA Forward Compatibility 包：
     ```bash
     # Ubuntu/Debian
     sudo apt install cuda-compat-12-8
     
     # 設置環境變量（在運行應用程序前）
     export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
     ```
   - **注意**：Forward Compatibility 僅適用於 NVIDIA Data Center GPUs、部分 RTX 卡和 Jetson 板
   - 更多信息請參考 [NVIDIA CUDA Compatibility 文檔](https://docs.nvidia.com/cuda/cuda-compatibility/)

4. **最佳實踐**：
   - 優先使用 NVIDIA 官方推薦的配置（CUDA 12.6.2.004 + Driver 560.35.03 + vLLM 0.5.5）
   - 使用與系統 CUDA 版本匹配的 PyTorch 版本以獲得最佳性能和穩定性
   - 如果無法升級 Driver，考慮使用 Forward Compatibility 包

### Q: 模型下載失敗

**A:** 使用 HuggingFace 鏡像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/olmOCR-2-7B-1025-FP8')"
```

### Q: vLLM 或依賴安裝錯誤

**A:** 如果遇到依賴問題，請確保使用乾淨的 Python 環境：

```bash
cd 03-advanced-tools
source .venv/bin/activate

# 重新安裝 olmOCR（會自動安裝所有依賴）
uv pip install --force-reinstall "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

**為什麼安裝時間很長？**
- olmOCR 安裝通常需要 **10-20 分鐘**是正常的
- 需要下載大型依賴包（vLLM、PyTorch 等，數 GB）
- 可能需要編譯部分組件
- 請耐心等待，不要中斷安裝過程

### Q: FlashInfer 相關問題

**A:** 關於 FlashInfer 的說明：

1. **FlashInfer 是可選的**：
   - FlashInfer 用於加速推理，但不是必需的
   - 如果安裝失敗，olmOCR 仍然可以正常運行，只是速度可能較慢

2. **⚠️ 重要警告：FlashInfer 可能降級 PyTorch**：
   - 安裝 FlashInfer 可能會降級 PyTorch 版本（例如從 2.9.1+cu128 降級到 2.7.1+cu126）
   - **但 CUDA 支持仍然保留**（例如 cu126 或 cu128）
   - 如果遇到版本降級，可以選擇：
     - **選項 1**：接受降級（通常仍可正常工作，CUDA 支持保留）
     - **選項 2**：跳過 FlashInfer 安裝，保持最新 PyTorch 版本
     - **選項 3**：安裝 FlashInfer 後，重新安裝所需版本的 PyTorch：
       ```bash
       # ⚠️ 重要：必須明確指定版本號，否則可能安裝 cu126 版本
       uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
       
       # 驗證安裝
       python -c "import torch; print(f'版本: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
       ```

3. **如何安裝 FlashInfer**：
   ```bash
   # CUDA 12.8 (推薦，有官方 wheel)
   # ⚠️ 注意：可能會降級 PyTorch 版本
   uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
   
   # CUDA 12.4 用戶：
   # FlashInfer 可能沒有 cu124 的官方 wheel，可以跳過此步驟
   ```

4. **驗證安裝後**：
   ```bash
   # 檢查 PyTorch 版本和 CUDA 支持
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda}')"
   ```
   - 如果 `CUDA 可用: True`，則 GPU 支持正常
   - 如果版本降級但 CUDA 仍可用，通常可以正常使用

5. **如果 FlashInfer 安裝失敗**：
   - **CUDA 12.4 用戶**：這是正常的，可以安全跳過 FlashInfer 安裝
   - **CUDA 12.8 用戶**：檢查 CUDA 版本是否匹配，確認 PyTorch 版本兼容性
   - 無論如何，olmOCR 都可以正常運行，只是推理速度可能稍慢

### Q: GPU 記憶體不足

**A:** 這是常見問題，特別是當 GPU 記憶體較小時。解決方法：

1. **降低 GPU 記憶體使用率**：
   ```bash
   python -m olmocr.pipeline /path/to/workspace \
       --pdfs document.pdf \
       --markdown \
       --gpu-memory-utilization 0.80  # 使用 80% 的 GPU 記憶體（默認可能更高）
       --max_model_len 16384  # 限制最大模型長度
   ```

2. **進一步降低（如果仍然不足）**：
   ```bash
   python -m olmocr.pipeline /path/to/workspace \
       --pdfs document.pdf \
       --markdown \
       --gpu-memory-utilization 0.60  # 使用 60% 的 GPU 記憶體
       --max_model_len 8192  # 進一步降低最大模型長度
   ```

3. **檢查 GPU 記憶體**：
   ```bash
   nvidia-smi  # 查看 GPU 記憶體使用情況
   ```

4. **demo.py 已自動配置**：
   - `demo.py` 已自動設置 `--gpu-memory-utilization 0.80` 和 `--max_model_len 16384`
   - 如果仍然失敗，可以手動修改 `demo.py` 中的這些參數

**錯誤信息示例**：
```
If you run out of GPU memory during start-up or get 'KV cache is larger than available memory' errors, retry with lower values, e.g. --gpu_memory_utilization 0.80  --max_model_len 16384
```

**建議的 GPU 記憶體配置**：
- **16GB+ GPU**：`--gpu-memory-utilization 0.90`，`--max_model_len 16384`
- **12-16GB GPU**：`--gpu-memory-utilization 0.80`，`--max_model_len 16384`（demo.py 默認）
- **8-12GB GPU**：`--gpu-memory-utilization 0.60`，`--max_model_len 8192`
- **<8GB GPU**：可能需要使用 CPU 模式或升級 GPU

### Q: 處理超時

**A:** 默認超時時間為 900 秒（15 分鐘）。對於大型文檔，可以在 `demo.py` 中調整 `timeout` 參數。

### Q: 找不到生成的輸出文件

**A:** 檢查 `output/` 目錄。olmOCR 會在該目錄下創建處理結果文件。

### Q: 安裝 PyTorch cu128 後版本是 2.7.1+cu126 而不是 cu128

**A:** 這是因為 FlashInfer 或其他依賴的版本約束導致的。解決方法：

1. **明確指定版本號**：
   ```bash
   # ⚠️ 必須明確指定 +cu128 後綴
   uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

2. **驗證安裝**：
   ```bash
   python -c "import torch; print(f'版本: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
   # 應該顯示：版本: 2.7.1+cu128，CUDA: 12.8
   ```

3. **如果仍然安裝 cu126**：
   - 可能是 FlashInfer 的依賴約束
   - 可以暫時卸載 FlashInfer，安裝 PyTorch，再重新安裝 FlashInfer
   - 或者接受 cu126 版本（通常仍可正常工作）

**注意**：
- `torch==2.7.1` 和 `torch==2.7.1+cu128` 是不同的包
- 不指定版本號時，pip 會根據依賴約束選擇版本
- 明確指定 `+cu128` 後綴才能確保安裝 CUDA 12.8 版本

### Q: `pip list` 和 `uv pip list` 顯示的版本不一樣

**A:** 這是正常的，應該以 `uv pip list` 為準：

1. **`pip list`** 可能顯示系統級別的包（Python 3.10），不是虛擬環境中的包
2. **`uv pip list`** 顯示的是虛擬環境中實際安裝的包（Python 3.11）
3. **實際運行時**：Python 使用的是虛擬環境中的包

**驗證方法**：
```bash
source .venv/bin/activate
# 檢查實際使用的包版本
python -c "import torch; print(f'torch 版本: {torch.__version__}')"
python -c "import torch; print(f'torch 位置: {torch.__file__}')"  # 應指向 .venv 目錄
```

**建議**：
- 使用 `uv pip list` 查看虛擬環境中的包
- 使用 `uv pip install` 安裝包到虛擬環境
- 避免使用系統級別的 `pip`，它可能指向不同的 Python 環境

---

## 快速參考命令

### 完整安裝流程

```bash
# 1. 安裝 uv
pip install uv

# 2. 使用現有公用虛擬環境
cd 03-advanced-tools
source .venv/bin/activate

# 3. 安裝系統套件（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts \
    fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

# 4. 安裝 olmOCR（會自動安裝所有必需依賴）
# 根據您的 CUDA 版本選擇：
# CUDA 12.6 (NVIDIA 官方推薦，可以使用 cu128 向後兼容):
uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
# CUDA 12.8:
# uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
# CUDA 12.4 (通常可以使用 cu128，如果失敗則使用 cu124):
# uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
# 或
# uv pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu124

# 5. （可選）安裝 FlashInfer 以加速推理（僅 CUDA 12.8）
# CUDA 12.4 用戶可以跳過此步驟或查找對應版本
uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl

# 6. 驗證安裝
python -c "import olmocr; print('olmOCR 安裝成功')"

# 7. 運行 demo
cd olmocr
python demo.py
```

### 日常使用

```bash
cd 03-advanced-tools
source .venv/bin/activate
cd olmocr
python demo.py
deactivate
```

---

## 文件結構

```
03-advanced-tools/
├── .venv/              # 公用 OCR 虛擬環境（所有工具共享）
└── olmocr/
    ├── demo.py              # 主程序
    ├── README.md           # 本文件
    └── output/             # 輸出目錄（運行後生成）
        ├── olmocr_results.json  # 處理結果統計
        ├── *.json          # olmOCR 處理後的 JSON 文件
        └── *.md            # olmOCR 處理後的 Markdown 文件（如果有）
```

---

## NVIDIA 官方推薦配置

NVIDIA 官方推薦的 vLLM 容器配置（最佳兼容性）：

- **Container Name**: vllm-python-py3
- **Triton release version**: 24.10
- **NGC Tag**: `nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3`
- **Python version**: Python 3.10.12
- **vLLM version**: 0.5.5
- **CUDA version**: 12.6.2.004
- **CUDA Driver version**: 560.35.03
- **Size**: 21G

**使用 Docker 容器（可選）**：
```bash
# 拉取 NVIDIA 官方推薦的容器
docker pull nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3

# 運行容器
docker run --gpus all -it nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3
```

**注意**：如果使用本地安裝而非 Docker 容器，建議盡可能匹配上述配置以獲得最佳兼容性。

## 相關資源

- [olmOCR GitHub](https://github.com/allenai/olmocr)
- [HuggingFace 模型頁面](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [AllenAI 官方文檔](https://www.allenai.org/)
- [uv 官方文檔](https://github.com/astral-sh/uv)
- [PyTorch CUDA 安裝指南](https://pytorch.org/get-started/locally/)
- [vLLM 官方文檔](https://docs.vllm.ai/)
- [NVIDIA NGC 容器](https://catalog.ngc.nvidia.com/containers)

---

## 注意事項

1. **避免全域安裝**：使用 `uv` 虛擬環境可以避免污染系統的全域 Python 環境
2. **使用乾淨的環境**：官方建議在乾淨的 Python 環境中安裝 olmOCR，避免依賴衝突
3. **依賴自動安裝**：`olmocr[gpu]` 會自動安裝所有必需的依賴（vLLM、PyTorch、HuggingFace Hub 等）
4. **FlashInfer 是可選的**：FlashInfer 用於加速推理，但不是必需的。如果安裝失敗，olmOCR 仍可正常運行
5. **⚠️ FlashInfer 可能降級 PyTorch**：安裝 FlashInfer 可能會降級 PyTorch 版本，但 CUDA 支持通常會保留。如果不想降級，可以跳過 FlashInfer 安裝，或在安裝後重新安裝所需版本的 PyTorch
6. **安裝時間**：olmOCR 安裝需要 10-20 分鐘是正常的（包括下載大型依賴包），請耐心等待
7. **網絡問題**：如果遇到模型下載問題，可以使用 HuggingFace 鏡像
8. **推理引擎**：從 v0.1.75 開始，olmOCR 使用 vLLM 而不是 sglang 作為推理引擎
9. **NVIDIA 官方推薦配置**：為獲得最佳兼容性和性能，建議使用 NVIDIA 官方推薦的配置：
   - CUDA 12.6.2.004 + CUDA Driver 560.35.03 + vLLM 0.5.5 + Python 3.10.12
   - 或使用 NVIDIA NGC 容器：`nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3`

---

## 引用

如果您在研究中使用了 olmOCR，請引用：

**olmOCR v1 和 OlmOCR-bench**：
```bibtex
@misc{olmocrbench,
      title={{olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models}},
      author={Jake Poznanski and Jon Borchardt and Jason Dunkelberger and Regan Huff and Daniel Lin and Aman Rangapur and Christopher Wilhelm and Kyle Lo and Luca Soldaini},
      year={2025},
      eprint={2502.18443},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18443},
}
```

**olmOCR v2 Unit Testing Rewards with RL**：
```bibtex
@misc{olmocr2,
      title={olmOCR 2: Unit Test Rewards for Document OCR}, 
      author={Jake Poznanski and Luca Soldaini and Kyle Lo},
      year={2025},
      eprint={2510.19817},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.19817}, 
}
```


