# MinerU PDF 處理工具

## 簡介

MinerU 是一個強大的 PDF 解析工具，特別適合處理複雜格式的 PDF 文檔，包括圖文混排、表格密集的文檔。它使用 `mineru` 命令進行處理，能夠保持文檔的原始格式。

## 特點

- ✅ **格式保持優秀**：對複雜格式文檔的格式保持率可達 95%
- ✅ **處理複雜文檔**：擅長處理圖文混排、表格密集的 PDF
- ⚠️ **資源消耗高**：需要 GPU 支持，記憶體消耗約 4GB
- ⚠️ **處理速度較慢**：平均處理時間較長，適合對準確性要求高的場景

## 安裝要求

### 1. 使用公用 OCR 環境

**本項目使用統一的 OCR 虛擬環境** (`.venv`)，位於 `03-advanced-tools/` 目錄。

```bash
# 進入項目目錄
cd 03-advanced-tools

# 啟動公用環境
source .venv/bin/activate
```

### 2. 安裝 MinerU

```bash
# 確保環境已啟動
source .venv/bin/activate

# 使用 uv 安裝（推薦）
uv pip install -U "mineru[core]"

# 或使用 pip 安裝
pip install -U "mineru[core]"
```

### 3. 系統要求

- Python 3.8+
- CUDA 支持的 GPU（推薦）
- 至少 4GB 可用記憶體

## 使用方法

### 基本使用

```bash
# 進入項目目錄並啟動環境
cd 03-advanced-tools
source .venv/bin/activate

# 運行 mineru
cd mineru
python demo.py
```

### 功能說明

1. **自動掃描測試文件**：程序會自動掃描父目錄 `../test_pdfs/` 下的所有 PDF 文件
2. **批量處理**：自動處理所有找到的 PDF 文件
3. **結果分析**：處理完成後會顯示統計信息
4. **結果保存**：處理結果會保存到 `mineru_results.json`

### 輸出說明

- **處理結果 JSON**：`mineru_results.json` 包含每個文件的處理詳情
- **Markdown 文件**：處理後的文檔會轉換為 Markdown 格式，保存在 `output/<pdf_name>/<pdf_name>/auto/<pdf_name>.md`
- **圖片文件**：提取的圖片保存在 `output/<pdf_name>/<pdf_name>/auto/images/` 目錄
- **其他文件**：可能還包含 JSON 格式的內容列表、布局 PDF 等

## 性能指標

根據測試數據（複雜格式 PDF，20 個樣本）：
- **平均處理時間**：31.8 秒/文件
- **格式保持率**：95%
- **記憶體消耗**：約 4GB
- **GPU 需求**：必須

## 適用場景

✅ **推薦使用**：
- 複雜格式的學術論文
- 圖文混排的技術文檔
- 表格密集的報告
- 需要高格式保持率的場景

❌ **不推薦使用**：
- 純文本 PDF（處理速度慢）
- 簡單格式文檔（資源浪費）
- 對處理速度要求高的批量處理

## 常見問題

### Q: 提示 "mineru命令未找到"

**A:** 請確認已正確安裝 `mineru`，並確保命令在 PATH 中。

```bash
# 檢查是否安裝
which mineru
mineru --version

# 如果未找到，重新安裝
cd 03-advanced-tools
source .venv/bin/activate
uv pip install -U "mineru[core]"
# 或
pip install -U "mineru[core]"
```

### Q: 處理超時

**A:** 默認超時時間為 600 秒（10分鐘）。對於大型或複雜文檔，可能需要更長時間。可以修改 `demo.py` 中的 `timeout` 參數。

### Q: GPU 記憶體不足

**A:** MinerU 需要較多 GPU 記憶體。如果遇到記憶體不足，可以：
1. 關閉其他使用 GPU 的程序
2. 處理較小的文件
3. 考慮使用其他工具（如 Unstructured）處理簡單文檔

## 文件結構

```
mineru/
├── demo.py              # 主程序
├── README.md           # 本文件
├── mineru_results.json # 處理結果統計（運行後生成）
└── output/             # 輸出目錄（運行後生成）
    ├── <pdf_name>/     # 每個PDF的輸出目錄
    │   └── <pdf_name>/
    │       └── auto/
    │           ├── <pdf_name>.md          # Markdown 輸出
    │           ├── <pdf_name>_content_list.json  # 內容列表
    │           ├── <pdf_name>_layout.pdf  # 布局PDF
    │           └── images/               # 提取的圖片
    │               └── *.jpg
    └── ...
```

## 命令格式

MinerU 使用以下命令格式：

```bash
mineru -p <input_path> -o <output_path> [OPTIONS]
```

常用選項：
- `-p, --path`: 輸入文件路徑（PDF、PNG、JPG等）
- `-o, --output`: 輸出目錄
- `-m, --method`: 解析方法（auto/txt/ocr），默認為 auto
- `-b, --backend`: 後端引擎（pipeline/vlm-transformers等），默認為 pipeline
- `-l, --lang`: 語言設置（ch/en/korean等），默認為 ch

示例：
```bash
# 基本使用
mineru -p document.pdf -o output/

# 指定方法和語言
mineru -p document.pdf -o output/ -m auto -l en
```

## 相關資源

- [MinerU 官方文檔](https://github.com/opendatalab/MinerU)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)

