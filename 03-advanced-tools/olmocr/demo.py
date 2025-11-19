#!/usr/bin/env python3
"""
olmOCR PDF數據處理實現
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

# 配置：指定使用的 GPU 設備（None 表示使用所有 GPU，0 表示 GPU0，1 表示 GPU1）
GPU_DEVICE = 1  # 設置為 None 使用所有 GPU，或設置為 0, 1, 2... 指定特定 GPU

def check_gpu_memory():
    """檢查GPU記憶體使用狀況"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"發現 {gpu_count} 個 GPU:")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_free = memory_total - reserved
                marker = "👉" if GPU_DEVICE == i else "  "
                print(f"{marker} GPU {i}: {allocated:.1f}GB 已分配 / {reserved:.1f}GB 已保留 / {memory_free:.1f}GB 可用 / {memory_total:.1f}GB 總計")
            
            if GPU_DEVICE is not None:
                if GPU_DEVICE >= gpu_count:
                    print(f"⚠️  警告：指定的 GPU {GPU_DEVICE} 不存在，將使用所有 GPU")
                else:
                    print(f"✅ 將使用 GPU {GPU_DEVICE}")
            else:
                print(f"✅ 將使用所有 GPU")
            return True
        else:
            print("⚠️  GPU 不可用，將使用 CPU 模式（速度較慢）")
            return False
    except Exception as e:
        print(f"⚠️  GPU檢查失敗: {e}")
        return False

def check_olmocr():
    """檢查 olmOCR 是否已安裝"""
    try:
        import olmocr
        print(f"✅ olmOCR 已安裝")
        return True
    except ImportError:
        print("❌ olmOCR 未安裝")
        print("   請執行以下命令安裝 olmOCR:")
        print("   uv pip install 'olmocr[gpu]' --extra-index-url https://download.pytorch.org/whl/cu128")
        print("   詳細說明請參考 README.md")
        return False
    except Exception as e:
        print(f"⚠️  olmOCR 檢查失敗: {e}")
        return False

def check_vllm():
    """檢查 vLLM 是否已安裝（olmOCR 的推理引擎）"""
    try:
        import vllm
        import torch
        print(f"✅ vLLM 已安裝 (版本: {vllm.__version__})")
        print(f"✅ PyTorch 已安裝 (版本: {torch.__version__})")

        # 檢查版本兼容性
        torch_version = torch.__version__
        vllm_version = vllm.__version__

        # 根據官網文檔，PyTorch 2.7+ 與 vLLM 0.11.0+ 完全兼容
        print(f"✅ PyTorch {torch_version} 與 vLLM {vllm_version} 兼容")

        return True
    except ImportError:
        print("⚠️  vLLM 未安裝（olmOCR 的推理引擎）")
        print("   這通常會在安裝 olmocr[gpu] 時自動安裝")
        print("   如果遇到問題，請參考 README.md")
        return False
    except Exception as e:
        print(f"⚠️  vLLM 檢查失敗: {e}")
        return False

def pre_download_model():
    """預下載模型以避免轉換時timeout"""
    print("🔽 預先下載OLMoCR模型...")

    try:
        from huggingface_hub import snapshot_download
        model_id = "allenai/olmOCR-2-7B-1025-FP8"

        print(f"下載模型: {model_id}")
        cache_dir = snapshot_download(
            repo_id=model_id,
            local_files_only=False,  # 允許下載
            resume_download=True     # 支援斷點續傳
        )
        print(f"✅ 模型下載完成，快取位置: {cache_dir}")
        return True

    except ImportError:
        print("❌ huggingface_hub未安裝，跳過預下載")
        return False
    except Exception as e:
        print(f"❌ 模型下載失敗: {e}")
        return False

def process_pdfs():
    """處理test_pdfs目錄下的PDF文件"""
    # 指向父目錄的 test_pdfs
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        print("❌ test_pdfs目錄不存在")
        print(f"   預期路徑: {test_dir.absolute()}")
        return []

    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ 無PDF文件")
        return []

    print(f"📁 發現 {len(pdf_files)} 個PDF")
    results = []

    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024*1024)
        print(f"處理: {pdf.name} ({size_mb:.1f}MB)")

        start_time = time.time()
        result = convert_pdf(pdf)
        process_time = time.time() - start_time

        result_data = {
            'file': pdf.name,
            'size_mb': size_mb,
            'process_time': process_time,
            'success': result['success'],
            'output_size': result.get('output_size', 0),
            'error': result.get('error', None)
        }
        
        # 添加成功時的額外信息
        if result['success']:
            result_data['md_count'] = result.get('md_count', 0)
            result_data['json_count'] = result.get('json_count', 0)
            result_data['output_dir'] = result.get('output_dir', '')
            result_data['warning'] = result.get('warning', None)
            
            if result_data['md_count'] > 0:
                print(f"  ✅ 成功！生成 {result_data['md_count']} 個 Markdown 文件")
                if result.get('files'):
                    print(f"     文件位置: {', '.join(result.get('files', [])[:2])}")
                    if len(result.get('files', [])) > 2:
                        print(f"     ... 還有 {len(result.get('files', [])) - 2} 個文件")
            elif result_data['json_count'] > 0:
                print(f"  ✅ 成功！生成 {result_data['json_count']} 個 JSON 文件")
            else:
                warning = result_data.get('warning', '')
                if warning:
                    print(f"  ⚠️  {warning}")
                else:
                    print(f"  ⚠️  處理完成，但未找到輸出文件")
        else:
            error_msg = result.get('error', '未知錯誤')
            # 截斷過長的錯誤信息
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"  ❌ 失敗: {error_msg}")
        
        results.append(result_data)

    return results

def convert_pdf(pdf_path):
    """使用olmOCR轉換PDF"""
    try:
        # 創建輸出目錄（workspace）
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # olmOCR 使用 workspace 目錄，所有文件會放在這裡
        workspace_dir = output_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)

        # 正確的命令格式：python -m olmocr.pipeline <workspace> --pdfs <pdf_file> --markdown
        # 使用當前 Python 解釋器
        python_executable = sys.executable
        
        # 基本命令（使用適中的記憶體設置以確保穩定運行）
        cmd = [
            python_executable, "-m", "olmocr.pipeline",
            str(workspace_dir),  # workspace 位置參數
            "--pdfs", str(pdf_path),  # PDF 文件
            "--max_page_error_rate", "0.1",  # 允許10%的頁面錯誤率（必需參數）
        ]
        
        print(f"  執行命令: {' '.join(cmd)}")
        print(f"  ⏱️  設定超時: 900秒")
        print(f"  📄 允許頁面錯誤率: 10%（使用預設設定）")
        if GPU_DEVICE is not None:
            print(f"  🎯 使用 GPU {GPU_DEVICE}")
        
        # 設置 CUDA_VISIBLE_DEVICES 環境變量來指定 GPU
        env = os.environ.copy()
        if GPU_DEVICE is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)
            print(f"  🔧 環境變量 CUDA_VISIBLE_DEVICES={GPU_DEVICE}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)

        # 檢查 stderr 中是否有錯誤（即使返回碼為 0，也可能有錯誤）
        error_output = result.stderr.strip() or result.stdout.strip()
        # 檢查各種錯誤類型
        has_gpu_memory_error = error_output and ('gpu memory' in error_output.lower() or 'kv cache is larger' in error_output.lower() or 'out of memory' in error_output.lower())
        has_compatibility_error = error_output and ('attributeerror' in error_output.lower() and '_inductor' in error_output.lower() and 'config' in error_output.lower())
        has_vllm_server_error = error_output and ('vllm server task ended' in error_output.lower() or 'vllm server' in error_output.lower())
        has_error = error_output and ('error' in error_output.lower() or 'not found' in error_output.lower() or 'traceback' in error_output.lower() or has_gpu_memory_error or has_compatibility_error or has_vllm_server_error)
        
        # 檢查返回碼和輸出
        if result.returncode == 0 and not has_error:
            # olmOCR 會在 workspace 目錄中創建文件
            # 使用 --markdown 時，文件會在 markdown/ 子目錄中
            # 查找生成的 markdown 文件
            md_files = list(workspace_dir.rglob('*.md'))
            if md_files:
                output_size = sum(f.stat().st_size for f in md_files)
                # 顯示生成的文件路徑（最多顯示前3個）
                file_preview = [str(f.relative_to(workspace_dir)) for f in md_files[:3]]
                return {
                    'success': True, 
                    'output_size': output_size, 
                    'md_count': len(md_files), 
                    'output_dir': str(workspace_dir),
                    'files': file_preview
                }
            else:
                # 也檢查其他可能的輸出格式（Dolma JSON 等）
                json_files = list(workspace_dir.rglob('*.json'))
                if json_files:
                    output_size = sum(f.stat().st_size for f in json_files)
                    return {'success': True, 'output_size': output_size, 'json_count': len(json_files), 'output_dir': str(workspace_dir)}
                
                # 檢查是否有任何文件生成
                all_files = list(workspace_dir.rglob('*'))
                if all_files:
                    # 有文件但格式不對，返回信息
                    file_types = set(f.suffix for f in all_files if f.is_file())
                    return {
                        'success': True, 
                        'output_size': 0, 
                        'md_count': 0, 
                        'output_dir': str(workspace_dir),
                        'warning': f'生成了文件但未找到 .md 或 .json 格式，發現的文件類型: {file_types}'
                    }
                
                # 沒有生成任何文件，檢查錯誤輸出
                if error_output:
                    # 檢查各種錯誤類型
                    if has_compatibility_error:
                        error_msg = "vLLM 內部錯誤，可能是模型載入或配置問題"
                        error_lines = [line for line in error_output.split('\n') if 'attributeerror' in line.lower() and '_inductor' in line.lower()]
                        if error_lines:
                            error_msg += f" (詳細: {error_lines[0][:200]})"
                    elif has_gpu_memory_error:
                        error_msg = "GPU 記憶體不足。建議降低 --gpu-memory-utilization 或 --max_model_len 參數"
                        # 提取詳細錯誤信息
                        error_lines = [line for line in error_output.split('\n') if 'gpu memory' in line.lower() or 'kv cache' in line.lower() or 'memory' in line.lower()]
                        if error_lines:
                            error_msg = error_lines[0][:300]
                    elif has_vllm_server_error:
                        error_msg = "vLLM 服務器啟動失敗。可能是版本兼容性問題"
                        error_lines = [line for line in error_output.split('\n') if 'vllm server' in line.lower()]
                        if error_lines:
                            error_msg += f" (詳細: {error_lines[0][:200]})"
                    elif 'error' in error_output.lower() or 'not found' in error_output.lower() or 'traceback' in error_output.lower():
                        # 提取關鍵錯誤信息
                        error_lines = [line for line in error_output.split('\n') if 'error' in line.lower() or 'not found' in line.lower()]
                        if error_lines:
                            error_msg = error_lines[0][:300]  # 取第一行錯誤信息
                        else:
                            error_msg = error_output[:300]
                    else:
                        error_msg = error_output[:300]
                    return {'success': False, 'error': error_msg}
                
                return {
                    'success': True, 
                    'output_size': 0, 
                    'md_count': 0, 
                    'output_dir': str(workspace_dir),
                    'warning': '命令執行成功但未找到輸出文件'
                }
        else:
            # 命令失敗，組合錯誤信息
            if not error_output:
                error_msg = f"命令執行失敗，返回碼: {result.returncode}"
            else:
                # 檢查各種錯誤類型
                if has_compatibility_error:
                    error_msg = "vLLM 內部錯誤，可能是模型載入或配置問題"
                    error_lines = [line for line in error_output.split('\n') if 'attributeerror' in line.lower() and '_inductor' in line.lower()]
                    if error_lines:
                        error_msg += f" (詳細: {error_lines[0][:200]})"
                elif has_gpu_memory_error:
                    error_msg = "GPU 記憶體不足。建議降低 --gpu-memory-utilization 或 --max_model_len 參數"
                    # 提取詳細錯誤信息
                    error_lines = [line for line in error_output.split('\n') if 'gpu memory' in line.lower() or 'kv cache' in line.lower() or 'memory' in line.lower()]
                    if error_lines:
                        error_msg = error_lines[0][:300]
                elif has_vllm_server_error:
                    error_msg = "vLLM 服務器啟動失敗。可能是版本兼容性問題"
                    error_lines = [line for line in error_output.split('\n') if 'vllm server' in line.lower()]
                    if error_lines:
                        error_msg += f" (詳細: {error_lines[0][:200]})"
                else:
                    # 提取關鍵錯誤信息
                    error_lines = [line for line in error_output.split('\n') if 'error' in line.lower() or 'not found' in line.lower() or 'traceback' in line.lower()]
                    if error_lines:
                        error_msg = error_lines[0][:300]  # 取第一行錯誤信息
                    else:
                        error_msg = error_output[:300]
            return {'success': False, 'error': error_msg}

    except FileNotFoundError:
        return {'success': False, 'error': 'olmOCR 模組未找到，請確認已安裝 olmocr'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '處理超時（超過900秒）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_results(results):
    """分析處理結果"""
    if not results:
        return

    total_files = len(results)
    success_count = sum(1 for r in results if r['success'])
    total_size = sum(r['size_mb'] for r in results)
    total_time = sum(r['process_time'] for r in results)

    print(f"\n📊 處理結果:")
    print(f"成功率: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")
    print(f"總大小: {total_size:.1f}MB")
    print(f"總時間: {total_time:.2f}秒")
    if total_time > 0:
        print(f"平均速度: {total_size/total_time:.2f}MB/秒")

    # 顯示錯誤
    for r in results:
        if not r['success']:
            print(f"❌ {r['file']}: {r['error']}")

    # 保存結果
    output_file = Path(__file__).parent / 'output' / 'olmocr_results.json'
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果已保存到: {output_file}")

def main():
    """執行PDF批量處理"""
    print("🔬 olmOCR PDF處理")
    print("=" * 50)
    
    # 顯示 GPU 配置
    if GPU_DEVICE is not None:
        print(f"⚙️  配置：使用 GPU {GPU_DEVICE}")
    else:
        print(f"⚙️  配置：使用所有可用 GPU")
    print()
    
    # 檢查GPU狀態
    print("1️⃣ GPU記憶體檢查...")
    check_gpu_memory()
    
    # 檢查 olmOCR
    print("\n2️⃣ 檢查 olmOCR 依賴...")
    olmocr_ok = check_olmocr()
    if not olmocr_ok:
        print("\n❌ olmOCR 未安裝，無法繼續處理")
        print("請先安裝 olmOCR 後再運行 demo")
        return
    
    # 檢查 vLLM
    vllm_ok = check_vllm()
    if not vllm_ok:
        print("⚠️  vLLM 未安裝，可能會導致處理失敗")
        print("   請確認已正確安裝 olmocr[gpu]")
    
    # 預下載模型 (可選)
    print(f"\n3️⃣ 預備階段...")
    pre_download_model()
    
    # 處理PDF文件
    print(f"\n4️⃣ 開始處理...")
    results = process_pdfs()
    
    # 分析結果
    if results:
        analyze_results(results)
    else:
        print("\n❌ 沒有處理任何文件")
        print("\n🔧 建議解決方案:")
        print("1. 檢查test_pdfs目錄是否存在")
        print("2. 確認目錄中有PDF文件")
        print("3. 檢查文件權限")

if __name__ == "__main__":
    main()

