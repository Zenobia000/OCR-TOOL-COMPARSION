#!/usr/bin/env python3
"""
olmOCR PDF數據處理實現 - 修正版
"""

import subprocess
import sys
import os
import time
import json
import socket
from pathlib import Path

# 配置：指定使用的 GPU 設備
GPU_DEVICE = 1

def start_sglang_server(model_path, port=30024):
    """啟動優化記憶體設定的 SGLang server"""
    try:
        # 檢查端口是否已被佔用
        def is_port_open(host="127.0.0.1", port=port, timeout=1.0):
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                return False

        if is_port_open(port=port):
            print(f"  ✅ SGLang server 已在 port {port} 運行")
            return None

        print(f"  🚀 啟動 SGLang server (CPU offload 配置)...")
        print(f"     模型: {model_path}")
        print(f"     CPU Offload: 8GB 模型權重到 RAM")
        print(f"     記憶體設定: mem_fraction_static=0.3")
        print(f"     Context: 1024 tokens")

        # 設置 CUDA_VISIBLE_DEVICES 環境變量
        env = os.environ.copy()
        if GPU_DEVICE is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)
            print(f"     GPU: {GPU_DEVICE}")

        # 啟動 SGLang server，使用 CPU offload 減少 GPU 記憶體壓力
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-running-requests", "1",       # 限制並發請求
            "--context-length", "1024",          # 保守的 context 長度
            "--cpu-offload-gb", "8",             # 將 8GB 模型權重 offload 到 CPU RAM
            "--mem-fraction-static", "0.3",     # 適中的 KV cache 設定
        ]

        print(f"  ⚙️  啟動命令: {' '.join(cmd[2:])}")

        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        # 等待 server 啟動
        print(f"  ⏳ 等待 SGLang server 啟動 (最多60秒)...")
        start_time = time.time()
        while time.time() - start_time < 60:
            # 檢查進程是否還在運行
            if server_proc.poll() is not None:
                # 進程已經退出，讀取錯誤信息
                stdout, stderr = server_proc.communicate()
                print(f"  ❌ SGLang server 進程已退出，返回碼: {server_proc.returncode}")
                if stderr:
                    print(f"  錯誤輸出: {stderr[:500]}")
                if stdout:
                    print(f"  標準輸出: {stdout[:500]}")
                return None

            if is_port_open(port=port):
                print(f"  ✅ SGLang server 已成功啟動在 port {port}")
                return server_proc
            time.sleep(2)

        # 超時處理
        print(f"  ❌ SGLang server 啟動超時")
        # 獲取進程輸出
        try:
            stdout, stderr = server_proc.communicate(timeout=1)
            if stderr:
                print(f"  超時時錯誤輸出: {stderr[:500]}")
            if stdout:
                print(f"  超時時標準輸出: {stdout[:500]}")
        except subprocess.TimeoutExpired:
            print("  無法獲取進程輸出")

        server_proc.terminate()
        return None

    except Exception as e:
        print(f"  ❌ SGLang server 啟動失敗: {e}")
        return None

def convert_pdf_fixed(pdf_path):
    """使用olmOCR轉換PDF - 修正版"""
    server_proc = None

    try:
        # 創建輸出目錄（workspace）
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # olmOCR 使用 workspace 目錄，所有文件會放在這裡
        workspace_dir = output_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)

        # 1. 先啟動 SGLang server（記憶體優化版）
        model_path = "allenai/olmOCR-7B-0225-preview"  # 回到原始模型，但加上 CPU offload
        server_proc = start_sglang_server(model_path)
        if server_proc is None:
            return {'success': False, 'error': 'SGLang server 啟動失敗，無法處理 PDF'}

        # 2. 運行 olmOCR pipeline（連接到現有 server）
        cmd = [
            sys.executable, "-m", "olmocr.pipeline",
            str(workspace_dir),  # workspace 位置參數
            "--pdfs", str(pdf_path),  # PDF 文件
            "--max_page_error_rate", "0.1",  # 允許10%的頁面錯誤率（必需參數）
            "--model", model_path,  # 使用 FP8 量化版本
            "--model_max_context", "1024",  # 匹配 SGLang server 的 context 設定
        ]

        print(f"  執行命令: olmocr.pipeline {' '.join(cmd[3:])}")
        print(f"  ⏱️  設定超時: 900秒")
        print(f"  📄 允許頁面錯誤率: 10%")
        print(f"  🧠 模型: olmOCR-7B-0225-preview (Qwen2-VL 兼容版)")
        print(f"  📏 最大 Context: 2048 tokens")
        print(f"  💾 記憶體優化: KV cache 限制在 ~3.8GB")

        # 設置 CUDA_VISIBLE_DEVICES 環境變量
        env = os.environ.copy()
        if GPU_DEVICE is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)

        # 檢查 stderr 中是否有錯誤
        error_output = result.stderr.strip() or result.stdout.strip()
        has_gpu_memory_error = error_output and ('gpu memory' in error_output.lower() or 'kv cache is larger' in error_output.lower() or 'out of memory' in error_output.lower())
        has_compatibility_error = error_output and ('attributeerror' in error_output.lower() and '_inductor' in error_output.lower() and 'config' in error_output.lower())
        has_vllm_server_error = error_output and ('vllm server task ended' in error_output.lower() or 'vllm server' in error_output.lower())
        has_error = error_output and ('error' in error_output.lower() or 'not found' in error_output.lower() or 'traceback' in error_output.lower() or has_gpu_memory_error or has_compatibility_error or has_vllm_server_error)

        # 檢查返回碼和輸出
        if result.returncode == 0 and not has_error:
            # 查找生成的 markdown 文件
            md_files = list(workspace_dir.rglob('*.md'))
            if md_files:
                output_size = sum(f.stat().st_size for f in md_files)
                file_preview = [str(f.relative_to(workspace_dir)) for f in md_files[:3]]
                return {
                    'success': True,
                    'output_size': output_size,
                    'md_count': len(md_files),
                    'output_dir': str(workspace_dir),
                    'files': file_preview
                }
            else:
                # 檢查其他可能的輸出格式
                json_files = list(workspace_dir.rglob('*.json'))
                if json_files:
                    output_size = sum(f.stat().st_size for f in json_files)
                    return {'success': True, 'output_size': output_size, 'json_count': len(json_files), 'output_dir': str(workspace_dir)}

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
                if has_compatibility_error:
                    error_msg = "vLLM 內部錯誤，可能是模型載入或配置問題"
                elif has_gpu_memory_error:
                    error_msg = "GPU 記憶體不足，已嘗試優化但仍不足"
                elif has_vllm_server_error:
                    error_msg = "vLLM 服務器啟動失敗，可能是版本兼容性問題"
                else:
                    error_msg = error_output[:300]
            return {'success': False, 'error': error_msg}

    except FileNotFoundError:
        return {'success': False, 'error': 'olmOCR 模組未找到，請確認已安裝 olmocr'}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '處理超時（超過900秒）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        # 清理 SGLang server 進程
        if server_proc is not None:
            try:
                print(f"  🧹 清理 SGLang server 進程...")
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception as e:
                print(f"  ⚠️  清理進程時出現問題: {e}")
                try:
                    server_proc.kill()
                except:
                    pass

if __name__ == "__main__":
    test_pdf = "/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/OCR-tool-comparsion/03-advanced-tools/test_pdfs/2021_CLIP.pdf"
    print("🧪 測試修正版 olmOCR...")
    result = convert_pdf_fixed(test_pdf)
    print(f"結果: {result}")