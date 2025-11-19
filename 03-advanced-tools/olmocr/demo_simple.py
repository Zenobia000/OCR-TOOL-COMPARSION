#!/usr/bin/env python3
"""
olmOCR 簡化版本 - 讓 olmOCR 自己管理 SGLang server
"""

import subprocess
import sys
import os
from pathlib import Path

# 配置：指定使用的 GPU 設備
GPU_DEVICE = 1

def convert_pdf_simple(pdf_path):
    """使用 olmOCR 內建方式轉換PDF，讓它自己管理 server"""
    try:
        # 創建輸出目錄（workspace）
        output_dir = Path(__file__).parent / "output_simple"
        output_dir.mkdir(exist_ok=True)

        # olmOCR 使用 workspace 目錄
        workspace_dir = output_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)

        print(f"📁 工作目錄: {workspace_dir}")
        print(f"📄 處理文件: {pdf_path}")

        # 直接運行 olmOCR pipeline，讓它內部管理 SGLang
        cmd = [
            sys.executable, "-m", "olmocr.pipeline",
            str(workspace_dir),
            "--pdfs", str(pdf_path),
            "--max_page_error_rate", "0.3",  # 更寬鬆的錯誤容忍
        ]

        print(f"🚀 執行命令: olmocr.pipeline {' '.join(cmd[3:])}")

        # 設置環境變量
        env = os.environ.copy()
        if GPU_DEVICE is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)
            print(f"🎯 使用 GPU: {GPU_DEVICE}")

        # 執行命令，增加超時到 1800 秒 (30 分鐘)
        print("⏳ 開始處理...")
        result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=1800,
                              env=env)

        print(f"📊 返回碼: {result.returncode}")

        if result.stdout:
            print(f"📤 標準輸出:\n{result.stdout[-500:]}")  # 顯示最後 500 字符

        if result.stderr:
            print(f"🔍 錯誤輸出:\n{result.stderr[-500:]}")  # 顯示最後 500 字符

        # 檢查結果
        if result.returncode == 0:
            # 查找生成的文件
            all_files = list(workspace_dir.rglob('*'))
            output_files = [f for f in all_files if f.is_file() and f.suffix in ['.md', '.json']]

            print(f"✅ 處理完成！")
            print(f"📁 工作目錄: {workspace_dir}")
            print(f"📄 生成文件數量: {len(output_files)}")

            if output_files:
                print("📝 生成的文件:")
                for f in output_files[:5]:  # 顯示前5個
                    size = f.stat().st_size
                    print(f"   - {f.name} ({size} bytes)")
                if len(output_files) > 5:
                    print(f"   ... 還有 {len(output_files) - 5} 個文件")

            return {
                'success': True,
                'workspace': str(workspace_dir),
                'file_count': len(output_files),
                'files': [str(f.relative_to(workspace_dir)) for f in output_files[:10]]
            }
        else:
            return {
                'success': False,
                'error': f'olmOCR 處理失敗，返回碼: {result.returncode}',
                'stderr': result.stderr[-300:] if result.stderr else None
            }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '處理超時（超過30分鐘）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    test_pdf = "/home/os-sunnie.gd.weng/python_workstation/side-project/RAG/OCR-tool-comparsion/03-advanced-tools/test_pdfs/2021_CLIP.pdf"
    print("🧪 測試 olmOCR 簡化版本...")
    print("=" * 60)
    result = convert_pdf_simple(test_pdf)
    print("=" * 60)
    print(f"🏁 最終結果: {result}")