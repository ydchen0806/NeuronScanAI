#!/usr/bin/env python3
"""
LLM 部署和配置脚本

支持的 LLM 后端:
1. Ollama (推荐用于本地部署)
2. vLLM (高性能推理)
3. OpenAI 兼容 API

推荐模型:
- llama3:8b-instruct (轻量级，适合测试)
- llama3:70b-instruct (PRD 推荐)
- meditron:7b (医学专用)
- med42-v2 (医学专用)
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_ollama():
    """检查 Ollama 是否安装"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Ollama 已安装: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Ollama 未安装")
    return False


def install_ollama():
    """安装 Ollama"""
    print("\n📦 安装 Ollama...")
    print("请访问 https://ollama.ai 下载安装")
    print("\n或使用以下命令:")
    print("  curl -fsSL https://ollama.ai/install.sh | sh")
    
    return False


def list_ollama_models():
    """列出已安装的 Ollama 模型"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("\n📋 已安装的模型:")
            print(result.stdout)
            return result.stdout
    except:
        pass
    return ""


def pull_ollama_model(model_name: str):
    """下载 Ollama 模型"""
    print(f"\n⬇️  下载模型: {model_name}")
    print("这可能需要几分钟...")
    
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ 模型 {model_name} 下载完成")
            return True
        else:
            print(f"❌ 模型下载失败")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_ollama_model(model_name: str):
    """测试 Ollama 模型"""
    print(f"\n🧪 测试模型: {model_name}")
    
    try:
        import ollama
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一名放射科医生，请用中文回答。"
                },
                {
                    "role": "user",
                    "content": "请简单描述肺结节的影像特征。"
                }
            ]
        )
        
        print(f"\n📝 模型响应:")
        print("-" * 40)
        print(response['message']['content'])
        print("-" * 40)
        
        return True
    except ImportError:
        print("❌ 请安装 ollama Python 包: pip install ollama")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def update_config(llm_model: str, llm_backend: str = "ollama"):
    """更新配置文件"""
    env_path = PROJECT_ROOT / ".env"
    
    config = {
        "LLM_MODEL": llm_model,
        "LLM_BASE_URL": "http://localhost:11434/v1" if llm_backend == "ollama" else "http://localhost:8000/v1",
        "LLM_API_KEY": "ollama" if llm_backend == "ollama" else "local-key",
        "LLM_TEMPERATURE": "0.1",
        "LLM_MAX_TOKENS": "4096"
    }
    
    # 读取现有配置
    existing_config = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_config[key] = value
    
    # 更新配置
    existing_config.update(config)
    
    # 写入配置
    with open(env_path, 'w') as f:
        f.write("# NeuroScan AI 配置\n\n")
        f.write("# LLM 配置\n")
        for key, value in existing_config.items():
            f.write(f"{key}={value}\n")
    
    print(f"\n✅ 配置已更新: {env_path}")
    print(f"   LLM_MODEL={llm_model}")
    print(f"   LLM_BASE_URL={config['LLM_BASE_URL']}")


def test_report_generation():
    """测试报告生成"""
    print("\n🧪 测试报告生成...")
    
    from app.services.report import ReportGenerator
    
    # 使用模板模式测试 (不需要 LLM)
    generator = ReportGenerator(llm_backend="template")
    
    # 测试单次扫描报告
    findings = [
        {
            "nodule_id": "nodule_1",
            "organ": "右肺上叶",
            "location": "后段",
            "max_diameter_mm": 12.5,
            "volume_cc": 0.85,
            "mean_hu": 35.2,
            "shape": "分叶状",
            "density_type": "部分实性"
        }
    ]
    
    report = generator.generate_single_report(
        patient_id="TEST_001",
        study_date="2026-01-24",
        body_part="胸部",
        findings=findings,
        clinical_info="体检发现肺结节",
        modality="CT"
    )
    
    print("\n📄 单次扫描报告 (模板模式):")
    print("=" * 50)
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    # 保存报告
    output_dir = PROJECT_ROOT / "test_case" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = generator.save_report(
        report,
        output_dir / "single_scan_report",
        format="html"
    )
    print(f"\n✅ 报告已保存: {report_path}")
    
    # 测试纵向对比报告
    baseline_findings = [
        {
            "nodule_id": "nodule_1",
            "organ": "右肺上叶",
            "location": "后段",
            "max_diameter_mm": 10.0,
            "volume_cc": 0.52,
            "mean_hu": 32.0,
            "shape": "圆形",
            "density_type": "实性"
        }
    ]
    
    followup_findings = [
        {
            "nodule_id": "nodule_1",
            "organ": "右肺上叶",
            "location": "后段",
            "max_diameter_mm": 12.5,
            "volume_cc": 0.85,
            "mean_hu": 35.2,
            "shape": "分叶状",
            "density_type": "部分实性"
        }
    ]
    
    longitudinal_report = generator.generate_longitudinal_report(
        patient_id="TEST_001",
        baseline_date="2025-10-01",
        followup_date="2026-01-24",
        baseline_findings=baseline_findings,
        followup_findings=followup_findings,
        registration_results={"mae_before": 432.5, "mae_after": 385.5},
        change_results={"diameter_change_pct": 25.0},
        modality="CT"
    )
    
    print("\n📄 纵向对比报告 (模板模式):")
    print("=" * 50)
    print(longitudinal_report[:1000] + "..." if len(longitudinal_report) > 1000 else longitudinal_report)
    
    report_path = generator.save_report(
        longitudinal_report,
        output_dir / "longitudinal_report",
        format="html"
    )
    print(f"\n✅ 报告已保存: {report_path}")
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("NeuroScan AI - LLM 部署配置")
    print("=" * 60)
    
    # 1. 检查 Ollama
    has_ollama = check_ollama()
    
    if has_ollama:
        # 列出已安装模型
        models = list_ollama_models()
        
        # 推荐的医学模型
        recommended_models = [
            ("llama3:8b-instruct", "通用模型，适合测试"),
            ("llama3.1:8b-instruct", "最新通用模型"),
            ("meditron:7b", "医学专用模型"),
            ("medllama2:7b", "医学专用模型"),
        ]
        
        print("\n💡 推荐的模型:")
        for model, desc in recommended_models:
            print(f"   - {model}: {desc}")
        
        # 检查是否有可用模型
        if "llama3" in models.lower() or "meditron" in models.lower():
            print("\n✅ 已有可用的 LLM 模型")
        else:
            print("\n⚠️  建议下载一个模型:")
            print("   ollama pull llama3:8b-instruct")
    else:
        print("\n💡 Ollama 是推荐的本地 LLM 部署方案")
        print("   安装后可以运行: ollama pull llama3:8b-instruct")
    
    # 2. 测试报告生成 (模板模式)
    print("\n" + "=" * 60)
    print("测试报告生成 (模板模式 - 不需要 LLM)")
    print("=" * 60)
    
    try:
        test_report_generation()
        print("\n✅ 报告生成测试成功!")
    except Exception as e:
        print(f"\n❌ 报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("部署总结")
    print("=" * 60)
    
    print("""
📋 报告生成模块已就绪!

支持的模式:
1. 模板模式 (无需 LLM): 使用预定义模板生成报告
2. Ollama 模式: 使用本地 LLM 生成更智能的报告
3. vLLM/OpenAI 模式: 使用高性能推理服务

使用方法:
```python
from app.services.report import ReportGenerator

# 模板模式
generator = ReportGenerator(llm_backend="template")

# Ollama 模式 (需要先安装 Ollama 和模型)
generator = ReportGenerator(llm_backend="ollama")

# 生成报告
report = generator.generate_single_report(
    patient_id="P001",
    study_date="2026-01-24",
    body_part="胸部",
    findings=[...],
    modality="CT"
)
```

如需使用 LLM 增强报告:
1. 安装 Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. 下载模型: ollama pull llama3:8b-instruct
3. 启动服务: ollama serve
4. 使用 Ollama 模式: ReportGenerator(llm_backend="ollama")
""")


if __name__ == "__main__":
    main()

