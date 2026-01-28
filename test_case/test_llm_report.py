#!/usr/bin/env python3
"""
NeuroScan AI - LLM 报告生成测试
使用本地部署的 Ollama 模型生成医学报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "reports_llm"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_ollama_connection():
    """测试 Ollama 连接"""
    print("🔗 测试 Ollama 连接...")
    
    try:
        import ollama
        models = ollama.list()
        print(f"   ✅ Ollama 连接成功")
        print(f"   📋 可用模型:")
        for model in models.get('models', []):
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024**3)  # GB
            print(f"      - {name} ({size:.1f} GB)")
        return True
    except Exception as e:
        print(f"   ❌ Ollama 连接失败: {e}")
        return False


def test_model_inference(model_name: str = "meditron:7b"):
    """测试模型推理"""
    print(f"\n🧠 测试模型推理: {model_name}")
    
    try:
        import ollama
        
        # 简单的医学问题测试
        prompt = """你是一名放射科医生。请简要描述以下 CT 影像发现的临床意义：

发现：右肺上叶后段可见一个 12mm 的部分实性结节，边界清晰，内部可见磨玻璃成分。

请用中文回答，限制在 100 字以内。"""
        
        print("   发送测试请求...")
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一名经验丰富的放射科医生，请用专业但简洁的语言回答问题。"},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 200
            }
        )
        
        result = response['message']['content']
        print(f"   ✅ 模型响应成功")
        print(f"\n   📝 模型回复:")
        print("   " + "-" * 50)
        for line in result.split('\n'):
            print(f"   {line}")
        print("   " + "-" * 50)
        
        return True
    except Exception as e:
        print(f"   ❌ 模型推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_report_with_llm():
    """使用 LLM 生成完整报告"""
    print("\n📄 使用 LLM 生成医学报告...")
    
    try:
        import ollama
        
        # 模拟的影像发现数据
        findings_data = {
            "patient_id": "LEARN2REG_001",
            "study_date": "2026-01-24",
            "modality": "CT",
            "body_part": "胸部",
            "findings": [
                {
                    "location": "右肺上叶后段",
                    "type": "结节",
                    "size_mm": 12.5,
                    "density": "部分实性",
                    "shape": "分叶状",
                    "margin": "边界清晰",
                    "calcification": "无",
                    "enhancement": "未评估"
                }
            ],
            "comparison": {
                "baseline_date": "2025-10-01",
                "baseline_size_mm": 10.0,
                "change_percent": 25.0,
                "interval_days": 115
            }
        }
        
        # 构建详细的 prompt
        system_prompt = """你是一名资深的放射科医生，专长于胸部 CT 影像诊断。
请根据提供的影像数据生成符合 ACR (美国放射学会) 标准的诊断报告。

报告要求：
1. 使用专业的医学术语
2. 结构清晰，包含：临床信息、影像所见、诊断印象、建议
3. 对于纵向对比，需要评估 RECIST 1.1 标准
4. 建议要具体可行
5. 使用中文撰写"""

        user_prompt = f"""请根据以下影像数据生成诊断报告：

患者信息:
- 患者 ID: {findings_data['patient_id']}
- 检查日期: {findings_data['study_date']}
- 检查类型: {findings_data['modality']}
- 检查部位: {findings_data['body_part']}

影像发现:
{json.dumps(findings_data['findings'], indent=2, ensure_ascii=False)}

纵向对比信息:
- 基线检查日期: {findings_data['comparison']['baseline_date']}
- 基线病灶大小: {findings_data['comparison']['baseline_size_mm']} mm
- 当前病灶大小: {findings_data['findings'][0]['size_mm']} mm
- 变化幅度: {findings_data['comparison']['change_percent']}%
- 检查间隔: {findings_data['comparison']['interval_days']} 天

请生成完整的诊断报告，包括 RECIST 1.1 评估。"""

        print("   发送报告生成请求...")
        print("   (这可能需要 30-60 秒...)")
        
        response = ollama.chat(
            model="meditron:7b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 1500
            }
        )
        
        report_content = response['message']['content']
        
        print("   ✅ 报告生成成功")
        
        # 保存报告
        report_path = OUTPUT_DIR / "llm_generated_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# NeuroScan AI - LLM 生成的诊断报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**使用模型**: meditron:7b\n\n")
            f.write("---\n\n")
            f.write(report_content)
            f.write("\n\n---\n\n")
            f.write("*本报告由 NeuroScan AI 使用本地 LLM 生成，仅供参考*\n")
        
        print(f"   📁 报告已保存: {report_path}")
        
        # 生成 HTML 版本
        html_report = generate_html_report(report_content, findings_data)
        html_path = OUTPUT_DIR / "llm_generated_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"   📁 HTML 报告: {html_path}")
        
        # 打印报告预览
        print(f"\n   📝 报告预览:")
        print("   " + "=" * 60)
        preview = report_content[:800] + "..." if len(report_content) > 800 else report_content
        for line in preview.split('\n'):
            print(f"   {line}")
        print("   " + "=" * 60)
        
        return True
        
    except Exception as e:
        print(f"   ❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_html_report(content: str, data: dict) -> str:
    """生成 HTML 格式报告"""
    
    # 简单的 Markdown 到 HTML 转换
    html_content = content.replace('\n', '<br>\n')
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI - LLM 生成报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(100, 255, 218, 0.2);
        }}
        .header h1 {{
            font-size: 2.2em;
            background: linear-gradient(90deg, #64ffda, #00d9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 5px;
        }}
        .badge.llm {{ background: linear-gradient(90deg, #ff6b6b, #ffa500); color: #fff; }}
        .badge.model {{ background: rgba(100, 255, 218, 0.2); color: #64ffda; }}
        .content {{
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(100, 255, 218, 0.1);
            line-height: 1.8;
        }}
        .content h2 {{ color: #64ffda; margin: 20px 0 10px 0; }}
        .content h3 {{ color: #00d9ff; margin: 15px 0 8px 0; }}
        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .meta-item {{
            background: rgba(100, 255, 218, 0.05);
            padding: 15px;
            border-radius: 10px;
        }}
        .meta-item label {{ color: #8892b0; font-size: 0.9em; }}
        .meta-item value {{ color: #64ffda; font-weight: 500; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 NeuroScan AI</h1>
            <p>LLM 生成的医学诊断报告</p>
            <div style="margin-top: 15px;">
                <span class="badge llm">🤖 AI 生成</span>
                <span class="badge model">meditron:7b</span>
            </div>
        </div>
        
        <div class="meta">
            <div class="meta-item">
                <label>患者 ID</label><br>
                <value>{data['patient_id']}</value>
            </div>
            <div class="meta-item">
                <label>检查日期</label><br>
                <value>{data['study_date']}</value>
            </div>
            <div class="meta-item">
                <label>检查类型</label><br>
                <value>{data['modality']} {data['body_part']}</value>
            </div>
            <div class="meta-item">
                <label>生成时间</label><br>
                <value>{datetime.now().strftime('%Y-%m-%d %H:%M')}</value>
            </div>
        </div>
        
        <div class="content">
            {html_content}
        </div>
        
        <div class="footer">
            <p>NeuroScan AI - 智能医学影像纵向诊断系统</p>
            <p style="margin-top: 10px;">⚠️ 本报告由 AI 辅助生成，仅供参考，不作为临床诊断依据</p>
        </div>
    </div>
</body>
</html>"""
    return html


def compare_models():
    """对比不同模型的生成效果"""
    print("\n🔬 对比不同模型的生成效果...")
    
    models = ["llama3.1:8b", "meditron:7b", "medllama2:7b"]
    prompt = """作为放射科医生，请用一句话描述以下发现的临床意义：
右肺上叶可见一个 12mm 部分实性结节，较 3 个月前增大 25%。"""
    
    results = {}
    
    try:
        import ollama
        
        for model in models:
            print(f"\n   测试模型: {model}")
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是放射科医生，请用中文简洁回答。"},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.1, "num_predict": 150}
                )
                results[model] = response['message']['content']
                print(f"   ✅ {model}: {results[model][:100]}...")
            except Exception as e:
                print(f"   ❌ {model}: 失败 - {e}")
                results[model] = f"Error: {e}"
        
        # 保存对比结果
        comparison_path = OUTPUT_DIR / "model_comparison.md"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("# LLM 模型对比测试\n\n")
            f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**测试问题**: {prompt}\n\n")
            f.write("---\n\n")
            for model, result in results.items():
                f.write(f"## {model}\n\n")
                f.write(f"{result}\n\n")
                f.write("---\n\n")
        
        print(f"\n   📁 对比结果已保存: {comparison_path}")
        
    except Exception as e:
        print(f"   ❌ 对比测试失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NeuroScan AI - LLM 报告生成测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 测试连接
    if not test_ollama_connection():
        print("\n❌ Ollama 未运行，请先启动: ollama serve")
        return False
    
    # 2. 测试模型推理
    if not test_model_inference("meditron:7b"):
        print("\n⚠️ 模型推理测试失败，尝试使用其他模型...")
        if not test_model_inference("llama3.1:8b"):
            return False
    
    # 3. 生成完整报告
    generate_report_with_llm()
    
    # 4. 对比不同模型
    compare_models()
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("✅ LLM 报告生成测试完成!")
    print("=" * 60)
    print(f"\n📁 报告位置: {OUTPUT_DIR}")
    print("\n🌐 查看方式:")
    print(f"   cd {OUTPUT_DIR} && python -m http.server 8892")
    print("   然后访问 http://localhost:8892/llm_generated_report.html")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

