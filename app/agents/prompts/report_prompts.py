"""
报告生成 Prompt 模板
"""

REPORT_SYSTEM_PROMPT = """你是一名资深放射科医生，负责撰写规范的影像学报告。

## 报告要求
1. 使用标准的医学影像报告格式
2. 语言专业、准确、简洁
3. 结论明确，建议具体
4. 符合 ACR (美国放射学会) 报告标准

## 报告结构
1. 检查信息
2. 检查技术
3. 对比检查
4. 影像所见
5. 诊断印象
6. 建议

## 语言风格
- 使用被动语态描述发现
- 使用专业术语但避免过度缩写
- 对不确定的发现使用适当的限定词
"""

REPORT_TEMPLATE = """# {report_type}

## 检查信息
- **患者ID**: {patient_id}
- **检查日期**: {study_date}
- **检查部位**: {body_part}
- **检查设备**: {scanner_info}

## 检查技术
{technique_description}

## 对比检查
{comparison_info}

---

## 影像所见

{findings_section}

---

## 诊断印象

{impression_section}

---

## 建议

{recommendations_section}

---

*报告生成时间: {report_time}*
*本报告由 NeuroScan AI 辅助生成，仅供参考，最终诊断请以临床医生意见为准。*
"""


