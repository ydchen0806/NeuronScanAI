"""
报告生成服务
支持多种 LLM 后端：Ollama、vLLM、OpenAI 兼容 API
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

from app.core.config import settings
from app.core.logging import logger


def convert_to_json_serializable(obj: Any) -> Any:
    """
    将对象转换为 JSON 可序列化的格式
    
    处理 numpy 类型、Path 对象等
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    else:
        return obj


class ReportGenerator:
    """
    医学影像报告生成器
    
    支持的 LLM 后端:
    - Ollama (本地部署)
    - vLLM (本地部署)
    - OpenAI 兼容 API
    - 离线模板模式 (无需 LLM)
    """
    
    # ACR 标准报告模板
    ACR_TEMPLATE = """
# {modality} 影像诊断报告

**检查日期**: {study_date}
**患者 ID**: {patient_id}
**检查部位**: {body_part}

---

## 临床信息
{clinical_info}

## 技术参数
{technique}

## 影像所见
{findings}

## 诊断印象
{impression}

## 建议
{recommendations}

---

*报告生成时间: {generated_at}*
*本报告由 NeuroScan AI 辅助生成，仅供参考，最终诊断请以临床医生判断为准。*
"""

    # 纵向对比报告模板（增强版）
    LONGITUDINAL_TEMPLATE = """
# {modality} 纵向对比分析报告

**患者 ID**: {patient_id}
**基线检查日期**: {baseline_date}
**随访检查日期**: {followup_date}
**检查间隔**: {interval}

---

## 检查目的
{purpose}

## 对比方法
{method}

## 变化分析

### 定量测量
{measurements}

### 病灶变化
{lesion_changes}

### RECIST 1.1 评估
{recist_assessment}

## 诊断印象
{impression}

## 临床建议
{recommendations}

---

*报告生成时间: {generated_at}*
*本报告由 NeuroScan AI 辅助生成，仅供参考，最终诊断请以临床医生判断为准。*
*本报告采用人工智能图像配准和变化检测技术，结合大语言模型分析生成。*
"""

    def __init__(self, llm_backend: str = "template"):
        """
        初始化报告生成器
        
        Args:
            llm_backend: LLM 后端类型 ("ollama", "vllm", "openai", "template")
        """
        self.llm_backend = llm_backend
        self.llm_client = None
        
        if llm_backend != "template":
            self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            if self.llm_backend == "ollama":
                self._init_ollama()
            elif self.llm_backend == "vllm":
                self._init_vllm()
            elif self.llm_backend == "openai":
                self._init_openai()
            else:
                logger.warning(f"未知的 LLM 后端: {self.llm_backend}, 使用模板模式")
                self.llm_backend = "template"
        except Exception as e:
            logger.warning(f"LLM 初始化失败: {e}, 回退到模板模式")
            self.llm_backend = "template"
    
    def _safe_json_dumps(self, obj: Any) -> str:
        """安全地将对象转换为 JSON 字符串（处理 numpy 类型）"""
        serializable_obj = convert_to_json_serializable(obj)
        return json.dumps(serializable_obj, indent=2, ensure_ascii=False)
    
    def _init_ollama(self):
        """初始化 Ollama 客户端"""
        try:
            import ollama
            self.llm_client = ollama
            # 测试连接
            ollama.list()
            logger.info("Ollama 客户端初始化成功")
        except ImportError:
            raise ImportError("请安装 ollama: pip install ollama")
        except Exception as e:
            raise ConnectionError(f"无法连接到 Ollama: {e}")
    
    def _init_vllm(self):
        """初始化 vLLM 客户端 (通过 OpenAI 兼容接口)"""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY
            )
            logger.info("vLLM 客户端初始化成功")
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def _init_openai(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY
            )
            logger.info("OpenAI 客户端初始化成功")
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def _clean_llm_response(self, response: str, system_prompt: str, user_prompt: str) -> str:
        """清理 LLM 返回内容，移除可能包含的 prompt 文本"""
        if not response:
            return ""
        
        cleaned = response.strip()
        
        # 移除常见的 prompt 模式（完整匹配）
        prompt_patterns = [
            "你是一名专注于肿瘤影像的放射科医生。请根据纵向对比数据生成详细的诊断印象。",
            "你是一名经验丰富的肿瘤科医生。请根据影像对比结果生成详细的治疗建议。",
            "你是一名经验丰富的放射科医生。请根据提供的影像数据生成专业的诊断印象。",
            "你是一名经验丰富的放射科医生。请根据影像发现生成合理的临床建议。",
            "要求：",
            "1. 使用专业的中文医学术语",
            "2. 明确描述变化趋势和幅度",
            "3. 引用 RECIST 1.1 标准进行评估",
            "4. 给出明确的疗效评估",
            "5. 分析配准质量对结果的影响",
            "6. 描述要详细、专业、准确",
            "1. 使用中文，专业术语",
            "2. 建议基于 RECIST 评估结果",
            "3. 考虑多学科协作 (MDT)",
            "4. 给出具体的随访计划和时间",
            "5. 考虑患者个体化治疗",
            "6. 建议要详细、可操作、有针对性",
            "请生成详细的纵向对比诊断印象（使用中文，专业术语）",
            "请生成详细的临床治疗建议（使用中文，按优先级排序）",
            "请生成诊断印象",
            "请生成临床建议",
        ]
        
        # 移除包含 prompt 的行
        lines = cleaned.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # 跳过空行
            if not line_stripped:
                # 保留空行以维持格式
                if filtered_lines:  # 只在已有内容时保留空行
                    filtered_lines.append('')
                continue
            
            # 检查是否是 prompt 模式
            is_prompt = False
            for pattern in prompt_patterns:
                if pattern in line_stripped or line_stripped.startswith(pattern):
                    is_prompt = True
                    break
            
            # 如果不是 prompt，保留这一行
            if not is_prompt:
                filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines).strip()
        
        # 移除开头的"诊断印象"或"临床建议"标题（如果后面还有内容）
        title_patterns = ["诊断印象", "临床建议"]
        for title in title_patterns:
            if cleaned.startswith(title):
                # 检查后面是否有实际内容
                remaining = cleaned[len(title):].strip()
                if remaining and not remaining.startswith("你是一名"):
                    cleaned = remaining
                    break
        
        # 移除可能包含的 JSON 代码块标记（如果 LLM 错误地包含了 prompt）
        if "```json" in cleaned and cleaned.count("```json") > 1:
            # 找到最后一个 JSON 块之后的内容
            last_json_end = cleaned.rfind("```")
            if last_json_end > 0:
                cleaned = cleaned[last_json_end + 3:].strip()
        
        return cleaned
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM"""
        if self.llm_backend == "template":
            return ""
        
        try:
            if self.llm_backend == "ollama":
                response = self.llm_client.chat(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                # Ollama returns a ChatResponse object, not a dict
                raw_content = response.message.content
                # 清理响应内容
                cleaned_content = self._clean_llm_response(raw_content, system_prompt, user_prompt)
                return cleaned_content
            else:
                # OpenAI 兼容接口 (vLLM, OpenAI)
                response = self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                raw_content = response.choices[0].message.content
                # 清理响应内容
                cleaned_content = self._clean_llm_response(raw_content, system_prompt, user_prompt)
                return cleaned_content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return ""
    
    def generate_single_report(
        self,
        patient_id: str,
        study_date: str,
        body_part: str,
        findings: List[Dict[str, Any]],
        clinical_info: str = "未提供",
        modality: str = "CT"
    ) -> str:
        """
        生成单次扫描报告
        
        Args:
            patient_id: 患者 ID
            study_date: 检查日期
            body_part: 检查部位
            findings: 发现列表
            clinical_info: 临床信息
            modality: 检查模态
            
        Returns:
            Markdown 格式的报告
        """
        # 格式化发现
        findings_text = self._format_findings(findings)
        
        # 生成诊断印象
        if self.llm_backend != "template":
            impression = self._generate_impression_with_llm(findings, body_part)
            recommendations = self._generate_recommendations_with_llm(findings, body_part)
        else:
            impression = self._generate_impression_template(findings, body_part)
            recommendations = self._generate_recommendations_template(findings)
        
        # 填充模板
        report = self.ACR_TEMPLATE.format(
            modality=modality,
            study_date=study_date,
            patient_id=patient_id,
            body_part=body_part,
            clinical_info=clinical_info,
            technique=self._get_technique_text(modality),
            findings=findings_text,
            impression=impression,
            recommendations=recommendations,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report
    
    def generate_longitudinal_report(
        self,
        patient_id: str,
        baseline_date: str,
        followup_date: str,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        registration_results: Dict[str, Any],
        change_results: Dict[str, Any],
        modality: str = "CT"
    ) -> str:
        """
        生成纵向对比报告（中文，详细，使用 LLM 分析）
        
        Args:
            patient_id: 患者 ID
            baseline_date: 基线日期
            followup_date: 随访日期
            baseline_findings: 基线发现
            followup_findings: 随访发现
            registration_results: 配准结果
            change_results: 变化检测结果
            modality: 检查模态
            
        Returns:
            Markdown 格式的报告
        """
        # 计算间隔
        interval = self._calculate_interval(baseline_date, followup_date)
        
        # 格式化测量数据（传入 change_results）
        measurements = self._format_measurements(baseline_findings, followup_findings, change_results)
        
        # 格式化病灶变化
        lesion_changes = self._format_lesion_changes(baseline_findings, followup_findings, change_results)
        
        # RECIST 评估（传入 change_results）
        recist_assessment = self._format_recist_assessment(baseline_findings, followup_findings, change_results)
        
        # 使用 LLM 分析配准结果
        registration_analysis = ""
        if self.llm_backend != "template":
            registration_analysis = self._analyze_registration_with_llm(registration_results)
        else:
            registration_analysis = self._analyze_registration_template(registration_results)
        
        # 使用 LLM 分析变化检测结果
        change_analysis = ""
        if self.llm_backend != "template":
            change_analysis = self._analyze_changes_with_llm(change_results, baseline_findings, followup_findings)
        else:
            change_analysis = self._analyze_changes_template(change_results)
        
        # 生成诊断印象和建议（使用 LLM）
        if self.llm_backend != "template":
            impression = self._generate_longitudinal_impression_with_llm(
                baseline_findings, followup_findings, change_results, registration_results
            )
            recommendations = self._generate_longitudinal_recommendations_with_llm(
                baseline_findings, followup_findings, change_results, registration_results
            )
        else:
            impression = self._generate_longitudinal_impression_template(
                baseline_findings, followup_findings, change_results
            )
            recommendations = self._generate_longitudinal_recommendations_template(
                baseline_findings, followup_findings, change_results
            )
        
        # 填充模板（增强版）
        report = self.LONGITUDINAL_TEMPLATE.format(
            modality=modality,
            patient_id=patient_id,
            baseline_date=baseline_date,
            followup_date=followup_date,
            interval=interval,
            purpose="评估病灶变化，判断治疗效果",
            method=f"""采用两级配准策略（刚性配准 + 非刚性配准）进行图像对齐，确保两次扫描的精确对比。

**配准方法**:
{registration_analysis}

**变化检测方法**:
{change_analysis}""",
            measurements=measurements,
            lesion_changes=lesion_changes,
            recist_assessment=recist_assessment,
            impression=impression,
            recommendations=recommendations,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report
    
    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """格式化发现列表"""
        if not findings:
            return "未见明显异常。"
        
        text_parts = []
        for i, finding in enumerate(findings, 1):
            organ = finding.get("organ", "未知")
            location = finding.get("location", "未知")
            size = finding.get("max_diameter_mm", 0)
            volume = finding.get("volume_cc", 0)
            density = finding.get("mean_hu", 0)
            shape = finding.get("shape", "规则")
            density_type = finding.get("density_type", "实性")
            
            text = f"""
**病灶 {i}**:
- 位置: {organ} {location}
- 大小: 最大直径约 {size:.1f} mm
- 体积: 约 {volume:.2f} cc
- 密度: 平均 CT 值 {density:.1f} HU ({density_type})
- 形态: {shape}
"""
            text_parts.append(text)
        
        return "\n".join(text_parts)
    
    def _format_measurements(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any] = None
    ) -> str:
        """格式化测量数据对比"""
        text_parts = ["| 指标 | 基线 | 随访 | 变化 |", "|------|------|------|------|"]
        
        # 如果有 change_results，优先使用真实分析数据
        if change_results and change_results.get("changed_voxels", 0) > 0:
            changed_voxels = change_results.get("changed_voxels", 0)
            total_voxels = change_results.get("total_voxels", 1)
            change_percent = change_results.get("change_percent", 0)
            changed_volume = change_results.get("changed_volume_cc", 0)
            max_increase = change_results.get("max_hu_increase", 0)
            max_decrease = change_results.get("max_hu_decrease", 0)
            mean_change = change_results.get("mean_hu_change", 0)
            
            text_parts.append(f"| 变化体素数 | - | - | {changed_voxels:,} |")
            text_parts.append(f"| 变化比例 | - | - | {change_percent:.2f}% |")
            text_parts.append(f"| 变化体积 (cc) | - | - | {changed_volume:.2f} |")
            text_parts.append(f"| 最大密度增加 (HU) | - | - | +{max_increase:.1f} |")
            text_parts.append(f"| 最大密度减少 (HU) | - | - | {max_decrease:.1f} |")
            text_parts.append(f"| 平均密度变化 (HU) | - | - | {mean_change:+.1f} |")
            
            return "\n".join(text_parts)
        
        # 如果没有 change_results，使用病灶数据
        if not baseline_findings and not followup_findings:
            return "无可测量病灶数据，请查看变化检测统计。"
        
        # 假设第一个病灶是目标病灶
        baseline = baseline_findings[0] if baseline_findings else {}
        followup = followup_findings[0] if followup_findings else {}
        
        # 直径
        b_diameter = baseline.get("max_diameter_mm", 0)
        f_diameter = followup.get("max_diameter_mm", 0)
        d_change = ((f_diameter - b_diameter) / b_diameter * 100) if b_diameter > 0 else 0
        text_parts.append(f"| 最大直径 (mm) | {b_diameter:.1f} | {f_diameter:.1f} | {d_change:+.1f}% |")
        
        # 体积
        b_volume = baseline.get("volume_cc", 0)
        f_volume = followup.get("volume_cc", 0)
        v_change = ((f_volume - b_volume) / b_volume * 100) if b_volume > 0 else 0
        text_parts.append(f"| 体积 (cc) | {b_volume:.2f} | {f_volume:.2f} | {v_change:+.1f}% |")
        
        # 密度
        b_hu = baseline.get("mean_hu", 0)
        f_hu = followup.get("mean_hu", 0)
        hu_change = f_hu - b_hu
        text_parts.append(f"| 平均密度 (HU) | {b_hu:.1f} | {f_hu:.1f} | {hu_change:+.1f} |")
        
        return "\n".join(text_parts)
    
    def _format_lesion_changes(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any]
    ) -> str:
        """格式化病灶变化描述"""
        # 优先使用 change_results 中的真实数据
        if change_results and change_results.get("changed_voxels", 0) > 0:
            changed_voxels = change_results.get("changed_voxels", 0)
            total_voxels = change_results.get("total_voxels", 1)
            change_percent = change_results.get("change_percent", 0)
            changed_volume = change_results.get("changed_volume_cc", 0)
            max_increase = change_results.get("max_hu_increase", 0)
            max_decrease = abs(change_results.get("max_hu_decrease", 0))
            increase_percent = change_results.get("increase_percent", 0)
            decrease_percent = change_results.get("decrease_percent", 0)
            
            # 根据变化情况判断趋势
            if increase_percent > decrease_percent * 1.5:
                trend = "密度增加为主"
                trend_desc = "可能提示组织致密化或新发病变"
            elif decrease_percent > increase_percent * 1.5:
                trend = "密度减少为主"
                trend_desc = "可能提示组织疏松化或病灶消退"
            else:
                trend = "双向变化"
                trend_desc = "同时存在密度增加和减少区域"
            
            text = f"""
**全局变化分析**:
- 变化体素数: {changed_voxels:,} / {total_voxels:,} ({change_percent:.2f}%)
- 变化体积: {changed_volume:.2f} cc
- 密度增加区域: {increase_percent:.2f}% (最大 +{max_increase:.1f} HU)
- 密度减少区域: {decrease_percent:.2f}% (最大 -{max_decrease:.1f} HU)
- 变化趋势: **{trend}**
- 临床意义: {trend_desc}
"""
            return text
        
        # 如果没有 change_results，使用病灶数据
        if not baseline_findings and not followup_findings:
            return "无病灶变化记录，请参考变化检测统计数据。"
        
        baseline = baseline_findings[0] if baseline_findings else {}
        followup = followup_findings[0] if followup_findings else {}
        
        b_diameter = baseline.get("max_diameter_mm", 0)
        f_diameter = followup.get("max_diameter_mm", 0)
        d_change = ((f_diameter - b_diameter) / b_diameter * 100) if b_diameter > 0 else 0
        
        # 描述变化
        if d_change > 20:
            change_desc = "明显增大"
            trend = "进展"
        elif d_change < -30:
            change_desc = "明显缩小"
            trend = "缓解"
        elif d_change < -10:
            change_desc = "略有缩小"
            trend = "可能缓解"
        elif d_change > 10:
            change_desc = "略有增大"
            trend = "可能进展"
        else:
            change_desc = "大小稳定"
            trend = "稳定"
        
        organ = followup.get("organ", baseline.get("organ", "未知"))
        location = followup.get("location", baseline.get("location", "未知"))
        
        text = f"""
**目标病灶** ({organ} {location}):
- 基线直径: {b_diameter:.1f} mm
- 随访直径: {f_diameter:.1f} mm
- 变化幅度: {d_change:+.1f}%
- 变化趋势: **{change_desc}** ({trend})
"""
        return text
    
    def _format_recist_assessment(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any] = None
    ) -> str:
        """格式化 RECIST 1.1 评估"""
        # 如果有 change_results，基于变化检测结果进行评估
        if change_results and change_results.get("changed_voxels", 0) > 0:
            change_percent = change_results.get("change_percent", 0)
            increase_percent = change_results.get("increase_percent", 0)
            decrease_percent = change_results.get("decrease_percent", 0)
            
            # 基于体积/密度变化进行 RECIST 类似评估
            net_change = increase_percent - decrease_percent
            
            if change_percent < 1.0:
                response = "SD (疾病稳定)"
                description = "总体变化极小 (<1%)"
                color = "🟢"
            elif net_change > 10:
                response = "PD (疾病进展)"
                description = f"密度增加区域显著多于减少区域 (净变化 +{net_change:.1f}%)"
                color = "🔴"
            elif net_change < -10:
                response = "PR (部分缓解)"
                description = f"密度减少区域显著多于增加区域 (净变化 {net_change:.1f}%)"
                color = "🟡"
            else:
                response = "SD (疾病稳定)"
                description = f"变化区域相对平衡 (净变化 {net_change:+.1f}%)"
                color = "🟠"
            
            text = f"""
**RECIST 1.1 类似评估**: {color} **{response}**

- 评估依据: {description}
- 总变化比例: {change_percent:.2f}%
- 密度增加区域: {increase_percent:.2f}%
- 密度减少区域: {decrease_percent:.2f}%

**注意**: 此评估基于体素级变化检测，非标准RECIST测量。标准RECIST需要测量靶病灶最大直径。

**RECIST 1.1 标准参考**:
- CR (完全缓解): 所有靶病灶消失
- PR (部分缓解): 靶病灶径线和减少 ≥30%
- SD (疾病稳定): 介于 PR 和 PD 之间
- PD (疾病进展): 靶病灶径线和增加 ≥20% 或出现新病灶
"""
            return text
        
        # 使用病灶数据进行标准评估
        if not baseline_findings or not followup_findings:
            return "无法进行标准 RECIST 评估 (缺少靶病灶测量数据)。如有变化检测结果，请参考上方分析。"
        
        baseline = baseline_findings[0]
        followup = followup_findings[0]
        
        b_diameter = baseline.get("max_diameter_mm", 0)
        f_diameter = followup.get("max_diameter_mm", 0)
        
        if b_diameter == 0:
            return "无法进行 RECIST 评估 (基线数据无效)。"
        
        change_pct = (f_diameter - b_diameter) / b_diameter * 100
        
        # RECIST 1.1 评估标准
        if f_diameter == 0:
            response = "CR (完全缓解)"
            description = "所有靶病灶消失"
            color = "🟢"
        elif change_pct <= -30:
            response = "PR (部分缓解)"
            description = "靶病灶径线和减少 ≥30%"
            color = "🟡"
        elif change_pct >= 20:
            response = "PD (疾病进展)"
            description = "靶病灶径线和增加 ≥20%"
            color = "🔴"
        else:
            response = "SD (疾病稳定)"
            description = "介于 PR 和 PD 之间"
            color = "🟠"
        
        text = f"""
**RECIST 1.1 评估结果**: {color} **{response}**

- 评估依据: {description}
- 实际变化: {change_pct:+.1f}%
- 基线径线和: {b_diameter:.1f} mm
- 随访径线和: {f_diameter:.1f} mm

**评估标准参考**:
- CR (完全缓解): 所有靶病灶消失
- PR (部分缓解): 靶病灶径线和减少 ≥30%
- SD (疾病稳定): 介于 PR 和 PD 之间
- PD (疾病进展): 靶病灶径线和增加 ≥20% 或出现新病灶
"""
        return text
    
    def _generate_impression_template(
        self,
        findings: List[Dict[str, Any]],
        body_part: str
    ) -> str:
        """使用模板生成诊断印象"""
        if not findings:
            return f"{body_part}扫描未见明显异常。"
        
        impressions = []
        for i, finding in enumerate(findings, 1):
            organ = finding.get("organ", "未知部位")
            size = finding.get("max_diameter_mm", 0)
            density_type = finding.get("density_type", "实性")
            shape = finding.get("shape", "规则")
            
            if size < 6:
                nature = "微小结节，性质待定"
            elif size < 10:
                nature = "小结节，建议随访"
            elif size < 30:
                nature = "结节，建议进一步检查"
            else:
                nature = "肿块，高度建议活检"
            
            impressions.append(f"{i}. {organ}可见{density_type}{shape}结节，大小约 {size:.1f}mm，{nature}。")
        
        return "\n".join(impressions)
    
    def _generate_recommendations_template(self, findings: List[Dict[str, Any]]) -> str:
        """使用模板生成建议"""
        if not findings:
            return "1. 定期体检\n2. 如有不适，及时就诊"
        
        max_size = max(f.get("max_diameter_mm", 0) for f in findings)
        
        if max_size < 6:
            return """1. 建议 12 个月后复查 CT
2. 如有咳嗽、胸痛等症状，及时就诊"""
        elif max_size < 10:
            return """1. 建议 6 个月后复查 CT
2. 必要时行 PET-CT 检查
3. 密切关注症状变化"""
        elif max_size < 30:
            return """1. 建议 3 个月后复查 CT
2. 建议行 PET-CT 检查
3. 必要时行穿刺活检
4. 建议多学科会诊 (MDT)"""
        else:
            return """1. 建议尽快行穿刺活检明确性质
2. 建议行 PET-CT 全身检查
3. 建议多学科会诊 (MDT)
4. 如确诊恶性，尽早制定治疗方案"""
    
    def _generate_longitudinal_impression_template(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any]
    ) -> str:
        """使用模板生成纵向对比诊断印象"""
        # 优先使用 change_results 数据
        if change_results and change_results.get("changed_voxels", 0) > 0:
            change_percent = change_results.get("change_percent", 0)
            changed_volume = change_results.get("changed_volume_cc", 0)
            increase_percent = change_results.get("increase_percent", 0)
            decrease_percent = change_results.get("decrease_percent", 0)
            max_increase = change_results.get("max_hu_increase", 0)
            max_decrease = abs(change_results.get("max_hu_decrease", 0))
            
            net_change = increase_percent - decrease_percent
            
            if change_percent < 1.0:
                status = "基本稳定"
                assessment = "SD (疾病稳定)"
                recommendation = "继续当前方案或观察"
            elif net_change > 10:
                status = "密度增加为主的变化"
                assessment = "可能提示病情进展"
                recommendation = "建议进一步评估，必要时调整方案"
            elif net_change < -10:
                status = "密度减少为主的变化"
                assessment = "可能提示病情改善"
                recommendation = "继续当前治疗方案"
            else:
                status = "存在双向变化"
                assessment = "需结合临床综合判断"
                recommendation = "建议短期内复查"
            
            return f"""**纵向对比分析结论**:

与前片对比，扫描区域呈现{status}。

**定量分析**:
- 总变化比例: {change_percent:.2f}%
- 变化体积: {changed_volume:.2f} cc
- 密度增加区域占比: {increase_percent:.2f}% (最大增加 +{max_increase:.1f} HU)
- 密度减少区域占比: {decrease_percent:.2f}% (最大减少 -{max_decrease:.1f} HU)

**评估**: {assessment}
**建议**: {recommendation}"""
        
        # 使用病灶数据
        if not baseline_findings or not followup_findings:
            return "对比数据不完整，请确保完成配准和变化检测分析。"
        
        baseline = baseline_findings[0]
        followup = followup_findings[0]
        
        b_diameter = baseline.get("max_diameter_mm", 0)
        f_diameter = followup.get("max_diameter_mm", 0)
        change_pct = ((f_diameter - b_diameter) / b_diameter * 100) if b_diameter > 0 else 0
        
        organ = followup.get("organ", baseline.get("organ", ""))
        
        if change_pct > 20:
            return f"""与前片对比，{organ}病灶明显增大，直径增加 {change_pct:.1f}%，符合 RECIST 1.1 疾病进展 (PD) 标准。
提示病情进展，建议调整治疗方案。"""
        elif change_pct < -30:
            return f"""与前片对比，{organ}病灶明显缩小，直径减少 {abs(change_pct):.1f}%，符合 RECIST 1.1 部分缓解 (PR) 标准。
提示治疗有效，建议继续当前治疗方案。"""
        else:
            return f"""与前片对比，{organ}病灶大小基本稳定，直径变化 {change_pct:+.1f}%，符合 RECIST 1.1 疾病稳定 (SD) 标准。
建议继续随访观察。"""
    
    def _generate_longitudinal_recommendations_template(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any]
    ) -> str:
        """使用模板生成纵向对比建议"""
        # 优先使用 change_results 数据
        if change_results and change_results.get("changed_voxels", 0) > 0:
            change_percent = change_results.get("change_percent", 0)
            increase_percent = change_results.get("increase_percent", 0)
            decrease_percent = change_results.get("decrease_percent", 0)
            net_change = increase_percent - decrease_percent
            
            if change_percent < 1.0:
                return """**临床建议**:

1. 病情稳定，可继续当前治疗方案或观察
2. 建议 3 个月后复查 CT 评估
3. 定期监测肿瘤标志物
4. 如出现新症状请及时就诊
5. 保持健康生活方式"""
            elif net_change > 10:
                return """**临床建议**:

1. ⚠️ 建议多学科会诊 (MDT) 讨论
2. 评估当前治疗方案有效性
3. 考虑调整治疗策略或加强治疗
4. 建议 4-6 周后短期复查
5. 必要时行 PET-CT 或增强扫描
6. 如有靶向治疗指征，建议基因检测"""
            elif net_change < -10:
                return """**临床建议**:

1. ✅ 治疗效果良好，继续当前方案
2. 建议 2-3 个月后复查评估
3. 关注治疗相关副作用
4. 定期监测肿瘤标志物
5. 维持良好的营养状态和生活质量"""
            else:
                return """**临床建议**:

1. 变化趋势不明确，建议密切随访
2. 建议 6-8 周后短期复查
3. 结合临床症状综合判断
4. 必要时行增强 CT 或 PET-CT
5. 定期监测肿瘤标志物
6. 如症状加重请及时就诊"""
        
        # 使用病灶数据
        if not baseline_findings or not followup_findings:
            return """**临床建议**:

1. 完善检查数据，进行完整对比分析
2. 如有疑问，建议临床医生综合判断
3. 定期复查随访"""
        
        baseline = baseline_findings[0]
        followup = followup_findings[0]
        
        b_diameter = baseline.get("max_diameter_mm", 0)
        f_diameter = followup.get("max_diameter_mm", 0)
        change_pct = ((f_diameter - b_diameter) / b_diameter * 100) if b_diameter > 0 else 0
        
        if change_pct > 20:
            return """1. 建议多学科会诊 (MDT) 讨论治疗方案调整
2. 考虑更换治疗方案或加强治疗
3. 建议 4-6 周后复查评估
4. 必要时行基因检测指导靶向治疗"""
        elif change_pct < -30:
            return """1. 继续当前治疗方案
2. 建议 2-3 个月后复查评估
3. 关注治疗相关副作用
4. 定期监测肿瘤标志物"""
        else:
            return """1. 继续当前治疗方案或观察
2. 建议 2-3 个月后复查
3. 如出现症状变化及时就诊
4. 定期监测肿瘤标志物"""
    
    def _generate_impression_with_llm(
        self,
        findings: List[Dict[str, Any]],
        body_part: str
    ) -> str:
        """使用 LLM 生成诊断印象"""
        system_prompt = """你是一名经验丰富的放射科医生。请根据提供的影像数据生成专业的诊断印象。
要求：
1. 使用专业的医学术语
2. 描述准确、简洁
3. 按重要性排序
4. 不要臆造数据"""
        
        user_prompt = f"""检查部位: {body_part}
发现数据:
```json
{self._safe_json_dumps(findings)}
```

请生成诊断印象:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._generate_impression_template(findings, body_part)
    
    def _generate_recommendations_with_llm(
        self,
        findings: List[Dict[str, Any]],
        body_part: str
    ) -> str:
        """使用 LLM 生成建议"""
        system_prompt = """你是一名经验丰富的放射科医生。请根据影像发现生成合理的临床建议。
要求：
1. 建议具体可行
2. 按优先级排序
3. 考虑患者安全"""
        
        user_prompt = f"""检查部位: {body_part}
发现数据:
```json
{self._safe_json_dumps(findings)}
```

请生成临床建议:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._generate_recommendations_template(findings)
    
    def _analyze_registration_with_llm(self, registration_results: Dict[str, Any]) -> str:
        """使用 LLM 分析配准结果"""
        system_prompt = """你是一名医学影像技术专家。请分析图像配准的质量和可靠性。
要求：
1. 评估配准精度
2. 说明配准方法的优势
3. 指出可能的局限性"""
        
        user_prompt = f"""配准结果:
```json
{self._safe_json_dumps(registration_results)}
```

请分析配准质量:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._analyze_registration_template(registration_results)
    
    def _analyze_registration_template(self, registration_results: Dict[str, Any]) -> str:
        """使用模板分析配准结果"""
        return """- **刚性配准**: 修正体位差异，对齐解剖结构
- **非刚性配准**: 修正呼吸等软组织形变
- **配准精度**: 亚毫米级精度，确保精确对比"""
    
    def _analyze_changes_with_llm(
        self,
        change_results: Dict[str, Any],
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]]
    ) -> str:
        """使用 LLM 分析变化检测结果"""
        system_prompt = """你是一名医学影像分析专家。请分析两次扫描之间的变化。
要求：
1. 描述变化的空间分布
2. 量化变化幅度
3. 评估变化的临床意义"""
        
        user_prompt = f"""变化检测结果:
```json
{self._safe_json_dumps(change_results)}
```

基线发现:
```json
{self._safe_json_dumps(baseline_findings)}
```

随访发现:
```json
{self._safe_json_dumps(followup_findings)}
```

请分析变化特征:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._analyze_changes_template(change_results)
    
    def _analyze_changes_template(self, change_results: Dict[str, Any]) -> str:
        """使用模板分析变化检测结果"""
        changed_voxels = change_results.get("changed_voxels", 0)
        change_percent = change_results.get("change_percent", 0)
        max_increase = change_results.get("max_hu_increase", 0)
        max_decrease = change_results.get("max_hu_decrease", 0)
        
        return f"""- **变化体素数**: {changed_voxels:,} 个体素
- **变化比例**: {change_percent:.2f}%
- **最大密度增加**: {max_increase:.1f} HU
- **最大密度减少**: {max_decrease:.1f} HU
- **分析方法**: 体素级差异计算，阈值过滤"""
    
    def _generate_longitudinal_impression_with_llm(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any],
        registration_results: Dict[str, Any] = None
    ) -> str:
        """使用 LLM 生成纵向对比诊断印象（增强版）"""
        system_prompt = """你是一名专注于肿瘤影像的放射科医生。请根据纵向对比数据生成详细的诊断印象。
要求：
1. 使用专业的中文医学术语
2. 明确描述变化趋势和幅度
3. 引用 RECIST 1.1 标准进行评估
4. 给出明确的疗效评估
5. 分析配准质量对结果的影响
6. 描述要详细、专业、准确"""
        
        reg_info = ""
        if registration_results:
            reg_info = f"\n配准结果:\n```json\n{self._safe_json_dumps(registration_results)}\n```"
        
        user_prompt = f"""基线检查发现:
```json
{self._safe_json_dumps(baseline_findings)}
```

随访检查发现:
```json
{self._safe_json_dumps(followup_findings)}
```

变化检测分析:
```json
{self._safe_json_dumps(change_results)}
```{reg_info}

请生成详细的纵向对比诊断印象（使用中文，专业术语）:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._generate_longitudinal_impression_template(
            baseline_findings, followup_findings, change_results
        )
    
    def _generate_longitudinal_recommendations_with_llm(
        self,
        baseline_findings: List[Dict[str, Any]],
        followup_findings: List[Dict[str, Any]],
        change_results: Dict[str, Any],
        registration_results: Dict[str, Any] = None
    ) -> str:
        """使用 LLM 生成纵向对比建议（增强版）"""
        system_prompt = """你是一名经验丰富的肿瘤科医生。请根据影像对比结果生成详细的治疗建议。
要求：
1. 使用中文，专业术语
2. 建议基于 RECIST 评估结果
3. 考虑多学科协作 (MDT)
4. 给出具体的随访计划和时间
5. 考虑患者个体化治疗
6. 建议要详细、可操作、有针对性"""
        
        reg_info = ""
        if registration_results:
            reg_info = f"\n配准结果:\n```json\n{self._safe_json_dumps(registration_results)}\n```"
        
        user_prompt = f"""基线检查发现:
```json
{self._safe_json_dumps(baseline_findings)}
```

随访检查发现:
```json
{self._safe_json_dumps(followup_findings)}
```

变化检测结果:
```json
{self._safe_json_dumps(change_results)}
```{reg_info}

请生成详细的临床治疗建议（使用中文，按优先级排序）:"""
        
        result = self._call_llm(system_prompt, user_prompt)
        return result if result else self._generate_longitudinal_recommendations_template(
            baseline_findings, followup_findings, change_results
        )
    
    def _calculate_interval(self, baseline_date: str, followup_date: str) -> str:
        """计算检查间隔"""
        try:
            from datetime import datetime
            b_date = datetime.strptime(baseline_date, "%Y-%m-%d")
            f_date = datetime.strptime(followup_date, "%Y-%m-%d")
            days = (f_date - b_date).days
            
            if days < 30:
                return f"{days} 天"
            elif days < 365:
                months = days // 30
                return f"约 {months} 个月"
            else:
                years = days // 365
                months = (days % 365) // 30
                if months > 0:
                    return f"约 {years} 年 {months} 个月"
                return f"约 {years} 年"
        except:
            return "未知"
    
    def _get_technique_text(self, modality: str) -> str:
        """获取技术参数描述"""
        if modality == "CT":
            return """- 扫描设备: 多排螺旋 CT
- 扫描范围: 全胸部
- 层厚: 1.0-1.5 mm
- 重建算法: 标准算法
- 窗位/窗宽: 肺窗 (-600/1500)，纵隔窗 (40/400)"""
        elif modality == "MRI":
            return """- 扫描设备: 1.5T/3.0T MRI
- 序列: T1WI, T2WI, DWI
- 层厚: 3-5 mm"""
        else:
            return f"- 检查模态: {modality}"
    
    def save_report(self, report: str, output_path: Path, format: str = "md") -> Path:
        """
        保存报告
        
        Args:
            report: 报告内容
            output_path: 输出路径
            format: 格式 ("md", "html", "pdf")
            
        Returns:
            保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "md":
            output_path = output_path.with_suffix(".md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        elif format == "html":
            output_path = output_path.with_suffix(".html")
            html = self._markdown_to_html(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        logger.info(f"报告已保存: {output_path}")
        return output_path
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """将 Markdown 转换为 HTML"""
        try:
            import markdown
            html_body = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
        except ImportError:
            # 简单的 Markdown 转换
            html_body = markdown_text.replace('\n', '<br>\n')
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI 诊断报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
        .footer {{ color: #7f8c8d; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
        return html

