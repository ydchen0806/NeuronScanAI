"""
Orchestrator - 中央调度器
协调多个专业 Agent 完成复杂任务
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.core.logging import logger
from app.schemas.analysis import AnalysisResult, ReportResponse, NoduleFinding
from .diagnostic_agent import DiagnosticAgent
from .comparative_agent import ComparativeAgent
from .prompts import REPORT_SYSTEM_PROMPT, REPORT_TEMPLATE


class OrchestratorState(dict):
    """Orchestrator 状态"""
    task_type: str  # "single" or "longitudinal"
    patient_id: str
    scans: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    comparisons: List[Dict[str, Any]]
    diagnostic_report: str
    comparative_report: str
    final_report: str
    key_images: List[str]
    status: str
    error: Optional[str]


class Orchestrator:
    """中央调度器"""
    
    def __init__(self, llm: ChatOpenAI = None):
        """
        初始化调度器
        
        Args:
            llm: LLM 实例 (可选)
        """
        if llm is None:
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
        else:
            self.llm = llm
        
        # 初始化子 Agent
        self.diagnostic_agent = DiagnosticAgent(self.llm)
        self.comparative_agent = ComparativeAgent(self.llm)
        
        # 构建工作流
        self.single_workflow = self._build_single_workflow()
        self.longitudinal_workflow = self._build_longitudinal_workflow()
    
    def _build_single_workflow(self) -> StateGraph:
        """构建单次分析工作流"""
        workflow = StateGraph(OrchestratorState)
        
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("run_diagnostic", self._run_diagnostic)
        workflow.add_node("generate_final_report", self._generate_final_report)
        
        workflow.set_entry_point("validate_input")
        workflow.add_edge("validate_input", "run_diagnostic")
        workflow.add_edge("run_diagnostic", "generate_final_report")
        workflow.add_edge("generate_final_report", END)
        
        return workflow.compile()
    
    def _build_longitudinal_workflow(self) -> StateGraph:
        """构建纵向分析工作流"""
        workflow = StateGraph(OrchestratorState)
        
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("run_diagnostic", self._run_diagnostic)
        workflow.add_node("run_comparative", self._run_comparative)
        workflow.add_node("generate_final_report", self._generate_final_report)
        
        workflow.set_entry_point("validate_input")
        workflow.add_edge("validate_input", "run_diagnostic")
        workflow.add_edge("run_diagnostic", "run_comparative")
        workflow.add_edge("run_comparative", "generate_final_report")
        workflow.add_edge("generate_final_report", END)
        
        return workflow.compile()
    
    def _validate_input(self, state: OrchestratorState) -> OrchestratorState:
        """验证输入"""
        logger.info("验证输入数据...")
        
        if not state.get("scans"):
            state["error"] = "未提供扫描数据"
            state["status"] = "failed"
        else:
            state["status"] = "processing"
        
        return state
    
    def _run_diagnostic(self, state: OrchestratorState) -> OrchestratorState:
        """运行诊断 Agent"""
        logger.info("运行诊断 Agent...")
        
        if state.get("error"):
            return state
        
        scans = state.get("scans", [])
        findings = state.get("findings", [])
        
        if scans:
            scan = scans[-1]  # 使用最新的扫描
            
            # 转换 findings 为 NoduleFinding 对象
            nodule_findings = []
            for f in findings:
                if isinstance(f, NoduleFinding):
                    nodule_findings.append(f)
                elif isinstance(f, dict):
                    nodule_findings.append(NoduleFinding(**f))
            
            result = self.diagnostic_agent.analyze(
                scan_id=scan.get("scan_id", "unknown"),
                patient_id=state.get("patient_id", "unknown"),
                study_date=scan.get("study_date", "unknown"),
                findings=nodule_findings,
                detected_organs=scan.get("detected_organs", []),
                body_part=scan.get("body_part", "胸部")
            )
            
            state["diagnostic_report"] = result.get("diagnosis", "")
        
        return state
    
    def _run_comparative(self, state: OrchestratorState) -> OrchestratorState:
        """运行对比 Agent"""
        logger.info("运行对比 Agent...")
        
        if state.get("error"):
            return state
        
        scans = state.get("scans", [])
        comparisons = state.get("comparisons", [])
        
        if len(scans) >= 2:
            baseline = scans[0]
            followup = scans[-1]
            
            result = self.comparative_agent.compare(
                patient_id=state.get("patient_id", "unknown"),
                baseline_scan_id=baseline.get("scan_id", "unknown"),
                baseline_date=baseline.get("study_date", "unknown"),
                followup_scan_id=followup.get("scan_id", "unknown"),
                followup_date=followup.get("study_date", "unknown"),
                comparisons=comparisons,
                heatmap_path=state.get("key_images", [None])[0]
            )
            
            state["comparative_report"] = result.get("analysis", "")
        
        return state
    
    def _generate_final_report(self, state: OrchestratorState) -> OrchestratorState:
        """生成最终报告"""
        logger.info("生成最终报告...")
        
        if state.get("error"):
            return state
        
        task_type = state.get("task_type", "single")
        
        # 组合报告
        if task_type == "single":
            report_type = "胸部 CT 检查报告"
            comparison_info = "无"
            findings_section = state.get("diagnostic_report", "无明显异常")
            impression_section = "详见影像所见"
        else:
            report_type = "胸部 CT 对比检查报告"
            scans = state.get("scans", [])
            if len(scans) >= 2:
                comparison_info = f"与 {scans[0].get('study_date', 'unknown')} 检查对比"
            else:
                comparison_info = "无"
            findings_section = state.get("diagnostic_report", "") + "\n\n" + state.get("comparative_report", "")
            impression_section = "详见影像所见及对比分析"
        
        # 使用模板生成报告
        scans = state.get("scans", [{}])
        latest_scan = scans[-1] if scans else {}
        
        final_report = REPORT_TEMPLATE.format(
            report_type=report_type,
            patient_id=state.get("patient_id", "Unknown"),
            study_date=latest_scan.get("study_date", "Unknown"),
            body_part=latest_scan.get("body_part", "胸部"),
            scanner_info="标准 CT 扫描",
            technique_description="常规胸部 CT 平扫，层厚 1.0mm，重建间隔 1.0mm",
            comparison_info=comparison_info,
            findings_section=findings_section,
            impression_section=impression_section,
            recommendations_section="1. 建议结合临床病史综合评估\n2. 必要时进一步检查",
            report_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        state["final_report"] = final_report
        state["status"] = "completed"
        
        return state
    
    def analyze_single(
        self,
        patient_id: str,
        scan_info: Dict[str, Any],
        findings: List[Dict[str, Any]]
    ) -> ReportResponse:
        """
        单次扫描分析
        
        Args:
            patient_id: 患者 ID
            scan_info: 扫描信息
            findings: 发现列表
            
        Returns:
            报告响应
        """
        initial_state = {
            "task_type": "single",
            "patient_id": patient_id,
            "scans": [scan_info],
            "findings": findings,
            "comparisons": [],
            "diagnostic_report": "",
            "comparative_report": "",
            "final_report": "",
            "key_images": [],
            "status": "pending",
            "error": None
        }
        
        result = self.single_workflow.invoke(initial_state)
        
        return ReportResponse(
            task_id=f"single_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status=result.get("status", "unknown"),
            markdown_report=result.get("final_report", ""),
            key_images=result.get("key_images", []),
            findings=[NoduleFinding(**f) if isinstance(f, dict) else f for f in findings]
        )
    
    def analyze_longitudinal(
        self,
        patient_id: str,
        baseline_scan: Dict[str, Any],
        followup_scan: Dict[str, Any],
        findings: List[Dict[str, Any]],
        comparisons: List[Dict[str, Any]],
        key_images: List[str] = None
    ) -> ReportResponse:
        """
        纵向对比分析
        
        Args:
            patient_id: 患者 ID
            baseline_scan: 基线扫描信息
            followup_scan: 随访扫描信息
            findings: 发现列表
            comparisons: 对比数据
            key_images: 关键图像路径
            
        Returns:
            报告响应
        """
        initial_state = {
            "task_type": "longitudinal",
            "patient_id": patient_id,
            "scans": [baseline_scan, followup_scan],
            "findings": findings,
            "comparisons": comparisons,
            "diagnostic_report": "",
            "comparative_report": "",
            "final_report": "",
            "key_images": key_images or [],
            "status": "pending",
            "error": None
        }
        
        result = self.longitudinal_workflow.invoke(initial_state)
        
        return ReportResponse(
            task_id=f"longitudinal_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status=result.get("status", "unknown"),
            markdown_report=result.get("final_report", ""),
            key_images=result.get("key_images", []),
            findings=[NoduleFinding(**f) if isinstance(f, dict) else f for f in findings],
            comparisons=comparisons
        )


