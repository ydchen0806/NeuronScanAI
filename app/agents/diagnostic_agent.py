"""
诊断 Agent - 单次扫描分析
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.core.logging import logger
from app.schemas.analysis import NoduleFinding, AnalysisResult
from .prompts import DIAGNOSTIC_SYSTEM_PROMPT, DIAGNOSTIC_USER_TEMPLATE


class DiagnosticAgentState(dict):
    """诊断 Agent 状态"""
    scan_id: str
    patient_id: str
    study_date: str
    body_part: str
    findings: List[NoduleFinding]
    detected_organs: List[str]
    diagnosis: str
    recommendations: str


class DiagnosticAgent:
    """诊断专家 Agent"""
    
    def __init__(self, llm: ChatOpenAI = None):
        """
        初始化诊断 Agent
        
        Args:
            llm: LLM 实例
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
        
        # 构建工作流图
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(DiagnosticAgentState)
        
        # 添加节点
        workflow.add_node("prepare_data", self._prepare_data)
        workflow.add_node("generate_diagnosis", self._generate_diagnosis)
        workflow.add_node("format_report", self._format_report)
        
        # 添加边
        workflow.set_entry_point("prepare_data")
        workflow.add_edge("prepare_data", "generate_diagnosis")
        workflow.add_edge("generate_diagnosis", "format_report")
        workflow.add_edge("format_report", END)
        
        return workflow.compile()
    
    def _prepare_data(self, state: DiagnosticAgentState) -> DiagnosticAgentState:
        """准备数据节点"""
        logger.info(f"准备诊断数据: {state.get('scan_id')}")
        
        # 数据已经在 state 中，直接返回
        return state
    
    def _generate_diagnosis(self, state: DiagnosticAgentState) -> DiagnosticAgentState:
        """生成诊断节点"""
        logger.info("生成诊断...")
        
        # 构建 prompt
        findings_json = json.dumps(
            [f.model_dump() if hasattr(f, 'model_dump') else f for f in state.get("findings", [])],
            indent=2,
            ensure_ascii=False
        )
        
        user_prompt = DIAGNOSTIC_USER_TEMPLATE.format(
            patient_id=state.get("patient_id", "Unknown"),
            study_date=state.get("study_date", "Unknown"),
            body_part=state.get("body_part", "胸部"),
            findings_json=findings_json,
            detected_organs=", ".join(state.get("detected_organs", []))
        )
        
        # 调用 LLM
        messages = [
            SystemMessage(content=DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["diagnosis"] = response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            state["diagnosis"] = f"诊断生成失败: {str(e)}"
        
        return state
    
    def _format_report(self, state: DiagnosticAgentState) -> DiagnosticAgentState:
        """格式化报告节点"""
        logger.info("格式化报告...")
        
        # 诊断结果已经在 state["diagnosis"] 中
        return state
    
    def analyze(
        self,
        scan_id: str,
        patient_id: str,
        study_date: str,
        findings: List[NoduleFinding],
        detected_organs: List[str],
        body_part: str = "胸部"
    ) -> Dict[str, Any]:
        """
        执行诊断分析
        
        Args:
            scan_id: 扫描 ID
            patient_id: 患者 ID
            study_date: 检查日期
            findings: 发现列表
            detected_organs: 检测到的器官
            body_part: 检查部位
            
        Returns:
            诊断结果
        """
        initial_state = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "study_date": study_date,
            "body_part": body_part,
            "findings": findings,
            "detected_organs": detected_organs,
            "diagnosis": "",
            "recommendations": ""
        }
        
        # 运行工作流
        result = self.workflow.invoke(initial_state)
        
        return {
            "scan_id": scan_id,
            "diagnosis": result.get("diagnosis", ""),
            "findings_count": len(findings),
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_simple(
        self,
        findings_data: Dict[str, Any]
    ) -> str:
        """
        简单诊断分析 (不使用工作流)
        
        Args:
            findings_data: 发现数据字典
            
        Returns:
            诊断文本
        """
        findings_json = json.dumps(findings_data, indent=2, ensure_ascii=False)
        
        user_prompt = f"""请分析以下 CT 影像数据：

```json
{findings_json}
```

请生成简洁的诊断印象。
"""
        
        messages = [
            SystemMessage(content=DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"诊断失败: {e}")
            return f"诊断生成失败: {str(e)}"


