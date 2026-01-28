"""
对比 Agent - 纵向时序分析
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.core.logging import logger
from app.schemas.analysis import ComparisonResult, RECISTResponse
from .prompts import COMPARATIVE_SYSTEM_PROMPT, COMPARATIVE_USER_TEMPLATE


class ComparativeAgentState(dict):
    """对比 Agent 状态"""
    patient_id: str
    baseline_scan_id: str
    baseline_date: str
    followup_scan_id: str
    followup_date: str
    days_between: int
    comparisons: List[ComparisonResult]
    heatmap_path: str
    analysis: str
    recist_evaluation: str


class ComparativeAgent:
    """对比专家 Agent"""
    
    def __init__(self, llm: ChatOpenAI = None):
        """
        初始化对比 Agent
        
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
        workflow = StateGraph(ComparativeAgentState)
        
        # 添加节点
        workflow.add_node("prepare_comparison", self._prepare_comparison)
        workflow.add_node("evaluate_recist", self._evaluate_recist)
        workflow.add_node("generate_analysis", self._generate_analysis)
        workflow.add_node("format_report", self._format_report)
        
        # 添加边
        workflow.set_entry_point("prepare_comparison")
        workflow.add_edge("prepare_comparison", "evaluate_recist")
        workflow.add_edge("evaluate_recist", "generate_analysis")
        workflow.add_edge("generate_analysis", "format_report")
        workflow.add_edge("format_report", END)
        
        return workflow.compile()
    
    def _prepare_comparison(self, state: ComparativeAgentState) -> ComparativeAgentState:
        """准备对比数据"""
        logger.info(f"准备对比数据: {state.get('baseline_scan_id')} vs {state.get('followup_scan_id')}")
        return state
    
    def _evaluate_recist(self, state: ComparativeAgentState) -> ComparativeAgentState:
        """RECIST 评估"""
        logger.info("执行 RECIST 评估...")
        
        comparisons = state.get("comparisons", [])
        recist_results = []
        
        for comp in comparisons:
            if isinstance(comp, dict):
                diameter_change = comp.get("diameter_change_percent", 0)
            else:
                diameter_change = comp.diameter_change_percent
            
            if diameter_change <= -30:
                recist = "PR (部分缓解)"
            elif diameter_change >= 20:
                recist = "PD (疾病进展)"
            else:
                recist = "SD (疾病稳定)"
            
            recist_results.append(recist)
        
        state["recist_evaluation"] = ", ".join(recist_results) if recist_results else "无法评估"
        return state
    
    def _generate_analysis(self, state: ComparativeAgentState) -> ComparativeAgentState:
        """生成对比分析"""
        logger.info("生成对比分析...")
        
        # 构建对比数据 JSON
        comparisons = state.get("comparisons", [])
        comparison_data = []
        
        for comp in comparisons:
            if isinstance(comp, dict):
                comparison_data.append(comp)
            elif hasattr(comp, 'model_dump'):
                comparison_data.append(comp.model_dump())
            else:
                comparison_data.append(str(comp))
        
        comparison_json = json.dumps(comparison_data, indent=2, ensure_ascii=False, default=str)
        
        # 构建 prompt
        user_prompt = COMPARATIVE_USER_TEMPLATE.format(
            patient_id=state.get("patient_id", "Unknown"),
            baseline_date=state.get("baseline_date", "Unknown"),
            baseline_scan_id=state.get("baseline_scan_id", "Unknown"),
            followup_date=state.get("followup_date", "Unknown"),
            followup_scan_id=state.get("followup_scan_id", "Unknown"),
            days_between=state.get("days_between", 0),
            comparison_json=comparison_json,
            heatmap_path=state.get("heatmap_path", "未生成")
        )
        
        # 调用 LLM
        messages = [
            SystemMessage(content=COMPARATIVE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["analysis"] = response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            state["analysis"] = f"分析生成失败: {str(e)}"
        
        return state
    
    def _format_report(self, state: ComparativeAgentState) -> ComparativeAgentState:
        """格式化报告"""
        logger.info("格式化对比报告...")
        return state
    
    def compare(
        self,
        patient_id: str,
        baseline_scan_id: str,
        baseline_date: str,
        followup_scan_id: str,
        followup_date: str,
        comparisons: List[Dict[str, Any]],
        heatmap_path: str = None
    ) -> Dict[str, Any]:
        """
        执行对比分析
        
        Args:
            patient_id: 患者 ID
            baseline_scan_id: 基线扫描 ID
            baseline_date: 基线日期
            followup_scan_id: 随访扫描 ID
            followup_date: 随访日期
            comparisons: 对比数据列表
            heatmap_path: 热力图路径
            
        Returns:
            对比分析结果
        """
        # 计算间隔天数
        try:
            from datetime import datetime
            d1 = datetime.fromisoformat(baseline_date.replace("Z", "+00:00"))
            d2 = datetime.fromisoformat(followup_date.replace("Z", "+00:00"))
            days_between = (d2 - d1).days
        except:
            days_between = 0
        
        initial_state = {
            "patient_id": patient_id,
            "baseline_scan_id": baseline_scan_id,
            "baseline_date": baseline_date,
            "followup_scan_id": followup_scan_id,
            "followup_date": followup_date,
            "days_between": days_between,
            "comparisons": comparisons,
            "heatmap_path": heatmap_path or "未生成",
            "analysis": "",
            "recist_evaluation": ""
        }
        
        # 运行工作流
        result = self.workflow.invoke(initial_state)
        
        return {
            "patient_id": patient_id,
            "baseline_scan_id": baseline_scan_id,
            "followup_scan_id": followup_scan_id,
            "days_between": days_between,
            "recist_evaluation": result.get("recist_evaluation", ""),
            "analysis": result.get("analysis", ""),
            "heatmap_path": heatmap_path,
            "generated_at": datetime.now().isoformat()
        }
    
    def compare_simple(
        self,
        comparison_data: Dict[str, Any]
    ) -> str:
        """
        简单对比分析 (不使用工作流)
        
        Args:
            comparison_data: 对比数据字典
            
        Returns:
            分析文本
        """
        comparison_json = json.dumps(comparison_data, indent=2, ensure_ascii=False, default=str)
        
        user_prompt = f"""请分析以下 CT 对比数据：

```json
{comparison_json}
```

请根据 RECIST 1.1 标准生成对比分析报告。
"""
        
        messages = [
            SystemMessage(content=COMPARATIVE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"对比分析失败: {e}")
            return f"分析生成失败: {str(e)}"


