"""
Linear Region Analysis Toolkit
用于分析 ReLU 神经网络的线性区域性质

主要组件：
- SimpleLinearRegionAnalyzer: 精简的分析器（推荐使用）
  只计算4个核心指标：区域数量、平均梯度范数、边界处梯度变化、边界处损失变化
  
- LinearRegionAnalyzer: 完整的分析器（旧版本，保持兼容）
- ModelWrapper: 模型封装器
- LinearRegionTraverser: 区域遍历器
- DecisionBoundaryDirectionFinder: 方向查找器
- RegionPropertyAnalyzer: 区域性质分析器
"""

# 新的精简 API（推荐使用）
from .core.linear_region_analyzer import SimpleLinearRegionAnalyzer, SimpleAnalysisResult

# 旧 API（保持兼容）
from .core.model_wrapper import ModelWrapper, normalize_direction
from .core.region_traverser import LinearRegionTraverser, TraversalResult, BatchTraversalResult
from .core.direction_finder import DecisionBoundaryDirectionFinder, find_decision_boundary_direction
from .core.region_properties import (
    RegionPropertyAnalyzer,
    RegionProperties,
    AdjacentRegionDiff,
    TraversalProperties
)
from .core.analyzer import LinearRegionAnalyzer, AnalysisResult, BatchAnalysisResult

__version__ = "0.2.0"
__all__ = [
    # 新的精简 API（推荐）
    "SimpleLinearRegionAnalyzer",
    "SimpleAnalysisResult",
    # 旧 API（保持兼容）
    "LinearRegionAnalyzer",
    "AnalysisResult",
    "BatchAnalysisResult",
    "ModelWrapper", 
    "LinearRegionTraverser",
    "TraversalResult",
    "BatchTraversalResult",
    "normalize_direction",
    "DecisionBoundaryDirectionFinder",
    "find_decision_boundary_direction",
    "RegionPropertyAnalyzer",
    "RegionProperties",
    "AdjacentRegionDiff",
    "TraversalProperties"
]