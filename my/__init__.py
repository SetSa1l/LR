"""
Linear Region Analysis Toolkit
用于分析 ReLU 神经网络的线性区域性质

主要组件：
- LinearRegionAnalyzer: 高层整合分析器（推荐使用）
- ModelWrapper: 模型封装器
- LinearRegionTraverser: 区域遍历器
- DecisionBoundaryDirectionFinder: 方向查找器
- RegionPropertyAnalyzer: 区域性质分析器
"""

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

__version__ = "0.1.0"
__all__ = [
    # 高层 API（推荐）
    "LinearRegionAnalyzer",
    "AnalysisResult",
    "BatchAnalysisResult",
    # 底层组件
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