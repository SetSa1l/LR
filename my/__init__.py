"""
Linear Region Analysis Toolkit
用于分析 ReLU 神经网络的线性区域性质

主要组件：
- FastLinearRegionAnalyzer: 极速分析器（推荐使用，10-20x 性能提升）
  使用解析方法计算边界距离，批量梯度计算
  
- SimpleLinearRegionAnalyzer: 精简的分析器
  只计算4个核心指标：区域数量、平均梯度范数、边界处梯度变化、边界处损失变化
  
- LinearRegionAnalyzer: 完整的分析器（旧版本，保持兼容）
- ModelWrapper: 模型封装器
- LinearRegionTraverser: 区域遍历器
- DecisionBoundaryDirectionFinder: 方向查找器
- RegionPropertyAnalyzer: 区域性质分析器
"""

# 极速版本 - 10-20x 性能提升（推荐使用）
from .core.fast_linear_region_analyzer import FastLinearRegionAnalyzer, FastAnalysisResult

# 精简 API
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

__version__ = "0.3.0"
__all__ = [
    # 极速版本 - 10-20x 性能提升（推荐）
    "FastLinearRegionAnalyzer",
    "FastAnalysisResult",
    # 精简 API
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