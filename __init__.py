# 导入所有节点（原节点+工具节点）
from .nodes import Qwen3_VQA  # 只保留实际存在的类
from .util_nodes import ImageLoader, VideoLoader, VideoLoaderPath
from .path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"

# 节点注册映射（移除不存在的uncensored节点）
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,  # 仅保留存在的原节点
    "ImageLoader": ImageLoader,
    "VideoLoader": VideoLoader,
    "VideoLoaderPath": VideoLoaderPath,
    "MultiplePathsInput": MultiplePathsInput,
}

# 节点显示名称（同步移除对应条目）
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",
    "ImageLoader": "Load Image Advanced",
    "VideoLoader": "Load Video Advanced",
    "VideoLoaderPath": "Load Video Advanced (Path)",
    "MultiplePathsInput": "Multiple Paths Input",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]