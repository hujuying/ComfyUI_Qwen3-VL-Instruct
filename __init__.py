# 导入所有节点（原节点+新节点+其他工具节点）
from .nodes import Qwen3_VQA, Qwen3_VQA_uncensored
from .util_nodes import ImageLoader, VideoLoader, VideoLoaderPath
from .path_nodes import MultiplePathsInput

WEB_DIRECTORY = "./web"

# 节点注册映射（保留所有原始节点，新增uncensored节点）
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,  # 原节点（双提示词版本）
    "Qwen3_VQA_uncensored": Qwen3_VQA_uncensored,  # 新节点（带内置提示词）
    "ImageLoader": ImageLoader,
    "VideoLoader": VideoLoader,
    "VideoLoaderPath": VideoLoaderPath,
    "MultiplePathsInput": MultiplePathsInput,
}

# 节点显示名称（带后缀区分）
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",  # 原节点显示名
    "Qwen3_VQA_uncensored": "Qwen3 VQA [uncensored]",  # 新节点显示名
    "ImageLoader": "Load Image Advanced",
    "VideoLoader": "Load Video Advanced",
    "VideoLoaderPath": "Load Video Advanced (Path)",
    "MultiplePathsInput": "Multiple Paths Input",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]