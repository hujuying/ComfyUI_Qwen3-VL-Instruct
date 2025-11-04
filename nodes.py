import os
import torch
import gc  # 新增：导入垃圾回收模块
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import comfy.model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path


class Qwen3_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.current_model_id = None  # Track the current model id
        self.current_quantization = None  # Track the current quantization

    # 新增：模型资源清理方法
    def clear_model_resources(self):
        if self.model is not None or self.processor is not None:
            print(f"\n[Qwen3_VQA] Starting model unload...")
            
            # 记录卸载前的显存使用
            vram_before = 0
            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated(self.device) / 1024**2
                print(f"[Qwen3_VQA] VRAM allocated before unload: {vram_before:.2f} MB")

            # 将模型移至CPU（如果在GPU上）
            try:
                if self.model is not None:
                    self.model.to('cpu')
                print(f"[Qwen3_VQA] Model moved to CPU successfully")
            except Exception as e:
                print(f"[Qwen3_VQA] Warning: Could not move model to CPU: {e}")

            # 删除模型和处理器引用
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_id = None
            self.current_quantization = None
            print(f"[Qwen3_VQA] Model references deleted")

            # 强制垃圾回收
            for i in range(3):
                collected = gc.collect()
                print(f"[Qwen3_VQA] GC pass {i+1}: collected {collected} objects")

            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保缓存清理完成
                vram_after = torch.cuda.memory_allocated(self.device) / 1024**2
                print(f"[Qwen3_VQA] VRAM allocated after unload: {vram_after:.2f} MB")
                print(f"[Qwen3_VQA] VRAM freed: {vram_before - vram_after:.2f} MB")

            print(f"[Qwen3_VQA] Model unload complete\n")
        else:
            print(f"[Qwen3_VQA] No model resources to unload")

    @classmethod
    def INPUT_TYPES(s):
        # 保持不变...
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                    ],
                    {"default": "Qwen3-VL-4B-Instruct-FP8"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                ),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def inference(
        self,
        text,
        text2,
        model,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
        seed,
        quantization,
        source_path=None,
        image=None,
        attention="eager",
    ):
        # 组合提示词逻辑保持不变
        prompt_parts = [p.strip() for p in [text, text2] if p.strip()]
        combined_text = ", ".join(prompt_parts) if prompt_parts else ""

        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        # 模型下载逻辑保持不变
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        # 模型加载逻辑保持不变
        if (
            self.current_model_id != model_id
            or self.current_quantization != quantization
            or self.processor is None
            or self.model is None
        ):
            # 加载新模型前先清理旧模型
            self.clear_model_resources()
            
            self.current_model_id = model_id
            self.current_quantization = quantization
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )
            
            # 量化配置逻辑保持不变
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

        # 图像处理逻辑保持不变
        temp_path = None
        if image is not None:
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
            pil_image.save(temp_path)

        # 推理逻辑保持不变
        with torch.no_grad():
            if source_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [{"type": "text", "text": combined_text}],
                    },
                ]
            elif temp_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": combined_text},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": combined_text}],
                    }
                ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        # 修改：使用新的清理方法卸载模型
        if not keep_model_loaded:
            self.clear_model_resources()

        return (result,)