import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import tempfile
from io import BytesIO
from google import genai
from google.genai import types
import time
import traceback
import pathlib

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
        # 创建图像保存目录
        self.images_dir = os.path.join(self.node_dir, "generated_images")
        if not os.path.exists(self.images_dir):
            try:
                os.makedirs(self.images_dir)
                self.log(f"创建图像保存目录: {self.images_dir}")
            except Exception as e:
                self.log(f"创建图像目录失败: {e}")
                self.images_dir = self.node_dir  # 如果创建失败，使用节点目录
        
        # 检查google-genai版本
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"当前google-genai版本: {genai_version}")
            
            # 检查版本是否满足最低要求
            from packaging import version
            if version.parse(genai_version) < version.parse('1.5.0'):  
                self.log("警告: google-genai版本过低，建议升级到最新版本")
                self.log("建议执行: pip install -q -U google-genai")
        except Exception as e:
            self.log(f"无法检查google-genai版本: {e}")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到文件中
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""
    
    def generate_empty_image(self, width, height):
        """生成标准格式的空白RGB图像张量 - 确保ComfyUI兼容格式 [B,H,W,C]"""
        # 创建一个符合ComfyUI标准的图像张量
        # ComfyUI期望 [batch, height, width, channels] 格式!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None
                
            self.log(f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}")
            
            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None
                
            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()
                
            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None
    
    def save_tensor_as_image(self, image_tensor, file_path):
        """将图像张量保存为文件"""
        try:
            # 转换为numpy数组
            if torch.is_tensor(image_tensor):
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]  # 获取批次中的第一张图像
                
                # [C, H, W] -> [H, W, C]
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor
            
            # 缩放到0-255
            image_np = (image_np * 255).astype(np.uint8)
            
            # 创建PIL图像
            pil_image = Image.fromarray(image_np)
            
            # 保存到文件
            pil_image.save(file_path, format="PNG")
            self.log(f"已保存图像到: {file_path}")
            return True
        except Exception as e:
            self.log(f"图像保存错误: {str(e)}")
            return False
    
    def process_image_data(self, image_data, width, height):
        """处理API返回的图像数据，返回ComfyUI格式的图像张量 [B,H,W,C]"""
        try:
            # 打印图像数据类型和大小以便调试
            self.log(f"图像数据类型: {type(image_data)}")
            self.log(f"图像数据长度: {len(image_data) if hasattr(image_data, '__len__') else '未知'}")
            
            # 尝试直接转换为PIL图像
            try:
                pil_image = Image.open(BytesIO(image_data))
                self.log(f"成功打开图像, 尺寸: {pil_image.width}x{pil_image.height}, 模式: {pil_image.mode}")
            except Exception as e:
                self.log(f"无法直接打开图像数据: {e}")
                
                # 尝试其他方式解析，例如base64解码
                try:
                    # 检查是否是base64编码的字符串
                    if isinstance(image_data, str):
                        # 尝试移除base64前缀
                        if "base64," in image_data:
                            image_data = image_data.split("base64,")[1]
                        decoded_data = base64.b64decode(image_data)
                        pil_image = Image.open(BytesIO(decoded_data))
                    else:
                        # 如果是向量或其他格式，生成一个占位图像
                        self.log("无法解析图像数据，创建一个空白图像")
                        return self.generate_empty_image(width, height)
                except Exception as e2:
                    self.log(f"备用解析方法也失败: {e2}")
                    return self.generate_empty_image(width, height)
            
            # 确保图像是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                self.log(f"图像已转换为RGB模式")
            
            # 调整图像大小
            if pil_image.width != width or pil_image.height != height:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                self.log(f"图像已调整为目标尺寸: {width}x{height}")
            
            # 关键修复: 使用ComfyUI兼容的格式 [batch, height, width, channels]
            # 而不是PyTorch标准的 [batch, channels, height, width]
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self.log(f"生成的图像张量格式: 形状={img_tensor.shape}, 类型={img_tensor.dtype}")
            return (img_tensor,)
            
        except Exception as e:
            self.log(f"处理图像数据时出错: {e}")
            traceback.print_exc()
            return self.generate_empty_image(width, height)
    
    def generate_image(self, prompt, api_key, model, width, height, temperature, seed=66666666, image=None):
        """生成图像 - 使用简化的API密钥管理"""
        temp_img_path = None
        response_text = ""
        
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(width, height), full_text)
            
            # 创建客户端实例
            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                self.log(f"使用指定的种子值: {seed}")
            
            # 构建包含尺寸信息的提示
            aspect_ratio = width / height
            if aspect_ratio > 1:
                orientation = "landscape (horizontal)"
            elif aspect_ratio < 1:
                orientation = "portrait (vertical)"
            else:
                orientation = "square"

            simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels. Ensure the composition fits properly within these dimensions without stretching or distortion."
            
            # 配置生成参数，使用用户指定的温度值
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}")
            
            # 处理参考图像
            contents = []
            has_reference = False
            
            if image is not None:
                try:
                    # 确保图像格式正确
                    if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] 格式
                        # 获取第一帧图像
                        input_image = image[0].cpu().numpy()
                        
                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        # 保存为临时文件
                        temp_img_path = os.path.join(self.images_dir, f"reference_{int(time.time())}.png")
                        pil_image.save(temp_img_path)
                        
                        self.log(f"参考图像处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 读取图像数据
                        with open(temp_img_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # 添加图像部分和文本部分
                        img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                        txt_part = {"text": simple_prompt + " Use this reference image as style guidance."}
                        
                        # 组合内容(图像在前，文本在后)
                        contents = [img_part, txt_part]
                        has_reference = True
                        self.log("参考图像已添加到请求中")
                    else:
                        self.log(f"参考图像格式不正确: {image.shape}")
                        contents = simple_prompt
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    contents = simple_prompt
            else:
                # 没有参考图像，只使用文本
                contents = simple_prompt
            
            # 打印请求信息
            self.log(f"请求Gemini API生成图像，种子值: {seed}, 包含参考图像: {has_reference}")
            
            # 调用API
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=contents,
                config=gen_config
            )
            
            # 响应处理
            self.log("API响应接收成功，正在处理...")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                self.log("API响应中没有candidates")
                # 合并日志和返回值
                full_text = "\n".join(self.log_messages) + "\n\nAPI返回了空响应"
                return (self.generate_empty_image(width, height), full_text)
            
            # 检查响应中是否有图像
            image_found = False
            
            # 遍历响应部分
            for part in response.candidates[0].content.parts:
                # 检查是否为文本部分
                if hasattr(part, 'text') and part.text is not None:
                    text_content = part.text
                    response_text += text_content
                    self.log(f"API返回文本: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
                # 检查是否为图像部分
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    self.log("API返回数据解析处理")
                    try:
                        # 获取图像数据
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                        self.log(f"图像数据类型: {type(image_data)}, MIME类型: {mime_type}, 数据长度: {len(image_data) if image_data else 0}")
                        
                        # 如果数据为空则跳过
                        if not image_data or len(image_data) < 100:
                            self.log("警告: 图像数据为空或太小")
                            continue
                        
                        # 直接保存为节点目录下的文件
                        timestamp = int(time.time())
                        filename = f"gemini_image_{timestamp}.png"
                        img_file = os.path.join(self.images_dir, filename)
                        with open(img_file, "wb") as f:
                            f.write(image_data)
                        self.log(f"已保存图像数据到文件: {img_file}")
                        
                        # 使用请求的尺寸创建一个新的空白图像
                        pil_image = Image.new('RGB', (width, height), color=(128, 128, 128))
                        
                        # 尝试打开已保存的图像
                        try:
                            saved_image = Image.open(img_file)
                            self.log(f"成功打开已保存的图像，尺寸: {saved_image.width}x{saved_image.height}")
                            
                            # 确保是RGB模式
                            if saved_image.mode != 'RGB':
                                saved_image = saved_image.convert('RGB')
                            
                            # 调整尺寸后粘贴到空白图像上
                            if saved_image.width != width or saved_image.height != height:
                                saved_image = saved_image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            pil_image = saved_image
                        except Exception as e:
                            self.log(f"无法打开保存的图像: {str(e)}")
                            # 使用备选方法：保存原始数据到备选文件
                            try:
                                alt_filename = f"gemini_alt_{timestamp}.bin"
                                alt_file = os.path.join(self.images_dir, alt_filename)
                                pathlib.Path(alt_file).write_bytes(image_data)
                                self.log(f"已保存原始数据到备选文件: {alt_file}")
                                self.log("请尝试手动打开此文件检查内容")
                            except Exception as e2:
                                self.log(f"备选方法也失败: {str(e2)}")
                        
                        # 转换为ComfyUI格式
                        img_array = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        
                        self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                        image_found = True
                        
                        # 合并日志和API返回文本
                        full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                        return (img_tensor, full_text)
                    except Exception as e:
                        self.log(f"图像处理错误: {e}")
                        traceback.print_exc()  # 添加详细的错误追踪信息
            
            # 没有找到图像数据，但可能有文本
            if not image_found:
                self.log("API响应中未找到图像数据，仅返回文本")
                if not response_text:
                    response_text = "API未返回任何图像或文本"
            
            # 合并日志和API返回文本
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
            return (self.generate_empty_image(width, height), full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")
            
            # 合并日志和错误信息
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + error_message
            return (self.generate_empty_image(width, height), full_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "Gemini 2.0 image"
} 