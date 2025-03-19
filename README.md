# ComfyUI Gemini API

用于在comfyUI中调用Google Gemini API。

**原仓库: [ComfyUI-Gemini-API](https://github.com/CY-CHENYUE/ComfyUI-Gemini-API)，感谢作者的分享**

API不稳定，如果遇到生成空白的情况，请重试几次。
## 小小改动
- 新增keep_proportion参数，如果为True,生成图像比例与输入图像一致，但是最长边会缩放到1024。如果为False，生成图像比例与输入的width、height一样。
- 新增instruction参数，暂时没有用到，出于习惯，不是必填。

## 安装说明

### 手动安装

1. 将此存储库克隆到ComfyUI的`custom_nodes`目录：
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/singlinhaha/ComfyUI-Gemini-API-EXP.git
   ```

2. 安装所需依赖：

   如果你使用ComfyUI便携版
   ```
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```

   如果你使用自己的Python环境
   ```
   path\to\your\python.exe -m pip install -r requirements.txt
   ```

安装完成后重启ComfyUI

## 节点说明

### Gemini 2.0 image

通过Gemini API生成图像的节点。

**输入参数：**
- **instruction** ：文本指令，暂时没有用到，不是必填
- **prompt** (必填)：描述你想要生成的图像的文本提示词
- **api_key** (必填)：你的Google Gemini API密钥（首次设置后会自动保存）
- **model**：模型选择
- **keep_proportion**：是否保持图像比例
- **width**：生成图像的宽度（512-2048像素）
- **height**：生成图像的高度（512-2048像素）
- **temperature**：控制生成多样性的参数（0.0-2.0）
- **seed** (可选)：随机种子，指定值可重现结果
- **image** (可选)：参考图像输入，用于风格引导

**输出：**
- **image**：生成的图像，可以连接到ComfyUI的其他节点
- **API Respond**：包含处理日志和API返回的文本信息

**使用场景：**
- 创建独特的概念艺术
- 基于文本描述生成图像
- 使用参考图像创建风格一致的新图像
- 基于图像的编辑操作

## 获取API密钥

1. 访问[Google AI Studio](https://aistudio.google.com/apikey?hl=zh-cn)
2. 创建一个账户或登录
3. 在"API Keys"部分创建一个新的API密钥
4. 复制API密钥并粘贴到节点的api_key参数中（只需首次输入，之后会自动保存）

## 温度参数说明

- 温度值范围：0.0到2.0
- 较低的温度（接近0）：生成更确定性、可预测的结果
- 较高的温度（接近2）：生成更多样化、创造性的结果
- 默认值1.0：平衡确定性和创造性

## 注意事项

- API可能有使用限制或费用，请查阅Google的官方文档
- 图像生成质量和速度取决于Google的服务器状态和您的网络连接
- 参考图像功能会将您的图像提供给Google服务，请注意隐私影响
- 首次使用时需要输入API密钥，之后会自动存储在节点目录中的gemini_api_key.txt文件中
- 关于图像方向Gemini API 会根据选择的方向（横屏、竖屏或方形）生成适合的图像（但是模型并不一定可以按照要求生成）。
