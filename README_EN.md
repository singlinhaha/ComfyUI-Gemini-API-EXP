# ComfyUI Gemini API

[中文](README.md) | English

A custom node for ComfyUI to integrate Google Gemini API.

## Installation

### Method 1: Manual Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/CY-CHENYUE/ComfyUI-Gemini-API
   ```

2. Install required dependencies:

   If you're using ComfyUI portable version:
   ```
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```

   If you're using your own Python environment:
   ```
   path\to\your\python.exe -m pip install -r requirements.txt
   ```

### Method 2: Install via ComfyUI Manager

   1. Install and open ComfyUI Manager in ComfyUI
   2. Search for "Gemini API"
   3. Click the install button

Restart ComfyUI after installation

## Node Description

### Gemini 2.0 image

![alt text](workflow/Gemini-API.png)

A node that generates images using the Gemini API.

**Input Parameters:**
- **prompt** (required): Text prompt describing the image you want to generate
- **api_key** (required): Your Google Gemini API key (automatically saved after first setup)
- **model**: Model selection
- **width**: Width of the generated image (512-2048 pixels)
- **height**: Height of the generated image (512-2048 pixels)
- **temperature**: Parameter controlling generation diversity (0.0-2.0)
- **seed** (optional): Random seed for reproducible results
- **image** (optional): Reference image input for style guidance

**Outputs:**
- **image**: Generated image that can be connected to other ComfyUI nodes
- **API Respond**: Text information containing processing logs and API response

**Use Cases:**
- Creating unique concept art
- Generating images from text descriptions
- Creating style-consistent new images using reference images
- Image editing based on existing images

## Getting API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an account or sign in
3. Create a new API key in the "API Keys" section
4. Copy the API key and paste it into the node's api_key parameter (only needed for first use, will be saved automatically)

## Temperature Parameter Guide

- Temperature range: 0.0 to 2.0
- Lower temperature (near 0): More deterministic, predictable results
- Higher temperature (near 2): More diverse, creative results
- Default value 1.0: Balances determinism and creativity

## Important Notes

- API may have usage limits or costs, please refer to Google's official documentation
- Image generation quality and speed depend on Google's server status and your network connection
- Reference image feature will send your images to Google services, please be aware of privacy implications
- API key needs to be entered only once, it will be stored in gemini_api_key.txt in the node directory

## Contact Me

- X (Twitter): [@cychenyue](https://x.com/cychenyue)
- TikTok: [@cychenyue](https://www.tiktok.com/@cychenyue)
- YouTube: [@CY-CHENYUE](https://www.youtube.com/@CY-CHENYUE)
- BiliBili: [@CY-CHENYUE](https://space.bilibili.com/402808950)
- Xiaohongshu: [@CY-CHENYUE](https://www.xiaohongshu.com/user/profile/6360e61f000000001f01bda0) 