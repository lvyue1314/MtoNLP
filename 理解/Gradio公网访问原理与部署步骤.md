# Gradio 公网访问原理与部署步骤

## 一、网络架构

云平台只有内网 IP（10.232.10.60），无法直接从公网访问。Gradio 的 `--share` 参数通过 **frp（Fast Reverse Proxy）隧道** 解决这个问题：

```
┌──────────────────────────────┐         ┌──────────────────────┐         ┌─────────────────────┐
│  云平台 (内网, 无公网IP)      │         │  HuggingFace 中转    │         │  你的电脑 (公网)      │
│                              │  frp    │  服务器              │ HTTPS   │                      │
│  Gradio :7860  ←── frpc ──────→ 隧道 ──→  xxx.gradio.live ────→ 浏览器 │                      │
│                              │         │                      │         │                      │
└──────────────────────────────┘         └──────────────────────┘         └─────────────────────┘

1. Gradio 启动 --share 参数
2. frpc 客户端连接 HuggingFace 的 frp 服务器，建立双向隧道
3. HuggingFace 分配一个临时域名 xxx.gradio.live
4. 你本地浏览器访问这个域名 → 流量通过隧道 → 到达云平台的 Gradio :7860
```

## 二、frpc 下载问题

云平台无外网，无法直接 `wget` HuggingFace CDN。解决方案：

1. 本地下载 frpc 二进制文件 → 放入 Git 仓库 `gradio/frpc_linux_amd64_v0.3`
2. 云平台 `git pull` 获取
3. 复制到 Gradio 期望的路径：`~/.cache/huggingface/gradio/frpc/`

## 三、完整部署步骤

### 在云平台首次部署

```bash
# 1. 克隆项目
cd /workspace/repo/src/fine-tune/models/gemma4
git clone <repo-url> ./MtoNLP
cd MtoNLP

# 2. 赋予权限
chmod +x *.sh

# 3. 检查/安装依赖（云平台预装大部分）
./install_deps.sh

# 4. 下载模型
./install_models.sh

# 5. 升级 vLLM（使 transformers 支持 Gemma 4 架构）
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

# 6. 复制 frpc 到 Gradio 缓存目录
mkdir -p /root/.cache/huggingface/gradio/frpc
cp gradio/frpc_linux_amd64_v0.3 /root/.cache/huggingface/gradio/frpc/
chmod +x /root/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

# 7. 升级 Gradio（修复 huggingface_hub 兼容性）
uv pip install "gradio>=6.0" --no-cache

# 8. 启动 vLLM 服务（终端 1）
vllm serve /models/google/gemma-4-E4B-it/ \
    --served-model-name gemma-4-E4B-it \
    --port 8000 --max-model-len 4096

# 9. 启动 Gradio（终端 2）
python src/gradio_app.py --share
```

### 后续使用（已部署过）

```bash
git pull origin main

# 启动服务（任选一个）
vllm serve /models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it --port 8000 --max-model-len 4096

# 启动 Gradio
python src/gradio_app.py --share
```

## 四、Gradio 在项目中的角色

```
用户上传图表 → Gradio 界面 → 调用 batch_inference API → vLLM 服务 → 返回答案
                    ↑                                        ↓
              --share 生成公网链接                     Gemma 4 / LLaVA
              你的浏览器访问                             GPU 推理
```

## 五、常见问题

| 问题 | 解决 |
|------|------|
| `--share` 报 `Missing file: frpc_linux_amd64_v0.3` | 复制仓库中的 `gradio/frpc_linux_amd64_v0.3` 到 `~/.cache/huggingface/gradio/frpc/` |
| 公网链接打不开 | 确认云平台 Gradio 进程未退出；链接有效期 1 周 |
| 推理返回全是错误 | 确认 vLLM 服务在运行且加了 `--served-model-name` |
| 切换模型需重启 Gradio | 是——Gradio 启动时建立连接，换服务后重开 `python src/gradio_app.py --share` |
| Gradio 6.0 报 `validate_outputs` 错误 | 已修复：返回 6 元组而非 dict |
