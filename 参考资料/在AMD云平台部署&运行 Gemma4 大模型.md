# 在AMD云平台部署\&运行 Gemma4 大模型

# 无序记录

> 这里大多参考了下述博客，以便后续理解：
> 
> [【Day1\-2】15分钟部署\&运行 Gemma4 大模型，撰写学习笔记](https://ailc.datawhale.cn/hall/group/100000144/task/100000036)
> 
> 

## 确认云环境是否可用，并检查目录

```Bash
# 查看目前GPU的利用率
amd-smi
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWZmYzZhY2ZkYTZkODUwYjI0Mjg4MDY3NzkyNzVmYWJfM2E3Mjc1NmU2YTE1ODZhZTMzMDhiYmI5ZGIwMzZiMDdfSUQ6NzY2MzgyMjk0MTg2MzQzMTEzNl8xNzg0MzgyNDU1OjE3ODQ0Njg4NTVfVjM)

> - Mem\-Uti（显存利用率）：显示显存带宽被占用的百分比（示例中为 0%），而非容量占用。它反映显存数据传输的繁忙程度，数值高说明GPU在频繁读写显存数据。
> 
> - Temp（温度）：GPU核心的当前温度（示例中为 32°C），是判断散热是否正常的关键指标。
> 
> - GFX\-Uti（图形引擎利用率）：GPU核心计算单元的忙碌程度（示例中为 20\.0%），相当于“GPU占用率”。数值高表示GPU正在密集计算（如渲染、AI推理）。
> 
> - Fan（风扇转速）：显卡风扇的转速百分比（示例中为 0%），表示风扇当前未旋转（通常因温度低，处于被动散热或停转节能状态）。
> 
> - GTT\_MEM（GTT 内存占用）
> 
>     - 全称：Graphics Translation Table（图形转换表）内存。
> 
>     - 含义：这是系统主内存（RAM）中划给 GPU 使用的一部分空间，用于存储暂时不放在显存中的数据，或者作为显存和系统内存之间的“桥梁”（类似交换区）。
> 
> - VRAM\_MEM（显存占用）
> 
>     - 全称：Video RAM（视频随机存取存储器），即显卡自带的物理显存。
> 
>     - 含义：这个进程实际正在使用的显存量，是硬性占用。如果数值过高（接近总显存上限），就说明该进程是“显存大户”。
> 
> - MEM\_USAGE（总内存使用量）
> 
>     - 含义：这个字段通常是 GTT\_MEM \+ VRAM\_MEM 的总和，即该进程占用的所有 GPU 相关内存（显存 \+ 系统内存共享部分）。
> 
> - CU %（计算单元利用率）
> 
>     - 全称：Compute Unit（计算单元）利用率。
> 
>     - 含义：这是更精细的 GPU 核心占用指标，表示该进程占用了多少个计算单元的执行资源。和顶部的 `GFX-Uti`（全局 GPU 占用率）不同，`CU %` 是按进程细分的。
> 
>     - 示例中为 N/A：通常是因为进程没有实际运行计算任务，或者驱动无法获取该进程的细分数据（可能需要 root 或更高权限）。
> 
> ---
> 
> 总结：
> 
> 这两个进程（PID 3720 和 68840）虽然被系统识别为与 GPU 相关，但既没占用显存（VRAM=0），也没使用系统内存做交换（GTT=0），更没有执行任何计算（CU %=N/A），所以它们相当于“挂名”进程，对 GPU 资源没有任何影响。
> 
> 

---

```Bash
# **确认 PyTorch 能识别 AMD GPU**
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=N2E4Mzc1OWVkN2JjM2NhMWRlNTUxZmFmYzk4N2M5ZjZfMmFlOGIyZGIyNjhiODI4Zjk3MzQ5YWRlNTMzZjYzMDJfSUQ6NzY2MzgyODA3NDIzMTkwOTY1N18xNzg0MzgyNDU1OjE3ODQ0Njg4NTVfVjM)

## 下载模型

```Bash
# 将 pip 的默认软件源（PyPI）永久更换为腾讯云镜像源，目的是为了加快 Python 包的下载和安装速度（尤其是在国内网络环境下）
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/

# 验证配置是否生效
pip config list

# **安装魔搭** **ModelScope**
pip install modelscope

# 下载 Gemma4 模型到当前目录
modelscope download --model google/gemma-4-E4B-it --cache_dir "./models"

# **确认 Gemma4 模型模型文件完整下载成功**
ls -lh ./models/google/gemma-4-E4B-it/
## -l : 长格式显示（long format），显示详细信息
## -h : 人类可读（human-readable），文件大小用 KB/MB/GB 显示
```

## 启动 vLLM 服务

> LLM 是一个本地高效推理大模型的项目，这里我们使用vLLM来测试刚才下载的模型能否正常使用。
> 
> 在使用 vLLM 前，需更新云环境中的 vLLM 版本才能运行 Gemma4 模型。 **操作步骤如下：**
> 
> ```Bash
> root@u-4572-5d640603:/workspace/template-repos/template-257/repo/src/fine-tune/models/gemma4# cat /opt/rocm/.info/version
> 7.2.1
> ```
> 
> 

```Bash
uv pip uninstall torchvision torchaudio # 经测试，在该云环境中，需卸载重新安装这个库才能正常使用
uv pip install 'vllm==0.23.0+rocm723' torchvision torchaudio 'fastapi[standard]==0.136.0' \
  --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/rocm/ \
  -U
```

'vllm==0\.23\.0\+rocm723'的解析：

```Bash
主版本号.次版本号.修订号[ - 预发布标识符][ + 本地版本标识符]
```

> “前面的 `0.23.0` 是官方版本号，从 `+` 开始的内容是本地构建信息，不影响版本大小的比较。”
> 
> 

---

```Bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it

# or如何显示显存不足，就减少上下文长度
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it --max-model-len 8192
```

> 注意：运行这个命令后，这个终端窗口就会 **被大模型服务“死死占满”** 。请 **保持运行，绝对不要关闭它** ，也不要按 `Ctrl+C` ，否则大模型服务就会立刻停止。
> 
> 

出现Application startup complete\.——就代表启动成功

## 打开新终端进行对话测试（即模型推理能力测试）

```Bash
vllm chat --url http://localhost:8000/v1 --model gemma-4-E4B-it

# 输入文本进行测试:
你是谁，你能做什么
```

## 退出vllm对话与服务

> 两个终端都用ctrl\+C
> 
> 

## 在毁掉实例之前

1. **📁 左侧文件栏（默认工作区**  `/workspace`  **）**

    - 你在网页左侧看到的文件、平时折腾的代码都在这里。 它会自动存盘，但 **无法跨机器移动** 。

    - 换句话说，如果你下次登录被分配到另一台服务器，这里的文件可能暂时看不到，但它们其实还在原服务器上。

2. **🌐 系统底层网络同步盘（绝对路径**  `/network-workspace`  **）**

    - 这里存的文件 **不会出现在左侧文件栏** ，必须在代码里用绝对路径访问。 每人有 20GB 空间，支持 **跨机器同步** 。

    - **建议：** 重要文件或者需要跨机器访问的项目都放这里。

3. **🚨 运行环境与安装的库**

    - 你在终端里用 `pip` 安装的库、配置的环境，不在前两个存储区域内。  一旦你 **断开连接或点 Destroy** ，这些环境会瞬间被清空，下次必须重新配置。

    - **建议：** 用 `requirements.txt` 或 Dockerfile 保存环境配置，随时可以重建。

> - 快照（Snapshot）：是云平台提供的磁盘级备份，备份的是整个系统盘的数据和状态。
> 
> - 环境配置文件（requirements\.txt / Dockerfile）：是应用级备份，只记录软件的依赖和环境定义，不包含数据。
> 
> 

# 参考资料

https://ailc\.datawhale\.cn/hall/group/100000144/task/100000036



