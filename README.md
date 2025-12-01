# AlphaGomoku：基于神经符号主义的五子棋 AI

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

这是 AlphaGomoku 的源码库。不同于纯粹依赖算力的 AlphaZero，本项目采用 **神经符号主义 (Neuro-Symbolic)** 架构。

核心逻辑：直觉（神经网络）+ 逻辑（VCF/VCT 求解器）。

通过 JIT 编译的求解器解决“算力贫困”问题，利用 ResNet 拟合人类棋谱解决“冷启动”问题。

代码没废话，结构如下。



## 1. 核心架构 (Core)



这部分是 AI 的大脑，动了这里就是动了根本。

- **`config.py`**
  - **作用**：全局超参数配置。
  - **逻辑**：包含显卡设置、神经网络结构（ResNet层数）、MCTS 模拟次数、VCF/VCT 搜索深度等。
  - **注意**：算力不够就把 `TRAINING_BATCH_SIZE` 调小。
- **`config4greedy.py`**
  - **作用**：为对战贪婪算法定制的简化配置
  - **逻辑**：关闭指数级耗时的VCT监测，一样能碾压贪婪算法
- **`model.py`**
  - **作用**：神经网络定义。
  - **逻辑**：标准的 AlphaZero 双头架构（策略头 Policy + 价值头 Value）。骨干网是 ResNet。
  - **输入**：15x15x4 的 Tensor（己方、对方、上一手、颜色）。
- **`mcts.py`**
  - **作用**：蒙特卡洛树搜索 (MCTS) 实现。
  - **逻辑**：实现了“短路 MCTS”机制。
    1. **层级 1**：规则硬编码（直接赢/必须防）。
    2. **层级 2**：调用 `game.py` 的 VCF/VCT 求解器进行必胜/必防检索。
    3. **层级 3**：神经网络介入，引导 MCTS 搜索。
  - **优势**：把算力用在刀刃上，必胜局面不需要神经网络瞎猜。
- **`game.py`** (替代 `gameOld.py`)
  - **作用**：游戏环境与规则引擎。
  - **逻辑**：**极其重要**。全部使用 `numba.jit` 静态编译加速。包含棋盘状态更新、落子合法性检查、以及核心的 **VCF (冲四)** 和 **VCT (连续活三)** 递归求解器。
  - **性能**：比纯 Python 快 60 倍以上。



## 2. 训练流水线 (Training Pipeline)



数据是养料，没有高质量数据，模型就是垃圾。

- **`parse.py`**
  - **作用**：ETL 工具。
  - **用法**：`python parse.py <psq文件夹> <输出.npz>`
  - **逻辑**：解析 Gomocup 的 `.psq` 棋谱文件，进行数据增强（旋转、翻转），转换成 `(state, policy, value)` 三元组并压缩存储。
- **`merge.py`**
  - **作用**：数据合并。
  - **逻辑**：把多个 `.npz` 合并成一个大的 Replay Buffer，方便一次性加载到内存。
- **`SLTrain.py`**
  - **作用**：监督学习 (Supervised Learning) 主程序。
  - **用法**：`python SLTrain.py dataset.npz`
  - **逻辑**：加载数据 -> 混合精度训练 (AMP) -> 最小化 KL 散度和均方误差。包含 TensorBoard 埋点。
- **`utils.py`**
  - **作用**：辅助函数。
  - **逻辑**：主要是数据增强的具体实现（D4 对称群变换）。
- **`checkNPZ.py`**
  - **作用**：数据探针。
  - **逻辑**：不加载数据进内存，仅读取 `.npz` 头部信息，快速检查数据量和维度。



## 3. 部署与交互 (Deployment & Interface)



模型练好了得拿出来遛遛。

- **`play_web.py`**
  - **作用**：Web 对战后端 (Flask)。
  - **用法**：运行后访问 `localhost:5000`。
  - **逻辑**：加载训练好的模型 (`sl_best.ckpt`)，实例化 `AIPilot` 类。处理前端传来的坐标，调用 MCTS 返回 AI 落子。记录对局日志。
- **`ABArena.py`**
  - **作用**：批量竞技场 (GPU 加速)。
  - **逻辑**：利用多进程 + 共享 GPU 推理，让 AI 自己打自己，或者打 Minimax。用于快速评估模型胜率，不用傻盯着屏幕看。
- **`test_greedy.py`**
  - **作用**：基准测试。
  - **逻辑**：让 SL 模型对战传统的贪心算法 (Greedy)。如果连这个都打不过，就别搞 RL 了，回去查数据。
- **`benchmark_vcf.py`**
  - **作用**：性能基准测试。
  - **逻辑**：对比 Python 递归和 Numba JIT 递归的速度差。写论文用的数据来源。



## 4. Web 前端与传统算法 (Legacy/Web)



位于 `gomoku_web/` 目录下。

- **`server.py`** & **`templates/index.html`** & **`static/gomoku.js`**:
  - 主要用于可视化交互。前端写死了逻辑，负责画棋盘、落子动画。
- **`ai/level1_greedy.py`** & **`ai/level2_minimax_alphabeta.py`**:
  - 这是用于 Web 版的纯传统算法实现（无神经网络）。用于作为 Baseline 对比，或者在没显卡的时候跑着玩。

------

**使用流程：**

1. **环境**：装好 PyTorch, Numba, Flask。没 GPU 别跑训练。
2. **数据**：用 `parse.py` 处理棋谱，跑`merge.py`合并多个`.npz`。
3. **训练**：跑 `SLTrain.py` 得到 `sl_best.ckpt`。
4. **运行**：跑 `play_web.py` 启动对战。

**文件说明完毕。代码自己看，逻辑都在上面。**