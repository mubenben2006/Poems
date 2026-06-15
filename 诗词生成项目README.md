七言绝句条件生成 —— 操作说明
===================================

环境要求: Python 3.10+, PyTorch 2.x, CUDA (可选)


一、项目结构
-----------------------------------
诗词生成/
├── config.py           参数配置（改 MODEL/MODE 即可切换）
├── run.py              一键运行入口
├── generate.py         生成 + 评测（独立运行）
├── train.py            训练脚本
├── eval.py             评测脚本
├── make_report.py      生成 Word 报告
├── models/             LSTM / Transformer / RhymeToneTransformer
├── utils/              Tokenizer / Sampler / 数据加载 / 评测指标
├── data/               数据集 + 词表 (138,500 首七言绝句)
├── outputs/
│   ├── checkpoints/    训练好的模型权重 (.pt)
│   └── samples/        生成的诗 (.txt) + 指标报告 (.json)
└── requirements.txt    依赖列表


二、快速运行
-----------------------------------

1. 安装依赖:
   pip install -r requirements.txt

2. 生成诗歌 (用已有的 checkpoint):
   python generate.py

   → 加载 LSTM / Transformer / RhymeTone 三个模型
   → 各生成 首句续写 25 首 + 藏头诗 25 首
   → 打印指标对比表 (押韵率/平仄/语义相关度/重复度)
   → 全部诗歌存入 outputs/samples/*.txt


三、单独测试某个模型
-----------------------------------
修改 config.py 第 20 行:
   MODEL = "transformer"   # 可选: lstm / transformer / rhyme_transformer

然后运行:
   python run.py

或命令行直接:
   python generate.py -m lstm -i "春风送暖入屠苏"     首句续写
   python generate.py -m transformer --acrostic "春暖花开"  藏头诗


四、Checkpoint 位置
-----------------------------------
outputs/checkpoints/
├── lstm_best.pt                     LSTM (Val PPL 93.5, ~100MB)
├── transformer_best.pt              Transformer (Val PPL 59.9, ~112MB)
├── rhyme_transformer_best.pt        RhymeTone (FT Loss 6.59, ~113MB)
├── *.png                            训练曲线图
└── *.json                           训练历史


五、完整训练流程 (如需重训)
-----------------------------------
1. 修改 config.py: MODE = "full"
2. python run.py
   → 自动下载数据 → 训练 LSTM (~35min) → 训练 Transformer (~50min)
   → 韵律微调 (~24min) → 生成 150 首诗 → 打印评测报告 → 逐样本分析
   → 采样策略对比 → 保存全部诗歌 + JSON 报告


六、关键参数速查
-----------------------------------
模型:    LSTM(3层/h=512)  Transformer(5层/d=384)  RhymeTone(12.8M)
采样:    T=0.8, top_k=40, repetition_penalty=1.25, no_repeat_ngram=3
训练:    batch=64, lr=5e-4, Adam, StepLR, early_stop(patience=8)
数据:    138,500首七言绝句, 训练/验证=9:1, 字符级词表~5,000字


七、评测指标
-----------------------------------
PPL (困惑度)        越低越好, 验证集逐token交叉熵
格式合规率          4句×7字完全正确的比例
押韵率              2/4句末字同韵部且平声 (去韵头后比较)
平仄匹配率          与4种标准格式的最佳匹配比例
语义相关度          IDF加权嵌入余弦相似度 (0~1+)
Bigram/Trigram重复   自重复检测 (越低多样性越好)
行间最大重叠        LCS占比 (检测行间抄袭)
