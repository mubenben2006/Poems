"""
项目配置 —— 七言绝句生成
修改此文件即可，无需在命令行传参。
运行: python run.py
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")
SAMPLE_DIR = os.path.join(BASE_DIR, "outputs", "samples")

# ============ 运行模式 ============
#   full:    训练 LSTM + Transformer → 生成 5 首句续写 + 5 藏头诗 → 评测报告
#   train:   仅训练（用 MODEL 指定模型）
#   generate:仅生成（需已有 checkpoint）
#   eval:    完整评测
MODE = "generate"                # full | train | generate | eval
MODEL = "rhyme_transformer"          # lstm | transformer | rhyme_transformer

# ============ 数据集 ============
GDRIVE_FILE_ID = "1YcP6B28KsOwacr7C_j1tcstkA-QtPd61"

# ============ 词表 ============
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN = \
    "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<UNK>"
PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX, UNK_IDX = 0, 1, 2, 3, 4
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN]
MIN_CHAR_FREQ = 3
MAX_VOCAB_SIZE = 6000

# ============ 模型超参 ============
EMBED_DIM = 128
DROPOUT = 0.25           # 适度提高 dropout 防过拟合
LSTM_HIDDEN = 512        # 加大隐藏层容量 (384→512)
LSTM_LAYERS = 3
T_D_MODEL = 384
T_NHEAD = 8
T_LAYERS = 5             # 6→5 减层防过拟合
T_FFN = 1280             # 1536→1280 降低 FFN 容量
MAX_SEQ_LEN = 40

# ============ 训练参数 ============
BATCH_SIZE = 64
LR = 5e-4             # 降低初始学习率，更稳定收敛
WEIGHT_DECAY = 1e-5
EPOCHS = 50           # 加大训练轮数（早停会自动截断）
GRAD_CLIP = 1.0
LR_STEP = 15          # LR 衰减周期（epoch 数增加后相应延后）
LR_GAMMA = 0.5
MAX_TRAIN = None      # 训练集上限（None=全用~124K 首）
TRAIN_SPLIT = 0.9
EARLY_STOP_PATIENCE = 8    # 过拟合时早停兜底
EARLY_STOP_MIN_DELTA = 0.005  # 改善阈值: PPL相对下降 ≥ 0.5% 才算有效改善

# ============ 生成参数 ============
GEN_MODE = "first_line"     # first_line | acrostic（generate.py 独立运行时的模式）
GEN_INPUT = "爆竹声中一岁除"  # generate.py 独立运行时的默认输入
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.0                 # Nucleus sampling (0=关闭, 建议 0.92)
REPETITION_PENALTY = 1.25   # 重复惩罚 (1.15→1.25, 更强抑制)
NO_REPEAT_NGRAM_SIZE = 3    # 禁止同一 3-gram 再次出现 (0=关闭)
MAX_GEN_LEN = 50
GEN_NUM = 5                 # 每条输入生成数

# ============ 首句续写测试输入（5句） ============
FIRST_LINES = [
    "爆竹声中一岁除",
    "一从大地起风雷",
    "早岁那知世事艰",
    "天门中断楚江开",
    "月落乌啼霜满天",
]

# ============ 藏头诗测试输入（5组） ============
ACROSTICS = [
    "春花秋月",
    "四季平安",
    "北京邮电",
    "中华复兴",
    "用晦而明",
]

# ============ 韵律微调参数 ============
FT_EPOCHS = 20           # 微调轮数（在预训练基础上）
FT_LR = 1e-4             # 微调学习率（远低于预训练）
FT_RHYME_WEIGHT = 2.0    # 韵母对比 loss 权重（直接作用在 LM logits）
FT_TONE_WEIGHT = 1.0     # 平仄对比 loss 权重

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
