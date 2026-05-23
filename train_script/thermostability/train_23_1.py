"""
蛋白质语言模型训练框架 - 回归模型版
用于预测蛋白质热稳定性的回归模型
支持 S-PLM、ESMC、ESM2 三种模型的单独或融合训练
"""
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import multiprocessing as mp
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'sans-serif']
# 设置负号显示
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置多进程启动方式为 'spawn'
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

# 禁用不必要的警告
warnings.filterwarnings('ignore')
# 单次训练
#===============================================================================
# 配置模块
#===============================================================================

class ModelConfig:
    """模型配置基类"""
    def __init__(self, model_name, model_path=None, enabled=True):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dim = None  # 模型输出维度
        self.enabled = enabled  # 是否启用该模型
        self.device = None      # 模型所在设备

    def to_dict(self):
        """将配置转换为字典，用于保存"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "output_dim": self.output_dim,
            "enabled": self.enabled
        }

class ESM2Config(ModelConfig):
    """ESM2模型配置"""
    def __init__(self, model_path="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D", enabled=True):
        super().__init__("esm2", model_path, enabled)
        self.output_dim = 1280

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"type": "ESM2Config"})
        return base_dict

class ESMCConfig(ModelConfig):
    """ESM-C模型配置"""
    def __init__(self, model_path="esmc_600m", enabled=True):
        super().__init__("esmc", model_path, enabled)
        self.output_dim = 1152

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"type": "ESMCConfig"})
        return base_dict

class SPLMConfig(ModelConfig):
    """S-PLM模型配置"""
    def __init__(self, config_path="./configs/representation_config.yaml",
                 checkpoint_path="/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth",
                 enabled=True):
        super().__init__("splm", None, enabled)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dim = 1280

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "type": "SPLMConfig",
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path
        })
        return base_dict

class TrainingConfig:
    """训练配置"""
    def __init__(self):
        # 数据路径
        self.train_csv = "s2c2_0_train.csv"  # 修改为热稳定性训练数据
        self.test_csv = "s2c2_0_test.csv"    # 修改为热稳定性测试数据
        self.target_column = "tgt_reg"        # 修改为热稳定性回归目标列名
        self.sequence_column = "sequence"     # 序列列名不变

        # 模型保存路径
        self.model_save_dir = "./protein_stability_results"

        # 训练参数
        self.batch_size = 16
        self.epochs = 20
        self.lr = 5e-5
        self.weight_decay = 1e-6
        self.max_seq_len = 600

        # 模型参数
        self.hidden_dim = 512
        self.dropout = 0.2


        # 特征参数
        self.normalize_features = True
        self.feature_cache_size = 400

        # 特征归一化参数
        self.normalize_features = True  # 是否启用特征归一化
        self.normalization_method = "global"  # 归一化方法: "none", "global", "sequence", "layer"
        # 预计算的统计值 (将在首次运行数据集时填充)
        self.esm2_mean = 0.0
        self.esm2_std = 1.0
        self.esmc_mean = 0.0
        self.esmc_std = 1.0
        self.splm_mean = 0.0
        self.splm_std = 1.0

        # 训练模式
        self.train_mode = "fusion"  # 'fusion', 'single', 'ensemble'
        self.fusion_type = "default"  # 'default', 'weighted', 'concat'

        # 训练设置
        self.use_amp = True  # 混合精度训练
        self.grad_clip = 1.0
        self.num_workers = 2
        self.num_folds = 5  # 5折交叉验证
        self.random_seed = 42
        self.warmup_ratio = 0.1
        self.patience = 5  # 早停

        # GPU设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1
        self.use_separate_gpus = True  # 特征提取模型和训练模型使用不同GPU

        if torch.cuda.device_count() >= 2:
            # 默认：特征提取GPU 0，训练GPU 1
            self.feature_extraction_device = torch.device("cuda:0")
            self.training_device = torch.device("cuda:1")
        else:
            # 只有一个GPU时共用
            self.feature_extraction_device = self.device
            self.training_device = self.device

    def set_seed(self):
        """设置随机种子以确保可重现性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def to_dict(self):
        """将配置转换为字典，用于保存"""
        return {k: v if not isinstance(v, torch.device) else str(v)
                for k, v in self.__dict__.items()}

class ExperimentConfig:
    """实验配置，整合训练配置和模型配置"""
    def __init__(self, name="protein_stability_experiment"):
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_config = TrainingConfig()

        # 默认包含所有模型
        self.model_configs = {
            "esm2": ESM2Config(),
            "esmc": ESMCConfig(),
            "splm": SPLMConfig(enabled=False)  # 默认禁用S-PLM，因为需要额外依赖
        }

    def get_run_dir(self):
        """获取运行目录"""
        base_dir = self.training_config.model_save_dir
        return os.path.join(base_dir, f"{self.name}_{self.timestamp}")

    def save_config(self, filepath, additional_data=None):
        """保存配置到JSON文件，可选择包含额外数据"""
        config_dict = {
            "name": self.name,
            "timestamp": self.timestamp,
            "training_config": self.training_config.to_dict(),
            "model_configs": {k: v.to_dict() for k, v in self.model_configs.items()}
        }

        # 添加额外数据
        if additional_data:
            config_dict.update(self._convert_numpy_types(additional_data))

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python标准类型"""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(i) for i in obj)
        else:
            return obj

    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        instance = cls(config_dict.get("name", "loaded_experiment"))
        instance.timestamp = config_dict.get("timestamp", instance.timestamp)

        # 加载训练配置
        for k, v in config_dict.get("training_config", {}).items():
            if hasattr(instance.training_config, k):
                setattr(instance.training_config, k, v)

        # 加载模型配置
        instance.model_configs = {}
        for model_name, model_config in config_dict.get("model_configs", {}).items():
            if model_config.get("type") == "ESM2Config":
                instance.model_configs[model_name] = ESM2Config(
                    model_path=model_config.get("model_path"),
                    enabled=model_config.get("enabled", True)
                )
            elif model_config.get("type") == "ESMCConfig":
                instance.model_configs[model_name] = ESMCConfig(
                    model_path=model_config.get("model_path"),
                    enabled=model_config.get("enabled", True)
                )
            elif model_config.get("type") == "SPLMConfig":
                instance.model_configs[model_name] = SPLMConfig(
                    config_path=model_config.get("config_path", "./configs/representation_config.yaml"),
                    checkpoint_path=model_config.get("checkpoint_path"),
                    enabled=model_config.get("enabled", False)
                )

        return instance

    @classmethod
    def load_config(cls, filepath):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

#===============================================================================
# 日志模块
#===============================================================================

class Logger:
    """日志管理类"""
    def __init__(self, log_file=None, console=True):
        self.log_file = log_file
        self.console = console

        # 创建日志目录
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # 打印到控制台
        if self.console:
            print(formatted_message)

        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")

    def info(self, message):
        self.log(message, "INFO")

    def warning(self, message):
        self.log(message, "WARNING")

    def error(self, message):
        self.log(message, "ERROR")

    def debug(self, message):
        self.log(message, "DEBUG")

#===============================================================================
# 数据加载与预处理模块
#===============================================================================

class SequenceDataset:
    """蛋白质序列数据集加载与预处理"""

    @staticmethod
    def load_from_csv(file_path, sequence_col="sequence", target_col="tgt_reg", logger=None):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(file_path)
            if logger:
                logger.info(f"从{file_path}加载了{len(df)}条数据")
            return df
        except Exception as e:
            if logger:
                logger.error(f"加载{file_path}失败: {str(e)}")
            return None

    @staticmethod
    def get_data_stats(df, target_col="tgt_reg"):
        """获取数据集统计信息"""
        stats = {}
        stats["total_count"] = len(df)

        # 目标分布
        if target_col in df.columns:
            stats["target_mean"] = float(df[target_col].mean())
            stats["target_std"] = float(df[target_col].std())
            stats["target_min"] = float(df[target_col].min())
            stats["target_max"] = float(df[target_col].max())
            stats["target_median"] = float(df[target_col].median())

        # 序列长度分布
        if "sequence" in df.columns:
            seq_lens = df["sequence"].str.len()
            stats["seq_len_mean"] = float(seq_lens.mean())
            stats["seq_len_std"] = float(seq_lens.std())
            stats["seq_len_min"] = int(seq_lens.min())
            stats["seq_len_max"] = int(seq_lens.max())
            stats["seq_len_median"] = int(seq_lens.median())

        return stats

#===============================================================================
# 特征提取模块
#===============================================================================

class FeatureExtractor:
    """特征提取基类"""
    _model_registry = {}  # 全局模型注册表，用于跨实例共享

    def __init__(self, config, device=None, logger=None):
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.logger = logger
        self._model_key = f"{self.__class__.__name__}_{config.model_name}"

    def load_model(self):
        """加载模型，优先从注册表获取"""
        if self._model_key in FeatureExtractor._model_registry:
            if self.logger:
                self.logger.info(f"从注册表获取已加载的{self.config.model_name}模型")
            self.model = FeatureExtractor._model_registry[self._model_key]
            return

        # 加载新模型的逻辑（子类实现）
        self._load_model_impl()

        # 加载成功后注册到全局
        if self.model is not None:
            FeatureExtractor._model_registry[self._model_key] = self.model

    def _load_model_impl(self):
        """实际加载模型的实现（子类必须重写）"""
        raise NotImplementedError("子类必须实现此方法")

    def extract_features(self, sequence, max_len):
        """提取特征"""
        if self.model is None:
            self.load_model()
        # 特征提取逻辑（子类实现）

    def cleanup(self):
        """清理资源（只在程序结束时调用）"""
        # 不删除模型，改为从注册表中移除
        if self._model_key in FeatureExtractor._model_registry:
            del FeatureExtractor._model_registry[self._model_key]
            if self.logger:
                self.logger.info(f"已从注册表移除{self.config.model_name}模型")

        # 手动清理GPU内存
        if self.model is not None:
            self.model = None

        torch.cuda.empty_cache()

class ESM2Extractor(FeatureExtractor):
    """ESM2特征提取器"""
    def __init__(self, config, device=None, logger=None):
        super().__init__(config, device, logger)
        self.tokenizer = None

    def load_model(self):
        """加载ESM2模型"""
        if self.model is not None:
            return self.model

        try:
            if self.logger:
                self.logger.info(f"加载ESM2模型: {self.config.model_path}")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            # 配置模型
            model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)

            # 关闭dropout以确保推理结果确定性
            model_config.hidden_dropout = 0.
            model_config.hidden_dropout_prob = 0.
            model_config.attention_dropout = 0.
            model_config.attention_probs_dropout_prob = 0.

            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                config=model_config
            ).to(self.device).eval()

            if self.logger:
                self.logger.info(f"ESM2模型加载成功，设备: {self.device}")

            return self.model

        except Exception as e:
            if self.logger:
                self.logger.error(f"ESM2模型加载失败: {e}")
            return None

    def extract_features(self, sequence, max_len):
        """从ESM2模型提取特征"""
        if self.model is None:
            self.load_model()

        with torch.no_grad():
            # 在氨基酸之间添加空格
            spaced_seq = " ".join(list(sequence))

            # 编码序列
            inputs = self.tokenizer.encode_plus(
                spaced_seq,
                return_tensors=None,
                add_special_tokens=True,
                padding=True,
                truncation=True
            )

            # 转换为tensor并移至设备
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0).to(self.device)

            # 前向传播
            outputs = self.model(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'])

            # 提取最后一层隐藏状态
            last_hidden_states = outputs[0]

            # 提取有效token的嵌入 (跳过首尾特殊标记)
            encoded_seq = last_hidden_states[0, inputs['attention_mask'][0].bool()][1:-1].cpu()

            # 处理序列长度
            current_len = encoded_seq.shape[0]
            if current_len < max_len:
                # 填充
                pad_len = max_len - current_len
                padded_residue = torch.zeros((max_len, encoded_seq.size(1)))
                padded_residue[:current_len] = encoded_seq
                padded_mask = torch.zeros(max_len, dtype=torch.bool)
                padded_mask[:current_len] = True
            elif current_len > max_len:
                # 截断
                padded_residue = encoded_seq[:max_len]
                padded_mask = torch.ones(max_len, dtype=torch.bool)
            else:
                padded_residue = encoded_seq
                padded_mask = torch.ones(max_len, dtype=torch.bool)

            # 计算整体表示（平均池化）
            global_representation = encoded_seq.mean(dim=0)

            return padded_residue, padded_mask, global_representation

    def _load_model_impl(self):
        """加载ESM2模型（仅在全局注册表没有时调用）"""
        try:
            if self.logger:
                self.logger.info(f"加载ESM2模型: {self.config.model_path}")

            # 加载模型并移至指定设备
            model_data = torch.load(os.path.join(self.config.model_path, "model.pt"))
            self.model = model_data["model"].to(self.device)
            self.model.eval()  # 设置为评估模式

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            if self.logger:
                self.logger.info(f"ESM2模型加载成功，设备: {self.device}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"ESM2模型加载失败: {str(e)}")
            self.model = None

class ESMCExtractor(FeatureExtractor):
    """ESM-C特征提取器"""
    _shared_model = None
    _load_lock = mp.Lock()

    def load_model(self):
        """加载ESM-C模型，加入共享机制和错误重试"""
        # 如果已有共享模型，直接使用
        if ESMCExtractor._shared_model is not None:
            self.model = ESMCExtractor._shared_model
            return self.model

        if self.model is not None:
            return self.model

        # 使用互斥锁防止多进程同时加载
        with ESMCExtractor._load_lock:
            # 双重检查，可能在获取锁的过程中已被其他进程加载
            if ESMCExtractor._shared_model is not None:
                self.model = ESMCExtractor._shared_model
                return self.model

            # 先清理GPU缓存
            torch.cuda.empty_cache()

            try:
                if self.logger:
                    self.logger.info(f"加载ESM-C模型: {self.config.model_path}")

                from esm.models.esmc import ESMC

                # 尝试在GPU上加载
                self.model = ESMC.from_pretrained(self.config.model_path).to(self.device).eval()

                # 设置为共享模型
                ESMCExtractor._shared_model = self.model

                if self.logger:
                    self.logger.info(f"ESM-C模型加载成功，设备: {self.device}")

                return self.model

            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and self.logger:
                    self.logger.warning(f"GPU内存不足，尝试在CPU上加载ESM-C模型: {e}")

                    try:
                        # 在CPU上加载
                        self.model = ESMC.from_pretrained(self.config.model_path).cpu().eval()

                        # 设置为共享模型
                        ESMCExtractor._shared_model = self.model

                        if self.logger:
                            self.logger.info("ESM-C模型在CPU上加载成功")

                        return self.model
                    except Exception as cpu_e:
                        if self.logger:
                            self.logger.error(f"ESM-C模型加载失败: {cpu_e}")
                        return None

                elif self.logger:
                    self.logger.error(f"ESM-C模型加载失败: {e}")
                return None

    def extract_features(self, sequence, max_len):
        """从ESM-C模型提取特征"""
        if self.model is None:
            self.load_model()

        # 如果模型加载失败，返回备用特征
        if self.model is None:
            if self.logger:
                self.logger.warning(f"ESM-C模型未加载，返回备用特征")
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global

        with torch.no_grad():
            try:
                # 确保当前操作在适当的设备上进行
                device = next(self.model.parameters()).device

                from esm.sdk.api import ESMProtein, LogitsConfig

                # 准备蛋白质数据
                protein = ESMProtein(sequence=sequence)
                protein_tensor = self.model.encode(protein).to(device)

                # 获取特征
                logits_output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )

                # 提取并处理嵌入特征，去除首尾标记
                embeddings = logits_output.embeddings[0][1:-1].cpu()

                # 处理长度
                current_len = embeddings.shape[0]
                if current_len < max_len:
                    # 填充
                    padded_residue = torch.zeros((max_len, embeddings.size(1)))
                    padded_residue[:current_len] = embeddings
                    padded_mask = torch.zeros(max_len, dtype=torch.bool)
                    padded_mask[:current_len] = True
                elif current_len > max_len:
                    # 截断
                    padded_residue = embeddings[:max_len]
                    padded_mask = torch.ones(max_len, dtype=torch.bool)
                else:
                    padded_residue = embeddings
                    padded_mask = torch.ones(max_len, dtype=torch.bool)

                # 计算整体表示
                global_representation = embeddings.mean(dim=0)

                return padded_residue, padded_mask, global_representation

            except Exception as e:
                if self.logger:
                    self.logger.error(f"ESM-C特征提取失败: {e}")

                # 返回备用特征
                dummy_features = torch.zeros((max_len, self.config.output_dim))
                dummy_mask = torch.zeros(max_len, dtype=torch.bool)
                dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
                dummy_global = torch.zeros(self.config.output_dim)
                return dummy_features, dummy_mask, dummy_global
    def _load_model_impl(self):
        """加载ESM-C模型（仅在全局注册表没有时调用）"""
        try:
            if self.logger:
                self.logger.info(f"加载ESM-C模型: {self.config.model_path}")

            # 这里使用 AutoModel 替代
            config = AutoConfig.from_pretrained(self.config.model_path)
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                config=config
            ).to(self.device)
            self.model.eval()  # 设置为评估模式

            # 确保tokenizer也被加载（如果需要）
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            if self.logger:
                self.logger.info(f"ESM-C模型加载成功，设备: {self.device}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"ESM-C模型加载失败: {str(e)}")
            self.model = None


# S-PLM特征提取器
class SPLMExtractor(FeatureExtractor):
    """S-PLM特征提取器"""
    def load_model(self):
        """加载S-PLM模型"""
        if self.model is not None:
            return self.model

        try:
            # 加载S-PLM模型
            if self.logger:
                self.logger.info(f"加载S-PLM模型: {self.config.checkpoint_path}")

            try:
                # 根据 splm_extract_4.py 中的实现调整
                import yaml
                from GG_thermostability._utils import load_configs, load_checkpoints_only
                from model import SequenceRepresentation

                # 加载配置文件
                try:
                    # 如果是字符串路径，直接加载文件
                    if isinstance(self.config.config_path, str):
                        with open(self.config.config_path) as file:
                            dict_config = yaml.full_load(file)
                        configs = load_configs(dict_config)
                    else:
                        # 如果已经是字典，直接使用
                        configs = load_configs(self.config.config_path)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"加载S-PLM配置文件失败: {e}")
                    return None

                # 创建模型
                model = SequenceRepresentation(logging=None, configs=configs)
                model.to(self.device)

                # 加载检查点
                load_checkpoints_only(self.config.checkpoint_path, model)
                model.eval()  # 设置为评估模式

                self.model = model

                if self.logger:
                    self.logger.info(f"S-PLM模型加载成功，设备: {self.device}")

                return self.model
            except ImportError as e:
                if self.logger:
                    self.logger.error(f"加载S-PLM模型所需的模块未找到，请确保相关依赖已安装: {e}")
                return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"S-PLM模型加载失败: {e}")
            return None

    def extract_features(self, sequence, max_len):
        """从S-PLM模型提取特征，基于 splm_extract_4.py 的实现"""
        # 尝试加载模型
        if self.model is None:
            self.model = self.load_model()

        # 如果模型仍然为 None，返回备用特征
        if self.model is None:
            if self.logger:
                self.logger.warning(f"使用S-PLM备用特征，模型未成功加载")

            # 返回备用特征
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global

        try:
            with torch.no_grad():
                # 准备序列
                esm2_seq = [(range(len(sequence)), str(sequence))]

                # 使用模型的转换器
                batch_labels, batch_strs, batch_tokens = self.model.batch_converter(esm2_seq)

                # 获取输入 token
                token = batch_tokens.to(self.device)

                # 填充/截断到合适长度
                if token.size(1) < max_len:
                    padding = torch.ones((1, max_len - token.size(1)), dtype=token.dtype, device=self.device) * self.model.alphabet.padding_idx
                    token = torch.cat([token, padding], dim=1)
                elif token.size(1) > max_len:
                    token = token[:, :max_len]  # 截断过长序列

                # 获取蛋白质表示、残基表示和掩码
                protein_representation, residue_representation, mask = self.model(token)

                # 处理输出
                global_repr = protein_representation.squeeze(0).cpu()  # 全局表示
                residue_repr = residue_representation.squeeze(0).cpu()  # 残基表示
                attention_mask = mask.squeeze(0).cpu().bool()  # 注意力掩码

                # 确保长度匹配
                if residue_repr.size(0) != max_len:
                    if residue_repr.size(0) < max_len:
                        # 填充
                        padded_residue = torch.zeros((max_len, residue_repr.size(1)), device=residue_repr.device)
                        padded_residue[:residue_repr.size(0)] = residue_repr
                        padded_mask = torch.zeros(max_len, dtype=torch.bool, device=attention_mask.device)
                        padded_mask[:attention_mask.size(0)] = attention_mask

                        residue_repr = padded_residue
                        attention_mask = padded_mask
                    else:
                        # 截断
                        residue_repr = residue_repr[:max_len]
                        attention_mask = attention_mask[:max_len]

                return residue_repr, attention_mask, global_repr

        except Exception as e:
            if self.logger:
                self.logger.error(f"S-PLM特征提取失败: {e}")

            # 返回备用特征
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global

    def tokenize_sequence(self, sequence):
        """将序列转换为token_ids和attention_mask"""
        # 这是一个示例，实际情况需要根据S-PLM的tokenizer调整
        # 假设我们有一个简单的氨基酸到id的映射
        aa_to_id = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_to_id["<pad>"] = 0

        # 转换序列
        token_ids = [aa_to_id.get(aa, 0) for aa in sequence]
        attention_mask = [1] * len(token_ids)

        # 转换为tensor
        token_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return token_ids, attention_mask

class FeatureManager:
    """特征管理器，用于管理多种特征提取器"""
    def __init__(self, config, logger=None):
        self.config = config
        self.extractors = {}
        self.feature_device = config.training_config.feature_extraction_device
        self.logger = logger

        # 预加载所有模型标志
        self.preload_models = True

        # 注册已启用的模型配置
        for name, model_config in self.config.model_configs.items():
            if model_config.enabled:
                self.register_extractor(name, model_config)

        # 预加载所有模型
        if self.preload_models:
            self.preload_all_models()

    def register_extractor(self, name, extractor_config):
        """注册特征提取器"""
        if not extractor_config.enabled:
            if self.logger:
                self.logger.info(f"特征提取器 {name} 已禁用")
            return False

        try:
            if name == "esm2":
                self.extractors[name] = ESM2Extractor(extractor_config, self.feature_device, self.logger)
            elif name == "esmc":
                self.extractors[name] = ESMCExtractor(extractor_config, self.feature_device, self.logger)
            elif name == "splm":
                self.extractors[name] = SPLMExtractor(extractor_config, self.feature_device, self.logger)
            else:
                if self.logger:
                    self.logger.warning(f"未知的特征提取器类型: {name}")
                return False

            if self.logger:
                self.logger.info(f"已注册特征提取器: {name}")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"注册特征提取器{name}失败: {str(e)}")
            return False

    def extract_all_features(self, sequence, max_len):
        """提取所有已注册特征提取器的特征"""
        features = {}
        for name, extractor in self.extractors.items():
            try:
                result = extractor.extract_features(sequence, max_len)
                features[name] = result
            except Exception as e:
                if self.logger:
                    self.logger.error(f"提取{name}特征失败: {str(e)}")

                # 返回备用特征
                zero_dim = self.config.model_configs[name].output_dim
                features[name] = (
                    torch.zeros(max_len, zero_dim),  # residue_repr
                    torch.zeros(max_len),            # mask
                    torch.zeros(zero_dim)            # global_repr
                )

        return features
    def extract_all_sequences_features(self, df, sequence_column, max_len):
        """预先提取所有序列的特征并缓存到内存"""
        features_dict = {}
        total = len(df)

        if self.logger:
            self.logger.info(f"开始预提取所有{total}个序列的特征...")

        for idx in tqdm(range(total), desc="提取特征"):
            sequence = df.iloc[idx][sequence_column]

            try:
                # 提取特征
                features = self.extract_all_features(sequence, max_len)
                features_dict[idx] = features

            except Exception as e:
                if self.logger:
                    self.logger.error(f"序列 {idx} 特征提取失败: {str(e)}")
                # 使用空特征作为备用
                features_dict[idx] = self._create_empty_features(max_len)

        if self.logger:
            self.logger.info(f"所有{total}个序列的特征提取完成，共缓存了{len(features_dict)}个序列的特征")

        return features_dict

    def _create_empty_features(self, max_len):
        """创建备用特征，当特征提取失败时使用"""
        empty_features = {}
        for name, model_config in self.config.model_configs.items():
            if model_config.enabled:
                output_dim = model_config.output_dim
                empty_features[name] = (
                    torch.zeros((max_len, output_dim)),  # residue_repr
                    torch.zeros(max_len, dtype=torch.bool),  # mask
                    torch.zeros(output_dim)  # global_repr
                )
        return empty_features

    def cleanup(self):
        """清理所有特征提取器资源"""
        for name, extractor in self.extractors.items():
            try:
                extractor.cleanup()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"清理{name}特征提取器资源失败: {str(e)}")

    def preload_all_models(self):
        """预加载所有特征提取器的模型"""
        if self.logger:
            self.logger.info("开始预加载所有特征提取模型")

        for name, extractor in self.extractors.items():
            if extractor.model is None:
                extractor.load_model()

        if self.logger:
            self.logger.info("所有特征提取模型加载完成")
#===============================================================================
# 数据集与数据加载模块
#===============================================================================

class ProteinFeatureDataset(Dataset):
    """蛋白质特征数据集"""
    def __init__(self, df, feature_manager, config, target_col="tgt_reg",
                 sequence_col="sequence", cache_size=4000, logger=None,
                 pre_extracted_features=None):
        self.df = df
        self.feature_manager = feature_manager
        self.config = config
        self.target_col = target_col
        self.sequence_col = sequence_col
        self.logger = logger

        # 特征缓存
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

        # 预提取特征（如果提供）
        self.pre_extracted_features = pre_extracted_features

        # 获取目标值的均值和标准差，用于归一化
        self.target_mean = df[target_col].mean()
        self.target_std = df[target_col].std()

    def __len__(self):
        return len(self.df)

    def _update_cache(self, idx, features):
        """更新特征缓存"""
        if len(self.cache_order) >= self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            if oldest_idx in self.cache:
                del self.cache[oldest_idx]

        self.cache[idx] = features
        self.cache_order.append(idx)

    def _save_feature(self, idx, sequence, features):
        """保存提取的特征到文件"""
        if hasattr(self.config.training_config, 'save_features') and self.config.training_config.save_features:
            feature_dir = self.config.training_config.feature_cache_dir
            # 使用序列哈希作为文件名
            import hashlib
            seq_hash = hashlib.md5(sequence.encode()).hexdigest()
            feature_file = os.path.join(feature_dir, f"{seq_hash}.pt")
            torch.save(features, feature_file)
            if self.logger:
                self.logger.debug(f"特征已保存到 {feature_file}")
    def _normalize_features(self, features, feature_type):
        """归一化特征"""
        if not self.config.training_config.normalize_features:
            return features

        method = self.config.training_config.normalization_method
        residue_repr, mask, global_repr = features

        if method == "none":
            return features

        elif method == "global":
            # 使用预计算的全局统计值归一化
            if feature_type == "esm2":
                mean = self.config.training_config.esm2_mean
                std = self.config.training_config.esm2_std
            elif feature_type == "esmc":
                mean = self.config.training_config.esmc_mean
                std = self.config.training_config.esmc_std
            elif feature_type == "splm":
                mean = self.config.training_config.splm_mean
                std = self.config.training_config.splm_std
            else:
                return features

            # 避免除零
            std = 1.0 if std == 0 else std

            # 归一化残基表示和全局表示
            norm_residue_repr = (residue_repr - mean) / std
            norm_global_repr = (global_repr - mean) / std

            return norm_residue_repr, mask, norm_global_repr

        elif method == "sequence":
            # 对每个序列单独归一化
            valid_mask = mask.bool()
            if valid_mask.sum() > 0:
                valid_repr = residue_repr[valid_mask]
                seq_mean = valid_repr.mean()
                seq_std = valid_repr.std()
                seq_std = 1.0 if seq_std == 0 else seq_std

                norm_residue_repr = residue_repr.clone()
                norm_residue_repr[valid_mask] = (valid_repr - seq_mean) / seq_std
                norm_global_repr = (global_repr - seq_mean) / seq_std

                return norm_residue_repr, mask, norm_global_repr

            return features

        elif method == "layer":
            # 对每个特征维度单独归一化
            valid_mask = mask.bool()
            if valid_mask.sum() > 0:
                valid_repr = residue_repr[valid_mask]
                layer_mean = valid_repr.mean(dim=0, keepdim=True)
                layer_std = valid_repr.std(dim=0, keepdim=True)
                layer_std[layer_std == 0] = 1.0

                norm_residue_repr = residue_repr.clone()
                norm_residue_repr[valid_mask] = (valid_repr - layer_mean) / layer_std

                # 全局表示使用同样的归一化
                norm_global_repr = (global_repr - layer_mean.squeeze(0)) / layer_std.squeeze(0)

                return norm_residue_repr, mask, norm_global_repr

            return features

        return features

    def __getitem__(self, idx):
        """获取数据集项"""
        # 获取目标值
        target = self.df.iloc[idx][self.target_col]

        # 优先使用预提取的特征
        if self.pre_extracted_features is not None and idx in self.pre_extracted_features:
            features = self.pre_extracted_features[idx]

            # 归一化特征
            normalized_features = {}
            for feature_type, feature_tuple in features.items():
                normalized_features[feature_type] = self._normalize_features(feature_tuple, feature_type)

            return normalized_features, target

        # 其次尝试从缓存获取特征
        if idx in self.cache:
            features = self.cache[idx]
            return features, target

        # 提取特征的现有代码...
        sequence = self.df.iloc[idx][self.sequence_col]

        # 提取特征
        features = self.feature_manager.extract_all_features(
            sequence,
            self.config.training_config.max_seq_len
        )

        # 归一化特征
        normalized_features = {}
        for feature_type, feature_tuple in features.items():
            normalized_features[feature_type] = self._normalize_features(feature_tuple, feature_type)

        # 更新缓存
        self._update_cache(idx, normalized_features)

        return normalized_features, target

def collate_protein_features(batch):
    """
    批处理函数，处理不同长度的序列

    batch结构:
    [(features_dict_1, label_1), (features_dict_2, label_2), ...]

    features_dict结构:
    {
        'esm2': (residue_repr, mask, global_repr),
        'esmc': (residue_repr, mask, global_repr),
        ...
    }
    """
    features_dict, labels = zip(*batch)
    result = {'labels': torch.tensor(labels, dtype=torch.float)}

    # 获取特征名列表
    feature_names = list(features_dict[0].keys())

    # 对于混合精度训练，统一使用 float32，模型内部会自动转换
    dtype = torch.float32

    for name in feature_names:
        # 收集所有批次的这个特征
        residue_repr_list = [item[name][0] for item in features_dict]
        mask_list = [item[name][1] for item in features_dict]
        global_repr_list = [item[name][2] for item in features_dict]

        # 堆叠(stack)成批次tensor
        result[f'{name}_residue'] = torch.stack(residue_repr_list).to(dtype)
        result[f'{name}_mask'] = torch.stack(mask_list).to(dtype)
        result[f'{name}_global'] = torch.stack(global_repr_list).to(dtype)

    return result

#===============================================================================
# 模型定义模块
#===============================================================================

class SingleModelRegressor(nn.Module):
    """单一蛋白质语言模型回归器"""
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, model_name="esm2"):
        super().__init__()
        self.model_name = model_name

        # 全局特征处理
        self.global_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层（回归）
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch):
        # 获取特征
        global_feats = batch[f'{self.model_name}_global']

        # 全局特征处理
        global_embedding = self.global_fc(global_feats)

        # 输出层（回归）
        output = self.output_layer(global_embedding).squeeze(-1)

        return output

    def save_model(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.global_fc[0].in_features,
                'hidden_dim': self.global_fc[0].out_features,
                'dropout': self.global_fc[2].p,
                'model_name': self.model_name
            }
        }, path)

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            model_name=config['model_name']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

class WeightedFusionRegressor(nn.Module):
    """加权融合模型回归器，使用与 ThermostabilityFusionModel 相同的架构"""
    def __init__(self, model_configs, hidden_dim=512, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.model_names = [name for name, config in model_configs.items() if config.enabled]
        self.output_dims = {name: config.output_dim for name, config in model_configs.items() if config.enabled}

        # 确保至少有一个模型被启用
        assert len(self.model_names) > 0, "至少需要一个启用的模型"

        # 确保只有两个模型被使用（ESM2和ESMC）
        if len(self.model_names) > 2:
            raise ValueError("当前实现仅支持两个模型的融合")

        # 记录模型名称
        self.esm2_name = self.model_names[0]  # 假设第一个是 ESM2
        self.esmc_name = self.model_names[1] if len(self.model_names) > 1 else None  # 第二个是 ESMC

        # 获取输入维度
        self.esm2_dim = self.output_dims[self.esm2_name]
        self.esmc_dim = self.output_dims[self.esmc_name] if self.esmc_name else 0

        self.use_layer_norm = use_layer_norm

        # 可选的层归一化
        if use_layer_norm:
            self.esm2_norm = nn.LayerNorm(self.esm2_dim)
            if self.esmc_name:
                self.esmc_norm = nn.LayerNorm(self.esmc_dim)

        # 特征投影层
        self.esm2_proj = nn.Linear(self.esm2_dim, hidden_dim)
        if self.esmc_name:
            self.esmc_proj = nn.Linear(self.esmc_dim, hidden_dim)

        # 特征编码层 - 为每个模型分别添加
        self.esm2_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        if self.esmc_name:
            self.esmc_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

        # 可学习特征权重参数
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

        # 保存归一化后的权重
        self.normalized_weights = None

    def forward(self, batch):
        # 获取特征
        esm2_features = batch[f'{self.esm2_name}_global']
        esmc_features = batch[f'{self.esmc_name}_global'] if self.esmc_name else None

        # 可选的层归一化
        if self.use_layer_norm:
            esm2_features = self.esm2_norm(esm2_features)
            if esmc_features is not None:
                esmc_features = self.esmc_norm(esmc_features)

        # 投影到相同维度
        p_esm2 = self.esm2_proj(esm2_features)

        # 通过特征编码层增强特征表示
        p_esm2 = self.esm2_encoder(p_esm2)

        # 如果有第二个模型
        if esmc_features is not None:
            p_esmc = self.esmc_proj(esmc_features)
            p_esmc = self.esmc_encoder(p_esmc)

            # 加权融合
            alpha = torch.sigmoid(self.alpha)  # 转换到0-1范围
            weighted = alpha * p_esm2 + (1 - alpha) * p_esmc

            # 保存权重用于分析
            self.normalized_weights = {
                self.esm2_name: float(alpha.item()),
                self.esmc_name: float(1 - alpha.item())
            }
        else:
            # 只有一个模型时
            weighted = p_esm2
            self.normalized_weights = {self.esm2_name: 1.0}

        # 预测
        return self.head(weighted).squeeze(-1)

    def get_model_weights(self):
        """获取各模型的当前权重"""
        if self.normalized_weights is None:
            # 如果未进行前向传播，手动计算一次
            alpha = torch.sigmoid(self.alpha).item()
            if self.esmc_name:
                return {
                    self.esm2_name: alpha,
                    self.esmc_name: 1.0 - alpha
                }
            else:
                return {self.esm2_name: 1.0}
        return self.normalized_weights

    def save_model(self, path):
        # 确保权重被计算
        if self.normalized_weights is None:
            self.get_model_weights()

        torch.save({
            'state_dict': self.state_dict(),
            'model_names': self.model_names,
            'output_dims': self.output_dims,
            'normalized_weights': self.normalized_weights,
            'config': {
                'hidden_dim': self.esm2_encoder[0].out_features,
                'dropout': self.esm2_encoder[3].p,
                'use_layer_norm': self.use_layer_norm
            }
        }, path)

    @classmethod
    def load_model(cls, path, model_configs=None):
        checkpoint = torch.load(path)

        if model_configs is None:
            # 根据保存的配置重新创建model_configs
            model_configs = {}
            for name in checkpoint['model_names']:
                config = ModelConfig(name)
                config.output_dim = checkpoint['output_dims'][name]
                config.enabled = True
                model_configs[name] = config

        model = cls(
            model_configs=model_configs,
            hidden_dim=checkpoint['config']['hidden_dim'],
            dropout=checkpoint['config']['dropout'],
            use_layer_norm=checkpoint['config'].get('use_layer_norm', True)
        )
        model.load_state_dict(checkpoint['state_dict'])

        # 加载保存的归一化权重
        if 'normalized_weights' in checkpoint:
            model.normalized_weights = checkpoint['normalized_weights']

        return model

class FusionModelRegressor(nn.Module):
    """融合多个蛋白质语言模型的回归器"""
    def __init__(self, model_configs, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.model_names = [name for name, config in model_configs.items() if config.enabled]
        self.output_dims = {name: config.output_dim for name, config in model_configs.items() if config.enabled}

        # 确保至少有一个模型被启用
        assert len(self.model_names) > 0, "至少需要一个启用的模型"

        # 为每个模型创建嵌入提取层
        self.global_fc_layers = nn.ModuleDict()
        for name in self.model_names:
            self.global_fc_layers[name] = nn.Sequential(
                nn.Linear(self.output_dims[name], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # 融合层（合并所有特征）
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * len(self.model_names), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层（回归）
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, batch):
        # 处理每个模型的全局特征
        global_embeddings = []
        for name in self.model_names:
            global_feat = batch[f'{name}_global']
            global_embedding = self.global_fc_layers[name](global_feat)
            global_embeddings.append(global_embedding)

        # 连接所有嵌入
        concat_embedding = torch.cat(global_embeddings, dim=1)

        # 融合层
        fused_embedding = self.fusion_layer(concat_embedding)

        # 输出层（回归）
        output = self.output_layer(fused_embedding).squeeze(-1)

        return output

    def save_model(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'model_names': self.model_names,
            'output_dims': self.output_dims,
            'config': {
                'hidden_dim': next(iter(self.global_fc_layers.values()))[0].out_features,
                'dropout': next(iter(self.global_fc_layers.values()))[2].p
            }
        }, path)

    @classmethod
    def load_model(cls, path, model_configs=None):
        checkpoint = torch.load(path)

        if model_configs is None:
            # 根据保存的配置重新创建model_configs
            model_configs = {}
            for name in checkpoint['model_names']:
                config = ModelConfig(name)
                config.output_dim = checkpoint['output_dims'][name]
                config.enabled = True
                model_configs[name] = config

        model = cls(
            model_configs=model_configs,
            hidden_dim=checkpoint['config']['hidden_dim'],
            dropout=checkpoint['config']['dropout']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

#===============================================================================
# 训练与评估模块
#===============================================================================

class ModelTrainer:
    """模型训练与评估类"""
    def __init__(self, experiment_config, logger=None):
        self.config = experiment_config
        self.logger = logger

        # 训练配置
        self.train_config = experiment_config.training_config
        self.model_configs = experiment_config.model_configs

        # 设置随机种子
        self.train_config.set_seed()

        # 特征管理器
        self.feature_manager = self._init_feature_manager()

        # 加载数据
        self.train_df = SequenceDataset.load_from_csv(
            self.train_config.train_csv,
            self.train_config.sequence_column,
            self.train_config.target_column,
            self.logger
        )

        self.test_df = SequenceDataset.load_from_csv(
            self.train_config.test_csv,
            self.train_config.sequence_column,
            self.train_config.target_column,
            self.logger
        )

        if self.train_df is None or self.test_df is None:
            raise ValueError("数据加载失败")

        # 打印数据统计
        train_stats = SequenceDataset.get_data_stats(self.train_df, self.train_config.target_column)
        test_stats = SequenceDataset.get_data_stats(self.test_df, self.train_config.target_column)

        self.log(f"训练集统计: {train_stats}")
        self.log(f"测试集统计: {test_stats}")

    def log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _init_feature_manager(self):
        """初始化特征管理器"""
        feature_manager = FeatureManager(self.config, self.logger)

        # 注册已启用的特征提取器
        for name, model_config in self.model_configs.items():
            if model_config.enabled:
                feature_manager.register_extractor(name, model_config)

        return feature_manager

    def _create_model(self):
        """根据训练配置创建模型"""
        if self.train_config.train_mode == "fusion":
            return self._create_fusion_model()
        else:
            return self._create_single_model()

    def _create_fusion_model(self):
        """创建融合模型"""
        if self.train_config.fusion_type == "weighted":
            return WeightedFusionRegressor(
                self.model_configs,
                self.train_config.hidden_dim,
                self.train_config.dropout
            )
        else:
            return FusionModelRegressor(
                self.model_configs,
                self.train_config.hidden_dim,
                self.train_config.dropout
            )

    def _create_single_model(self):
        """创建单一模型"""
        # 使用第一个启用的模型
        for name, config in self.model_configs.items():
            if config.enabled:
                return SingleModelRegressor(
                    config.output_dim,
                    self.train_config.hidden_dim,
                    self.train_config.dropout,
                    name
                )

        raise ValueError("没有启用的模型可用")

    def pre_extract_all_features(self):
        """提前提取所有序列特征"""
        self.log("开始预提取所有序列的特征...")

        # 提取训练集特征
        self.train_features = self.feature_manager.extract_all_sequences_features(
            self.train_df,
            self.train_config.sequence_column,
            self.train_config.max_seq_len
        )

        # 提取测试集特征
        self.test_features = self.feature_manager.extract_all_sequences_features(
            self.test_df,
            self.train_config.sequence_column,
            self.train_config.max_seq_len
        )

        self.log("所有特征预提取完成")
        return self.train_features, self.test_features

    def train_kfold(self):
        """K折交叉验证训练"""
        # 准备路径
        run_dir = self.config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        # 保存初始配置
        config_path = os.path.join(run_dir, "config.json")
        self.config.save_config(config_path)

        # 在训练前预提取所有特征
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            self.log(f"检测到只有{gpu_count}个GPU，启用特征预提取模式并使用同一GPU进行训练")
            # 确保训练设备与特征提取设备一致
            self.train_config.training_device = self.train_config.feature_extraction_device
            train_features, test_features = self.pre_extract_all_features()
            use_preextracted = True
        else:
            self.log(f"检测到 {gpu_count} 个GPU，使用标准训练模式")
            train_features, test_features = None, None
            use_preextracted = False

        # 准备K折交叉验证
        kf = KFold(n_splits=self.train_config.num_folds, shuffle=True,
                random_state=self.train_config.random_seed)

        # 记录每折的指标
        all_metrics = []

        # 创建测试集数据加载器
        test_dataset = ProteinFeatureDataset(
            self.test_df,
            self.feature_manager,
            self.config,
            self.train_config.target_column,
            self.train_config.sequence_column,
            self.train_config.feature_cache_size,
            self.logger,
            pre_extracted_features=test_features if use_preextracted else None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=collate_protein_features,
            num_workers=self.train_config.num_workers
        )

        # 开始K折训练
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_df)):
            self.log(f"开始训练第 {fold+1}/{self.train_config.num_folds} 折")

            # 准备数据
            train_fold_df = self.train_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df = self.train_df.iloc[val_idx].reset_index(drop=True)

            # 如果使用预提取特征，创建每折的特征子集
            if use_preextracted:
                train_fold_features = {new_idx: train_features[old_idx] for new_idx, old_idx in enumerate(train_idx)}
                val_fold_features = {new_idx: train_features[old_idx] for new_idx, old_idx in enumerate(val_idx)}
            else:
                train_fold_features = None
                val_fold_features = None

            train_dataset = ProteinFeatureDataset(
                train_fold_df,
                self.feature_manager,
                self.config,
                self.train_config.target_column,
                self.train_config.sequence_column,
                self.train_config.feature_cache_size,
                self.logger,
                pre_extracted_features=train_fold_features
            )

            val_dataset = ProteinFeatureDataset(
                val_fold_df,
                self.feature_manager,
                self.config,
                self.train_config.target_column,
                self.train_config.sequence_column,
                self.train_config.feature_cache_size,
                self.logger,
                pre_extracted_features=val_fold_features
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                collate_fn=collate_protein_features,
                num_workers=self.train_config.num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                collate_fn=collate_protein_features,
                num_workers=self.train_config.num_workers
            )

            # 创建和训练模型
            model = self._create_model()
            fold_metrics = self._train_fold(model, train_loader, val_loader, test_loader, fold)
            all_metrics.append(fold_metrics)

        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(all_metrics)
        self.log(f"平均指标: {avg_metrics}")

        self.log("执行集成模型预测...")
        ensemble_metrics = self._ensemble_predict(all_metrics, run_dir)

        # 记录集成模型结果
        self.log(f"集成模型测试指标: {ensemble_metrics}")

        # 将所有结果添加到配置中
        additional_data = {
            'fold_metrics': self._serialize_metrics(all_metrics),
            'average_metrics': self._serialize_metrics(avg_metrics),
            'ensemble_metrics': self._serialize_metrics(ensemble_metrics)
        }

        # 更新配置文件，包含所有训练结果
        self.config.save_config(config_path, additional_data)
        self.log(f"已更新配置文件，包含所有训练结果")

        fold_models_summary = {
            'models': []
        }

        for fold in range(self.train_config.num_folds):
            model_info_path = os.path.join(run_dir, f"model_info_fold_{fold}.json")
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)

            fold_models_summary['models'].append({
                'fold': fold,
                'model_path': f"best_model_fold_{fold}.pth",
                'info': model_info
            })

        # 保存模型汇总信息
        summary_path = os.path.join(run_dir, "models_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(fold_models_summary, f, indent=2)

        self.log(f"所有折模型信息已保存到 {summary_path}")

        return avg_metrics

    def _train_fold(self, model, train_loader, val_loader, test_loader, fold):
        """训练单折模型"""
        # 准备路径 - 修改为使用共享目录
        run_dir = self.config.get_run_dir()

        # 将模型移至设备
        device = self.train_config.training_device
        model = model.to(device)

        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay
        )

        # 损失函数 - MSE用于回归
        criterion = nn.MSELoss()

        # 学习率调度器
        num_training_steps = len(train_loader) * self.train_config.epochs
        num_warmup_steps = int(num_training_steps * self.train_config.warmup_ratio)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.train_config.lr,
            total_steps=num_training_steps,
            pct_start=self.train_config.warmup_ratio,
            div_factor=10.0,
            final_div_factor=1000.0,
            anneal_strategy="linear"
        )

        # 梯度缩放器（用于混合精度训练）
        scaler = GradScaler() if self.train_config.use_amp else None

        # 记录最佳模型 - 修改为使用pearson
        best_pearson = -1.0  # Pearson范围从-1到1，初始化为最低值
        best_epoch = -1
        no_improve = 0

        # 新增：存储每个epoch的结果
        epoch_results = []

        # 训练循环
        for epoch in range(self.train_config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.train_config.epochs}")
            for batch in progress_bar:
                # 将数据移至设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                # 前向传播 - 使用混合精度
                optimizer.zero_grad()

                if self.train_config.use_amp:
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs, batch['labels'])

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if self.train_config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config.grad_clip)

                    # 更新优化器
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 不使用混合精度
                    outputs = model(batch)
                    loss = criterion(outputs, batch['labels'])

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    if self.train_config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config.grad_clip)

                    # 更新优化器
                    optimizer.step()

                # 更新学习率
                scheduler.step()

                # 累计损失
                train_loss += loss.item()
                train_steps += 1

                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})

            # 计算训练集平均损失
            train_loss = train_loss / train_steps

            # 验证阶段
            val_metrics = self._evaluate_model(model, val_loader, criterion, device)
            val_loss = val_metrics['loss']

            # 新增：测试集评估
            test_metrics = self._evaluate_model(model, test_loader, criterion, device,
                                        apply_temp_range=self.train_config.temp_range)

            # 记录日志
            self.log(f"Epoch {epoch+1}/{self.train_config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val RMSE: {val_metrics['rmse']:.4f}, "
                    f"Val R²: {val_metrics['r2']:.4f}, "
                    f"Val Pearson: {val_metrics['pearson']:.4f}, "
                    f"Test RMSE: {test_metrics['rmse']:.4f}, "
                    f"Test R²: {test_metrics['r2']:.4f}")

            # 新增：保存每个epoch的结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation': val_metrics,
                'test': test_metrics
            }
            epoch_results.append(epoch_result)

            # 保存当前epoch的结果到单独的文件
            epoch_result_path = os.path.join(run_dir, f"fold_{fold}_epoch_{epoch+1}_metrics.json")
            with open(epoch_result_path, 'w') as f:
                json.dump(self._serialize_metrics(epoch_result), f, indent=2)

            # 修改：使用pearson相关系数作为主要评估指标
            if val_metrics['pearson'] > best_pearson:
                best_pearson = val_metrics['pearson']
                best_epoch = epoch
                no_improve = 0

                # 保存最佳模型 - 修改路径以保存在同一文件夹
                model_path = os.path.join(run_dir, f"best_model_fold_{fold}.pth")
                model.save_model(model_path)

                # 保存模型的参数和结构信息，便于后续加载
                model_info = {
                    'fold': fold,
                    'epoch': epoch + 1,
                    'pearson': float(best_pearson),
                    'rmse': float(val_metrics['rmse']),
                    'r2': float(val_metrics['r2']),
                    'model_type': self.train_config.train_mode,
                    'fusion_type': self.train_config.fusion_type if self.train_config.train_mode == 'fusion' else None,
                    'hidden_dim': self.train_config.hidden_dim,
                    'dropout': self.train_config.dropout,
                    'enabled_models': [name for name, config in self.model_configs.items() if config.enabled]
                }

                # 保存模型信息到JSON文件
                model_info_path = os.path.join(run_dir, f"model_info_fold_{fold}.json")
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)

                self.log(f"保存新的最佳模型 (Fold {fold}, Epoch {epoch+1}, Pearson: {best_pearson:.4f})")
            else:
                no_improve += 1

            # 早停
            if no_improve >= self.train_config.patience:
                self.log(f"早停: {no_improve}个epoch没有改善")
                break

        # 加载最佳模型进行测试
        best_model_path = os.path.join(run_dir, f"best_model_fold_{fold}.pth")
        if self.train_config.train_mode == "fusion":
            if self.train_config.fusion_type == "weighted":
                model = WeightedFusionRegressor.load_model(best_model_path, self.model_configs)
            else:
                model = FusionModelRegressor.load_model(best_model_path, self.model_configs)
        else:
            model = SingleModelRegressor.load_model(best_model_path)

        model = model.to(device)

        # 评估最佳模型
        self.log(f"评估最佳模型 (Fold {fold}, Epoch {best_epoch+1}, Pearson: {best_pearson:.4f})")
        val_metrics = self._evaluate_model(model, val_loader, criterion, device,
                                        apply_temp_range=self.train_config.temp_range)
        test_metrics = self._evaluate_model(model, test_loader, criterion, device,
                                        apply_temp_range=self.train_config.temp_range)

        # 记录评估指标
        self.log(f"验证集指标: {val_metrics}")
        self.log(f"测试集指标: {test_metrics}")

        # 可视化预测结果
        self._plot_predictions(model, test_loader, run_dir, device, fold)

        # 保存评估指标
        metrics = {
            'fold': fold,
            'best_epoch': best_epoch,
            'validation': val_metrics,
            'test': test_metrics,
            'epoch_results': epoch_results
        }

        metrics_path = os.path.join(run_dir, f"metrics_fold_{fold}.json")
        with open(metrics_path, 'w') as f:
            json.dump(self._serialize_metrics(metrics), f, indent=2)

        return metrics

    # 新增：序列化指标的辅助方法
    def _serialize_metrics(self, metrics):
        """将指标序列化为可JSON化的格式"""
        import numpy as np

        if isinstance(metrics, dict):
            return {k: self._serialize_metrics(v) for k, v in metrics.items()}
        elif isinstance(metrics, list):
            return [self._serialize_metrics(i) for i in metrics]
        elif isinstance(metrics, np.integer):
            return int(metrics)
        elif isinstance(metrics, np.floating):
            return float(metrics)
        elif isinstance(metrics, np.ndarray):
            return metrics.tolist()
        else:
            return metrics

    def train_single(self):
        """单次训练，使用整个训练集训练，整个测试集(s2c2_0_test.csv)验证"""
        # 准备路径
        run_dir = self.config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        # 保存初始配置
        config_path = os.path.join(run_dir, "config.json")
        self.config.save_config(config_path)

        # 在训练前预提取所有特征
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            self.log(f"检测到只有{gpu_count}个GPU，启用特征预提取模式并使用同一GPU进行训练")
            # 确保训练设备与特征提取设备一致
            self.train_config.training_device = self.train_config.feature_extraction_device
            use_preextracted = False
            train_features, test_features = None, None
        else:
            self.log(f"检测到 {gpu_count} 个GPU，使用标准训练模式")
            train_features, val_features = None, None
            use_preextracted = False

        # 创建数据集
        train_dataset = ProteinFeatureDataset(
            self.train_df,
            self.feature_manager,
            self.config,
            self.train_config.target_column,
            self.train_config.sequence_column,
            self.train_config.feature_cache_size,
            self.logger,
            pre_extracted_features=train_features if use_preextracted else None
        )

        val_dataset = ProteinFeatureDataset(
            self.test_df,  # 这里使用test_df作为验证集
            self.feature_manager,
            self.config,
            self.train_config.target_column,
            self.train_config.sequence_column,
            self.train_config.feature_cache_size,
            self.logger,
            pre_extracted_features=val_features if use_preextracted else None
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            collate_fn=collate_protein_features,
            num_workers=self.train_config.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=collate_protein_features,
            num_workers=self.train_config.num_workers
        )

        # 创建模型
        model = self._create_model()

        # 训练模型
        self.log("开始单次训练...")
        metrics = self._train_single_model(model, train_loader, val_loader, run_dir)

        # 打印结果
        self.log(f"训练完成，验证集指标: {metrics['validation']}")

        # 保存结果
        results_path = os.path.join(run_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(self._serialize_metrics(metrics), f, indent=2)

        self.log(f"训练结果已保存到 {results_path}")

        # 更新配置文件
        additional_data = {'metrics': self._serialize_metrics(metrics)}
        self.config.save_config(config_path, additional_data)

        return metrics

    def _train_single_model(self, model, train_loader, val_loader, run_dir):
        """单次训练模型，验证使用s2c2_0_test.csv"""
        # 将模型移至设备
        device = self.train_config.training_device
        model = model.to(device)

        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay
        )

        # 损失函数 - MSE用于回归
        criterion = nn.MSELoss()

        # 学习率调度器
        num_training_steps = len(train_loader) * self.train_config.epochs

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.train_config.lr,
            total_steps=num_training_steps,
            pct_start=self.train_config.warmup_ratio,
            div_factor=10.0,
            final_div_factor=1000.0,
            anneal_strategy="linear"
        )

        # 梯度缩放器（用于混合精度训练）
        scaler = GradScaler() if self.train_config.use_amp else None

        # 记录最佳模型 - 使用pearson相关系数
        best_pearson = -1.0
        best_epoch = -1
        no_improve = 0

        # 存储每个epoch的结果
        epoch_results = []

        # 训练循环
        for epoch in range(self.train_config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.train_config.epochs}")
            for batch in progress_bar:
                # 将数据移至设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                # 前向传播 - 使用混合精度
                optimizer.zero_grad()

                if self.train_config.use_amp:
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs, batch['labels'])

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if self.train_config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config.grad_clip)

                    # 更新优化器
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 不使用混合精度
                    outputs = model(batch)
                    loss = criterion(outputs, batch['labels'])

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    if self.train_config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_config.grad_clip)

                    # 更新优化器
                    optimizer.step()

                # 更新学习率
                scheduler.step()

                # 累计损失
                train_loss += loss.item()
                train_steps += 1

                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})

            # 计算训练集平均损失
            train_loss = train_loss / train_steps

            # 验证阶段 - 使用s2c2_0_test.csv作为验证集
            val_metrics = self._evaluate_model(model, val_loader, criterion, device,
                                        apply_temp_range=self.train_config.temp_range)

            # 记录日志
            self.log(f"Epoch {epoch+1}/{self.train_config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val RMSE: {val_metrics['rmse']:.4f}, "
                    f"Val R²: {val_metrics['r2']:.4f}, "
                    f"Val Pearson: {val_metrics['pearson']:.4f}, "
                    f"Val Spearman: {val_metrics['spearman']:.4f}")

            # 保存每个epoch的结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation': val_metrics
            }
            epoch_results.append(epoch_result)

            # 保存当前epoch的结果到单独的文件
            epoch_result_path = os.path.join(run_dir, f"epoch_{epoch+1}_metrics.json")
            with open(epoch_result_path, 'w') as f:
                json.dump(self._serialize_metrics(epoch_result), f, indent=2)

            # 使用验证集的pearson相关系数作为主要评估指标
            if val_metrics['pearson'] > best_pearson:
                best_pearson = val_metrics['pearson']
                best_epoch = epoch
                no_improve = 0

                # 保存最佳模型
                model_path = os.path.join(run_dir, "best_model.pth")
                model.save_model(model_path)

                # 保存模型信息
                model_info = {
                    'epoch': epoch + 1,
                    'pearson': float(best_pearson),
                    'spearman': float(val_metrics['spearman']),
                    'rmse': float(val_metrics['rmse']),
                    'r2': float(val_metrics['r2']),
                    'model_type': self.train_config.train_mode,
                    'fusion_type': self.train_config.fusion_type if self.train_config.train_mode == 'fusion' else None,
                    'hidden_dim': self.train_config.hidden_dim,
                    'dropout': self.train_config.dropout,
                    'enabled_models': [name for name, config in self.model_configs.items() if config.enabled]
                }

                model_info_path = os.path.join(run_dir, "model_info.json")
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)

                self.log(f"保存新的最佳模型 (Epoch {epoch+1}, Pearson: {best_pearson:.4f})")
            else:
                no_improve += 1

            # 早停
            if no_improve >= self.train_config.patience:
                self.log(f"早停: {no_improve}个epoch没有改善")
                break

        # 加载最佳模型进行最终评估
        best_model_path = os.path.join(run_dir, "best_model.pth")
        if self.train_config.train_mode == "fusion":
            if self.train_config.fusion_type == "weighted":
                model = WeightedFusionRegressor.load_model(best_model_path, self.model_configs)
            else:
                model = FusionModelRegressor.load_model(best_model_path, self.model_configs)
        else:
            model = SingleModelRegressor.load_model(best_model_path)

        model = model.to(device)

        # 最终评估
        self.log(f"评估最佳模型 (Epoch {best_epoch+1}, Pearson: {best_pearson:.4f})")
        val_metrics = self._evaluate_model(model, val_loader, criterion, device,
                                    apply_temp_range=self.train_config.temp_range)

        self.log(f"验证集指标: {val_metrics}")

        # 可视化预测结果
        self._plot_predictions(model, val_loader, run_dir, device)

        # 整理结果
        metrics = {
            'best_epoch': best_epoch + 1,
            'validation': val_metrics,
            'epoch_results': epoch_results
        }

        return metrics

    def cleanup(self):
        """清理资源"""
        self.log("开始清理资源...")

        # 释放预提取特征
        if hasattr(self, 'train_features'):
            self.log("清理训练集特征缓存")
            self.train_features = None

        if hasattr(self, 'test_features'):
            self.log("清理测试集特征缓存")
            self.test_features = None

        # 清理特征提取器资源
        self.feature_manager.cleanup()

        # 手动触发垃圾回收
        import gc
        gc.collect()

        # 清理GPU缓存
        torch.cuda.empty_cache()

        self.log("资源清理完成")

    def _ensemble_predict(self, all_fold_metrics, run_dir):
            """集成不同折的模型进行预测"""
            self.log("开始集成模型预测...")

            # 加载测试集
            test_dataset = ProteinFeatureDataset(
                self.test_df,
                self.feature_manager,
                self.config,
                self.train_config.target_column,
                self.train_config.sequence_column,
                self.train_config.feature_cache_size,
                self.logger
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                collate_fn=collate_protein_features,
                num_workers=self.train_config.num_workers
            )

            # 准备设备
            device = self.train_config.training_device

            # 准备损失函数
            criterion = nn.MSELoss()

            # 加载所有折的最佳模型
            models = []
            fold_metrics_list = []  # 保存每个折模型在测试集上的性能

            for fold in range(self.train_config.num_folds):
                # 修改模型路径，从同一目录加载
                best_model_path = os.path.join(run_dir, f"best_model_fold_{fold}.pth")

                # 加载模型信息
                model_info_path = os.path.join(run_dir, f"model_info_fold_{fold}.json")
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)

                # 根据训练模式加载相应类型的模型
                if self.train_config.train_mode == "fusion":
                    if self.train_config.fusion_type == "weighted":
                        model = WeightedFusionRegressor.load_model(best_model_path, self.model_configs)
                    else:
                        model = FusionModelRegressor.load_model(best_model_path, self.model_configs)
                else:
                    model = SingleModelRegressor.load_model(best_model_path)

                model = model.to(device)
                model.eval()
                models.append(model)

                # 评估单个模型在测试集上的性能
                fold_metrics = self._evaluate_model(model, test_loader, criterion, device,
                                                apply_temp_range=self.train_config.temp_range)
                fold_metrics_list.append(fold_metrics)
                self.log(f"折 {fold} 模型测试指标: RMSE={fold_metrics['rmse']:.4f}, R²={fold_metrics['r2']:.4f}, Pearson={fold_metrics['pearson']:.4f}")

            # 获取每个折模型的测试集评估结果，用于计算加权平均
            fold_weights = []
            if self.train_config.ensemble_strategy == 'weighted':
                # 修改：使用每折模型在验证集上的Pearson作为权重
                for metrics in all_fold_metrics:
                    # 使用Pearson相关系数作为权重，先转换为正值
                    pearson_score = metrics['validation']['pearson']
                    # 确保权重为正，这里我们使用 (pearson + 1)/2 转换到 0-1 范围
                    normalized_weight = (pearson_score + 1)/2
                    fold_weights.append(max(0.01, normalized_weight))  # 确保权重不会太小
                weight_source = "验证集Pearson系数"
            else:
                # 平均策略，所有模型权重相等
                fold_weights = [1.0] * len(models)
                weight_source = "等权重平均"

            # 归一化权重
            sum_weights = sum(fold_weights)
            fold_weights = [w / sum_weights for w in fold_weights]

            # 将权重信息记录到日志并保存
            weight_info = {}
            for i, weight in enumerate(fold_weights):
                weight_info[f"fold_{i}"] = {
                    "权重": float(weight),
                    "验证集Pearson": float(all_fold_metrics[i]['validation']['pearson']),
                    "验证集R²": float(all_fold_metrics[i]['validation']['r2']),
                    "测试集Pearson": float(fold_metrics_list[i]['pearson']),
                    "测试集R²": float(fold_metrics_list[i]['r2']),
                    "测试集RMSE": float(fold_metrics_list[i]['rmse'])
                }

            # 保存权重信息到文件
            weights_path = os.path.join(run_dir, "ensemble_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(weight_info, f, indent=2)

            self.log(f"集成模型权重策略: {self.train_config.ensemble_strategy} ({weight_source})")
            self.log(f"模型集成权重: {[round(w, 4) for w in fold_weights]}")

            # 在测试集上进行集成预测
            all_labels = []
            all_ensemble_preds = []
            fold_predictions = [[] for _ in range(len(models))]  # 存储每个折模型的预测结果

            with torch.no_grad():
                for batch in test_loader:
                    # 将数据移至设备
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)

                    # 收集标签
                    all_labels.extend(batch['labels'].cpu().numpy())

                    # 获取每个模型的预测
                    batch_preds = []
                    for i, model in enumerate(models):
                        outputs = model(batch)
                        fold_pred = outputs.cpu().numpy()
                        fold_predictions[i].extend(fold_pred)  # 保存每个折的原始预测
                        batch_preds.append(fold_pred * fold_weights[i])

                    # 组合预测结果
                    ensemble_preds = np.sum(batch_preds, axis=0)
                    all_ensemble_preds.extend(ensemble_preds)

            # 转换为numpy数组
            all_labels = np.array(all_labels)
            all_ensemble_preds = np.array(all_ensemble_preds)
            fold_predictions = [np.array(preds) for preds in fold_predictions]

            # 后处理预测结果
            if self.train_config.temp_range:
                all_ensemble_preds = self._postprocess_stability_predictions(
                    all_ensemble_preds,
                    temp_range=True
                )
                fold_predictions = [self._postprocess_stability_predictions(preds, temp_range=True)
                                    for preds in fold_predictions]

            # 计算评估指标
            rmse = np.sqrt(mean_squared_error(all_labels, all_ensemble_preds))
            mae = mean_absolute_error(all_labels, all_ensemble_preds)
            r2 = r2_score(all_labels, all_ensemble_preds)

            # 相关系数
            pearson_corr, pearson_p = pearsonr(all_labels, all_ensemble_preds)
            spearman_corr, spearman_p = spearmanr(all_labels, all_ensemble_preds)

            # 可视化集成预测结果，包括各折模型结果对比
            self._plot_ensemble_predictions(all_labels, all_ensemble_preds, fold_predictions,
                                        fold_weights, run_dir)

            # 保存集成预测结果及各折预测结果
            results_df = pd.DataFrame({'actual': all_labels, 'ensemble': all_ensemble_preds})
            for i, preds in enumerate(fold_predictions):
                results_df[f'fold_{i}'] = preds

            # 添加权重信息到dataframe
            for i, weight in enumerate(fold_weights):
                results_df[f'weight_fold_{i}'] = weight

            results_path = os.path.join(run_dir, 'ensemble_predictions.csv')
            results_df.to_csv(results_path, index=False)

            # 返回评估指标
            ensemble_metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson': pearson_corr,
                'pearson_p': pearson_p,
                'spearman': spearman_corr,
                'spearman_p': spearman_p,
                'weights': {f'fold_{i}': float(w) for i, w in enumerate(fold_weights)}
            }

            return ensemble_metrics

    def _plot_ensemble_predictions(self, all_labels, all_preds, fold_predictions, fold_weights, save_dir):
        """可视化集成预测结果，同时展示各折模型的贡献"""
        # 设置中文字体，确保支持中文显示
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 主图：散点图展示实际值vs预测值
        ax1.scatter(all_labels, all_preds, alpha=0.6, label="集成预测", color='darkblue')

        # 添加对角线 (理想预测线)
        min_val = min(min(all_labels), min(all_preds))
        max_val = max(max(all_labels), max(all_preds))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label="理想预测线")

        # 计算相关系数
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        # 添加标题和标签
        ax1.set_title(f'集成模型预测\nR²: {r2:.4f}, Pearson: {pearson_corr:.4f}, RMSE: {rmse:.4f}')
        ax1.set_xlabel('实际热稳定性值')
        ax1.set_ylabel('预测热稳定性值')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # 第二个图：显示各折模型的权重
        colors = plt.cm.viridis(np.linspace(0, 1, len(fold_weights)))
        bars = ax2.bar(range(len(fold_weights)), fold_weights, color=colors, alpha=0.7)

        # 在条形图上添加权重值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            r2_val = r2_score(all_labels, fold_predictions[i])
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fold_weights[i]:.3f}\nR²: {r2_val:.3f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)

        ax2.set_title('各折模型权重分布')
        ax2.set_xlabel('折序号')
        ax2.set_ylabel('权重值')
        ax2.set_xticks(range(len(fold_weights)))
        ax2.set_xticklabels([f'折 {i}' for i in range(len(fold_weights))])
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加模型融合策略说明
        strategy_name = "加权平均 (基于验证集R²)" if any(w != fold_weights[0] for w in fold_weights) else "等权重平均"
        plt.figtext(0.5, 0.01, f"集成策略: {strategy_name}", ha="center", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为底部文字留出空间
        plot_path = os.path.join(save_dir, 'ensemble_predictions.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def _evaluate_model(self, model, data_loader, criterion, device, apply_temp_range=False):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # 将数据移至设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                # 获取标签
                labels = batch['labels']

                # 前向传播
                outputs = model(batch)

                # 计算损失
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                # 收集预测和标签
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = self._postprocess_stability_predictions(
            np.array(all_preds),
            temp_range=apply_temp_range
        )
        # 计算平均损失
        avg_loss = total_loss / len(data_loader.dataset)

        # 计算评估指标
        # all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        # 相关系数
        pearson_corr, pearson_p = pearsonr(all_labels, all_preds)
        spearman_corr, spearman_p = spearmanr(all_labels, all_preds)

        # 返回指标
        return {
            'loss': avg_loss,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'spearman': spearman_corr,
            'spearman_p': spearman_p
        }

    def _calculate_average_metrics(self, metrics_list):
        """计算多折平均指标"""
        avg_metrics = {
            'validation': {},
            'test': {}
        }

        # 各折的测试集指标
        test_metrics_keys = metrics_list[0]['test'].keys()
        for key in test_metrics_keys:
            values = [m['test'][key] for m in metrics_list]
            avg_metrics['test'][key] = sum(values) / len(values)
            avg_metrics['test'][f'{key}_std'] = np.std(values)

        # 各折的验证集指标
        val_metrics_keys = metrics_list[0]['validation'].keys()
        for key in val_metrics_keys:
            values = [m['validation'][key] for m in metrics_list]
            avg_metrics['validation'][key] = sum(values) / len(values)
            avg_metrics['validation'][f'{key}_std'] = np.std(values)

        return avg_metrics

    def _plot_predictions(self, model, data_loader, save_dir, device, fold=None):
        """可视化预测结果"""
        # 设置中文字体，确保支持中文显示
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # 将数据移至设备
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                # 前向传播
                outputs = model(batch)

                # 收集预测和标签
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # 创建散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(all_labels, all_preds, alpha=0.6)

        # 添加对角线 (理想预测线)
        min_val = min(min(all_labels), min(all_preds))
        max_val = max(max(all_labels), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # 计算相关系数
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        # 添加标题和标签
        title = f'预测值 vs 实际值\nR²: {r2:.4f}, Pearson: {pearson_corr:.4f}, RMSE: {rmse:.4f}'
        if fold is not None:
            title = f'折 {fold} - ' + title
        plt.title(title)
        plt.xlabel('实际热稳定性值')
        plt.ylabel('预测热稳定性值')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图表
        plt.tight_layout()
        plot_filename = f'predictions_fold_{fold}.png' if fold is not None else 'predictions.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # 保存预测结果
        results_df = pd.DataFrame({
            'actual': all_labels,
            'predicted': all_preds
        })
        results_filename = f'predictions_fold_{fold}.csv' if fold is not None else 'predictions.csv'
        results_path = os.path.join(save_dir, results_filename)
        results_df.to_csv(results_path, index=False)
    def _postprocess_stability_predictions(self, predictions, temp_range=False):
        """热稳定性预测的后处理函数"""
        # 复制预测结果避免修改原数组
        processed_preds = predictions.copy()

        # 若启用温度范围限制，将预测结果限制在合理范围内
        if temp_range:
            # 大多数蛋白质的热稳定性通常在0-120°C范围内
            processed_preds = np.clip(processed_preds, 0.0, 120.0)

        # 丢弃明显异常值(可选)
        # processed_preds[processed_preds > 150] = np.nan

        return processed_preds
#===============================================================================
# 命令行参数解析
#===============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='蛋白质热稳定性预测训练脚本')

    # 数据参数
    parser.add_argument('--train_csv', type=str, default='s2c2_0_train.csv',
                        help='训练数据CSV文件路径')
    parser.add_argument('--test_csv', type=str, default='s2c2_0_test.csv',
                        help='测试数据CSV文件路径')
    parser.add_argument('--target_column', type=str, default='tgt_reg',
                        help='目标列名')
    parser.add_argument('--sequence_column', type=str, default='sequence',
                        help='序列列名')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='权重衰减')
    parser.add_argument('--max_seq_len', type=int, default=600,
                        help='最大序列长度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比例')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪值')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='数据加载器工作进程数')

    # 模型参数
    parser.add_argument('--train_mode', type=str, default='fusion',
                        choices=['fusion', 'single'],
                        help='训练模式')
    parser.add_argument('--fusion_type', type=str, default='default',
                        choices=['default', 'weighted'],
                        help='融合模型类型')
    parser.add_argument('--use_esm2', action='store_true', default=True,
                        help='使用ESM2模型')
    parser.add_argument('--use_esmc', action='store_true', default=True,
                        help='使用ESM-C模型')
    parser.add_argument('--use_splm', action='store_true', default=False,
                        help='使用S-PLM模型')

    # 路径参数
    parser.add_argument('--model_save_dir', type=str, default='./protein_stability_results',
                        help='模型保存目录')
    parser.add_argument('--experiment_name', type=str, default='protein_stability_experiment',
                        help='实验名称')

    # GPU参数
    parser.add_argument('--feature_gpu', type=int, default=0,
                        help='特征提取使用的GPU ID')
    parser.add_argument('--train_gpu', type=int, default=1,
                        help='训练使用的GPU ID')

    # 其他参数
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='使用混合精度训练')
    parser.add_argument('--normalize_features', action='store_true', default=True,
                        help='归一化特征')
    parser.add_argument('--feature_cache_size', type=int, default=400,
                        help='特征缓存大小')
    parser.add_argument('--temp_range', action='store_true', default=True,
                        help='预测结果是否限制在合理的温度范围内(0-120°C)')
    parser.add_argument('--standardize_target', action='store_true', default=False,
                        help='是否对目标值进行标准化')
    parser.add_argument('--normalize_method', type=str, default='global',
                        choices=['none', 'global', 'sequence', 'layer'],
                        help='特征归一化方法')
    parser.add_argument('--save_features', action='store_true', default=False,
                        help='是否保存提取的特征')
    parser.add_argument('--ensemble_strategy', type=str, default='mean',
                        choices=['mean', 'weighted'],
                        help='集成预测策略')

    return parser.parse_args()


#===============================================================================
# 主程序入口
#===============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建实验配置
    config = ExperimentConfig(args.experiment_name)

    # 更新训练配置
    config.training_config.train_csv = args.train_csv
    config.training_config.test_csv = args.test_csv
    config.training_config.target_column = args.target_column
    config.training_config.sequence_column = args.sequence_column
    config.training_config.batch_size = args.batch_size
    config.training_config.epochs = args.epochs
    config.training_config.lr = args.lr
    config.training_config.weight_decay = args.weight_decay
    config.training_config.max_seq_len = args.max_seq_len
    config.training_config.hidden_dim = args.hidden_dim
    config.training_config.dropout = args.dropout
    config.training_config.grad_clip = args.grad_clip
    config.training_config.num_folds = args.num_folds
    config.training_config.patience = args.patience
    config.training_config.random_seed = args.seed
    config.training_config.num_workers = args.num_workers
    config.training_config.train_mode = args.train_mode
    config.training_config.fusion_type = args.fusion_type
    config.training_config.model_save_dir = args.model_save_dir
    config.training_config.use_amp = args.use_amp
    config.training_config.normalize_features = args.normalize_features
    config.training_config.feature_cache_size = args.feature_cache_size
    config.training_config.normalization_method = args.normalize_method
    config.training_config.save_features = args.save_features
    config.training_config.temp_range = args.temp_range
    config.training_config.standardize_target = args.standardize_target
    config.training_config.ensemble_strategy = args.ensemble_strategy
    config.training_config.feature_gpu = args.feature_gpu
    config.training_config.train_gpu = args.train_gpu

    # 启用/禁用模型
    config.model_configs["esm2"].enabled = args.use_esm2
    config.model_configs["esmc"].enabled = args.use_esmc
    config.model_configs["splm"].enabled = args.use_splm

    # 创建运行目录
    run_dir = config.get_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # 创建日志
    log_path = os.path.join(run_dir, "training.log")
    logger = Logger(log_path, console=True)
    logger.info("开始蛋白质热稳定性预测模型训练")
    logger.info(f"运行目录: {run_dir}")

    # 更新GPU设置
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"检测到 {gpu_count} 个GPU")

        if gpu_count == 1:
            # 只有一个GPU时，特征提取和训练使用同一个GPU
            feature_device = torch.device("cuda:0")
            train_device = torch.device("cuda:0")
            logger.info("单GPU环境: 特征提取和训练将共用同一个GPU (cuda:0)")
        else:
            # 多GPU环境下，尊重用户的GPU选择
            feature_device = torch.device(f"cuda:{args.feature_gpu}")
            train_device = torch.device(f"cuda:{args.train_gpu}")
            logger.info(f"多GPU环境: 特征提取使用 cuda:{args.feature_gpu}，训练使用 cuda:{args.train_gpu}")

        config.training_config.feature_extraction_device = feature_device
        config.training_config.training_device = train_device
    # 保存配置
    config_path = os.path.join(run_dir, "config.json")
    config.save_config(config_path)
    logger.info(f"配置已保存到 {config_path}")
    # 是否标准化目标值
    if args.standardize_target:
        logger.info("启用目标值标准化")

        # 加载训练数据获取均值和标准差
        train_df = pd.read_csv(args.train_csv)
        target_mean = train_df[args.target_column].mean()
        target_std = train_df[args.target_column].std()
        logger.info(f"目标值均值: {target_mean:.4f}, 标准差: {target_std:.4f}")

        # 添加到配置中，供数据集类使用
        config.training_config.target_mean = target_mean
        config.training_config.target_std = target_std
        config.training_config.standardize_target = True
    else:
        config.training_config.standardize_target = False

    # 是否使用温度范围限制
    config.training_config.temp_range = args.temp_range
    if args.temp_range:
        logger.info("启用温度范围限制 (0-120°C)")

    # 是否保存提取的特征
    config.training_config.save_features = args.save_features
    if args.save_features:
        logger.info("启用特征保存，将在运行目录中创建特征缓存")
        feature_cache_dir = os.path.join(run_dir, "feature_cache")
        os.makedirs(feature_cache_dir, exist_ok=True)
        config.training_config.feature_cache_dir = feature_cache_dir

    # 集成预测策略
    config.training_config.ensemble_strategy = args.ensemble_strategy
    logger.info(f"集成预测策略: {args.ensemble_strategy}")
    # 优化：提前创建特征管理器并预加载模型
    logger.info("创建特征管理器并预加载模型")
    feature_manager = FeatureManager(config, logger)
    feature_manager.preload_all_models()
    logger.info("特征提取模型预加载完成")

    # 创建训练器 (不传入预加载的feature_manager)
    trainer = ModelTrainer(config, logger)

    # 手动替换trainer中的特征管理器
    trainer.feature_manager = feature_manager

    logger.info("开始K折交叉验证训练")
    metrics = trainer.train_single()

    # 打印结果
    logger.info("训练完成")
    logger.info(f"验证集RMSE: {metrics['validation']['rmse']:.4f}")
    logger.info(f"验证集R²: {metrics['validation']['r2']:.4f}")
    logger.info(f"验证集Pearson相关系数: {metrics['validation']['pearson']:.4f}")
    logger.info(f"验证集Spearman相关系数: {metrics['validation']['spearman']:.4f}")

    # 清理资源
    trainer.cleanup()
    logger.info("已清理资源")

if __name__ == "__main__":
    main()