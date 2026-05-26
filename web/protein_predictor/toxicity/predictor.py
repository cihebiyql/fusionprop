"""
蛋白质毒性预测核心功能
"""
import os
import json
import torch
import numpy as np
import logging
import time
import pandas as pd
import shutil
from pathlib import Path
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from .config import PredictorConfig, ModelConfig
from .model import WeightedFusionClassifier
from ..common.data import FeatureDataset, feature_collate_fn, extract_features
from .utils import interpret_toxicity, logger
from ..common.utils import set_weights_only_warning
class ToxicityPredictor:
    """蛋白质毒性预测器，使用集成模型进行预测"""
    
    def __init__(self, config=None, model_dir=None):
        """初始化预测器"""
        # 初始化配置
        self.config = config if config is not None else PredictorConfig()
        if model_dir:
            self.config.model_dir = model_dir
            
        # 设置设备和随机种子
        self.device = self.config.device
        self.config.set_seed()
        
        # 初始化模型列表和路径
        self.models = []
        self.model_paths = []
        
        # 加载配置
        self._load_model_config()
        
        # 加载特征统计数据
        self._load_feature_stats()
        
        # 加载模型
        self._load_models()
        
        logger.info(f"毒性预测器已初始化，设备: {self.device}, 已加载 {len(self.models)} 个模型")
        
    def _load_model_config(self):
        """加载模型配置"""
        config_path = Path(self.config.model_dir) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                
                # 创建模型配置
                self.model_configs = {}
                
                # 从model_configs键读取配置
                for model_name, model_config in config_data.get("model_configs", {}).items():
                    self.model_configs[model_name] = ModelConfig(**model_config)
                
                logger.info(f"已加载模型配置: {len(self.model_configs)} 个模型")
            except Exception as e:
                logger.warning(f"无法加载模型配置: {e}")
                # 使用默认配置
                self.model_configs = {
                    "esm2": ModelConfig(output_dim=1280),
                    "esmc": ModelConfig(output_dim=1152)
                }
        else:
            logger.warning(f"模型配置文件不存在: {config_path}，使用默认配置")
            # 使用默认配置
            self.model_configs = {
                "esm2": ModelConfig(output_dim=1280),
                "esmc": ModelConfig(output_dim=1152)
            }
    
    def _load_feature_stats(self):
        """加载特征统计数据（均值、标准差）"""
        stats_path = Path(self.config.model_dir) / "feature_stats.json"
        
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                
                self.config.esm2_mean = stats.get("esm2_mean", 0.0)
                self.config.esm2_std = stats.get("esm2_std", 1.0)
                self.config.esmc_mean = stats.get("esmc_mean", 0.0)
                self.config.esmc_std = stats.get("esmc_std", 1.0)
                
                logger.info(f"已加载特征统计数据: ESM2 μ={self.config.esm2_mean:.4f}, σ={self.config.esm2_std:.4f}, "
                          f"ESMC μ={self.config.esmc_mean:.4f}, σ={self.config.esmc_std:.4f}")
            except Exception as e:
                logger.warning(f"无法加载特征统计数据: {e}")
        else:
            logger.warning(f"特征统计文件不存在: {stats_path}，将使用默认值")
    
    def _load_models(self):
        """加载所有保存的模型"""
        # 忽略torch.load的权重警告
        set_weights_only_warning()
        
        model_dir = Path(self.config.model_dir)
        logger.info(f"从 {model_dir} 加载训练好的模型...")
        
        # 优先加载 GitHub 归档的 toxicity ensemble；不存在时保持旧的 best_model.pt 回退。
        ensemble_dir = model_dir / "ensemble_20250513_105204"
        ensemble_files = sorted(ensemble_dir.glob("*.pt")) if ensemble_dir.exists() else []

        if ensemble_files:
            self.model_paths = ensemble_files
            logger.info(
                f"找到毒性集成模型目录: {ensemble_dir.name}, "
                f"将加载 {len(self.model_paths)} 个 checkpoint"
            )
        else:
            model_file = model_dir / "best_model.pt"
            if not model_file.exists():
                # Fallback to looking for any .pt file if best_model.pt is not found
                model_files_glob = sorted(model_dir.glob("*.pt"))
                if not model_files_glob:
                    error_msg = f"在 {model_dir} 中未找到 ensemble checkpoint、best_model.pt 或任何 .pt 模型文件"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                model_file = model_files_glob[0] # Take the first one found
                logger.warning(f"best_model.pt 未找到，回退加载: {model_file.name}")

            self.model_paths = [model_file]
            logger.warning("未找到 toxicity ensemble，使用单模型回退路径")

        logger.info(f"找到模型文件: {[path.name for path in self.model_paths]}")
        
        # 从配置文件读取hidden_dim和dropout
        config_path = Path(self.config.model_dir) / "config.json"
        hidden_dim = 768  # 默认使用训练时的hidden_dim
        dropout = 0.5     # 默认使用训练时的dropout
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    # 从训练配置中读取hidden_dim和dropout
                    train_config = config_data.get("training_config", {})
                    hidden_dim = train_config.get("hidden_dim", 768)
                    dropout = train_config.get("dropout", 0.5)
                    logger.info(f"从配置文件读取模型参数: hidden_dim={hidden_dim}, dropout={dropout}")
            except Exception as e:
                logger.warning(f"读取配置文件参数失败: {e}，使用默认值 hidden_dim={hidden_dim}, dropout={dropout}")
        
        # 加载模型
        for path in self.model_paths:
            try:
                # 加载保存的模型对象
                checkpoint = torch.load(path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # 提取model_state_dict
                    state_dict = checkpoint["model_state_dict"]
                    
                    # 提取model_configs (使用checkpoint中的，如果存在)
                    model_configs_from_checkpoint = checkpoint.get("model_configs", {})
                    
                    # 尝试从训练配置中获取模型配置
                    # 这部分逻辑参考了 train_12_2.py 的 ModelTrainer._create_model 和 FusionModelClassifier.load_model
                    
                    # 首先，尝试从 checkpoint 的 'model_configs' 加载
                    # 如果 checkpoint 中没有 'model_configs'，则使用 self.model_configs (从 predictor 的 config.json 加载的)
                    # 如果 self.model_configs 也没有（例如，旧的 predictor config.json），则需要一个默认或从 checkpoint 的其他信息推断
                    
                    current_model_configs_to_use = {}
                    if model_configs_from_checkpoint:
                        for name, cfg_dict in model_configs_from_checkpoint.items():
                            # Create ModelConfig instances from the dictionary in the checkpoint
                            mc = ModelConfig(model_name=name, output_dim=cfg_dict.get('output_dim'))
                            mc.enabled = cfg_dict.get('enabled', True) # Assume enabled if not specified
                            current_model_configs_to_use[name] = mc
                        logger.info(f"从模型checkpoint的 'model_configs' 构建了配置: {list(current_model_configs_to_use.keys())}")
                    elif self.model_configs: # these are loaded by _load_model_config from predictor's config.json
                        current_model_configs_to_use = self.model_configs
                        logger.info(f"使用从 PredictorConfig 加载的 model_configs: {list(current_model_configs_to_use.keys())}")
                    else:
                        # Fallback if no model_configs found in checkpoint or predictor config
                        # This might happen with very old models or if config.json is missing/malformed
                        logger.warning(f"在模型checkpoint或PredictorConfig中未找到 'model_configs'。将尝试使用默认配置。")
                        # As a last resort, create default ESM2/ESMC configs if model names suggest them
                        # This is a heuristic and might need adjustment based on actual model structure
                        # For 'best_model.pt' which was trained by train_12_2.py, it should have 'model_configs'
                        # So this path should ideally not be hit.
                        default_esm2_cfg = ModelConfig(model_name='esm2', output_dim=1280, enabled=True)
                        default_esmc_cfg = ModelConfig(model_name='esmc', output_dim=1152, enabled=True)
                        # A simple heuristic: if the state_dict suggests ESM-C and ESM2, use those
                        # This is hard to determine reliably without more info from the checkpoint.
                        # For now, we rely on the checkpoint having 'model_configs'
                        # If not, we need to ensure self.model_configs (from predictor's config.json) is valid
                        # The train_12_2.py saves 'model_configs' in the checkpoint.
                        if not current_model_configs_to_use: # Should not happen if model is from train_12_2.py
                             logger.error("无法确定模型配置。请确保模型checkpoint包含'model_configs'或Predictor的config.json有效。")
                             raise ValueError("无法确定模型配置。")


                    # 使用从模型checkpoint中获取的hidden_dim，如果存在，否则使用从config.json读取的
                    h_dim = checkpoint.get("hidden_dim", hidden_dim)
                    # 获取dropout，优先从checkpoint，然后是config.json的train_config，最后是默认值
                    d_out = checkpoint.get("dropout", dropout) # Assuming 'dropout' might be in checkpoint
                                        
                    # 创建与训练时相同结构的模型
                    # The model saved by train_12_2.py (WeightedFusionClassifier) requires model_configs
                    model = WeightedFusionClassifier(
                        model_configs=current_model_configs_to_use,
                        hidden_dim=h_dim,
                        dropout=d_out # Use dropout from checkpoint or config
                    )
                    
                    # 加载权重
                    model.load_state_dict(state_dict)
                    
                    # 将模型移动到设备上并设置为评估模式
                    model = model.to(self.device)
                    model.eval()
                    
                    self.models.append(model)
                    logger.info(f"成功加载模型: {path.name}")
                else:
                    logger.warning(f"模型 {path} 不包含预期的结构，跳过")
                    
            except Exception as e:
                logger.error(f"加载模型 {path} 失败: {str(e)}")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
        
        # 验证是否有加载成功的模型
        if not self.models:
            raise ValueError("没有成功加载任何模型。请检查模型文件格式是否兼容。")

    def predict_batch(self, 
                    esm2_features_dir: str, 
                    esmc_features_dir: str, 
                    sample_ids: Optional[List[str]] = None,
                    return_confidence: bool = False) -> Dict:
        """对一批样本进行毒性预测

        Args:
            esm2_features_dir: ESM2 特征文件目录
            esmc_features_dir: ESMC 特征文件目录
            sample_ids: 样本 ID 列表 (可选, 如果为None, 则处理目录中所有.pt文件)
            return_confidence: 是否返回 ensemble 模型间标准差作为置信度来源

        Returns:
            Dict: 包含预测结果的字典，格式为 {"prediction_map": {sample_id: prediction_details}, "mean_toxicity_prob": float}
        """
        logger.info(f"[ToxicityPredictor.predict_batch] Called with esm2_dir: '{esm2_features_dir}', esmc_dir: '{esmc_features_dir}', sample_ids: {sample_ids}")

        if not self.models:
            logger.error("[ToxicityPredictor.predict_batch] No models loaded, cannot predict.")
            return {"prediction_map": {}, "mean_toxicity_prob": None}

        try:
            # 创建数据集和数据加载器
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Creating FeatureDataset with esm2_dir='{esm2_features_dir}', esmc_dir='{esmc_features_dir}'...")
            dataset = FeatureDataset(
                esm2_dir=esm2_features_dir,
                esmc_dir=esmc_features_dir,
                config=self.config,
                sample_ids=sample_ids,
            )
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] FeatureDataset created. Length: {len(dataset)}")

            if len(dataset) == 0:
                logger.warning(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] FeatureDataset is empty. No features found for the given sample_ids or in the directories.")
                return {"prediction_map": {}, "mean_toxicity_prob": None}

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=feature_collate_fn,
                pin_memory=True
            )
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] DataLoader created.")

            # 存储每个批次的 ensemble 均值和模型间标准差
            all_batch_mean_probs = []
            all_batch_std_probs = []
            all_sample_ids_processed = [] # To store the actual IDs processed by DataLoader

            # 预测
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dataloader):
                    logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Processing batch {batch_idx + 1}/{len(dataloader)}")
                    batch_esm2_features = batch_data["esm2_features"].to(self.device)
                    batch_esmc_features = batch_data["esmc_features"].to(self.device)
                    batch_sample_ids = batch_data["sample_id"]
                    all_sample_ids_processed.extend(batch_sample_ids)
                    logger.debug(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Batch sample_ids: {batch_sample_ids}")
                    logger.debug(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Batch ESM2 features shape: {batch_esm2_features.shape}, device: {batch_esm2_features.device}")
                    logger.debug(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Batch ESMC features shape: {batch_esmc_features.shape}, device: {batch_esmc_features.device}")

                    batch_model_probs = []
                    for model_idx, model in enumerate(self.models, start=1):
                        with autocast(enabled=self.config.use_amp):
                            outputs = model(esm2_features=batch_esm2_features, esmc_features=batch_esmc_features)
                            probs = torch.sigmoid(outputs).detach().float().view(-1).cpu().numpy()
                        batch_model_probs.append(probs)
                        logger.debug(
                            f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] "
                            f"Model {model_idx}/{len(self.models)} raw outputs shape: {outputs.shape}, probs shape: {probs.shape}"
                        )

                    batch_probs = np.stack(batch_model_probs, axis=0) # (num_models, batch_size)
                    all_batch_mean_probs.append(batch_probs.mean(axis=0))
                    all_batch_std_probs.append(batch_probs.std(axis=0))

            if not all_batch_mean_probs:
                logger.warning(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] No model probabilities were generated.")
                return {"prediction_map": {}, "mean_toxicity_prob": None}

            # 合并所有批次的 ensemble 预测概率
            mean_probs = np.concatenate(all_batch_mean_probs, axis=0) # (total_samples,)
            std_probs = np.concatenate(all_batch_std_probs, axis=0) # (total_samples,)
            logger.info(
                f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] "
                f"Calculated ensemble probabilities from {len(self.models)} model(s). "
                f"Shape: {mean_probs.shape}. Total processed sample IDs: {len(all_sample_ids_processed)}"
            )

            # 构建预测结果映射
            prediction_map = {}
            for i, sample_id_processed in enumerate(all_sample_ids_processed):
                prob = float(mean_probs[i])
                is_toxic = prob >= self.config.threshold
                prediction = {
                    "toxicity_prob": prob,
                    "is_toxic": is_toxic,
                }
                if return_confidence and len(self.models) > 1:
                    prediction["confidence"] = float(std_probs[i])
                prediction_map[sample_id_processed] = prediction
                # logger.debug(f"[{sample_id_processed}] Processed prediction: Prob={prob:.4f}, IsToxic={is_toxic}")

            overall_mean_toxicity_prob = float(np.mean(mean_probs)) if mean_probs.size > 0 else None
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Final prediction map generated with {len(prediction_map)} entries. Overall mean prob: {overall_mean_toxicity_prob}")
            
            return {
                "prediction_map": prediction_map,
                "mean_toxicity_prob": overall_mean_toxicity_prob
            }
        
        except FileNotFoundError as fnf_error:
            logger.error(f"[ToxicityPredictor.predict_batch] FileNotFoundError: {fnf_error}. This might indicate missing feature files for some sample IDs.")
            return {"prediction_map": {}, "mean_toxicity_prob": None, "error": str(fnf_error)}
        except Exception as e:
            logger.error(f"[ToxicityPredictor.predict_batch] Error during prediction: {e}", exc_info=True)
            return {"prediction_map": {}, "mean_toxicity_prob": None, "error": str(e)}

def predict_toxicity(
    sequence: str, 
    return_confidence: bool = False,
    model_dir: Optional[str] = None,
    cleanup_features: bool = True
) -> Dict:
    """从蛋白质序列直接预测毒性
    
    Args:
        sequence: 蛋白质序列字符串
        return_confidence: 是否返回预测置信度
        model_dir: 模型目录，None则使用默认
        cleanup_features: 完成后是否删除临时特征文件
        
    Returns:
        Dict: 包含毒性预测结果的字典
    """
    try:
        # 1. 提取特征
        temp_dir = Path("./temp_features")
        feature_result = extract_features(sequence, output_dir=str(temp_dir))
        
        # 验证特征提取结果
        if not isinstance(feature_result, dict):
            raise TypeError(f"特征提取结果应为字典，实际为 {type(feature_result)}")
            
        if "sample_id" not in feature_result:
            # 如果找不到sample_id，手动生成一个
            logging.warning("特征提取结果中未找到sample_id，使用自动生成的ID")
            sample_id = f"sample_{hash(sequence) % 10000:04d}"
        else:
            sample_id = feature_result["sample_id"]
        
        # 2. 初始化预测器
        config = PredictorConfig()
        if model_dir:
            config.model_dir = model_dir
        predictor = ToxicityPredictor(config)
        
        # 3. 从特征预测
        prediction = predictor.predict_batch(
            esm2_features_dir=str(temp_dir / "esm2_features"),
            esmc_features_dir=str(temp_dir / "esmc_features"),
            sample_ids=[sample_id],
            return_confidence=return_confidence
        )
        
        # 4. 整理结果
        if "prediction_map" not in prediction or sample_id not in prediction["prediction_map"]:
            raise ValueError(f"预测失败，未找到样本 {sample_id} 的预测结果")
            
        sample_prediction = prediction["prediction_map"][sample_id]
        toxicity_prob = sample_prediction["toxicity_prob"]
        is_toxic = sample_prediction["is_toxic"]
        
        # 解释预测结果
        interpretation = interpret_toxicity(toxicity_prob)
        
        result = {
            "sequence": sequence,
            "toxicity_probability": toxicity_prob,
            "is_toxic": is_toxic,
            "risk_level": interpretation["risk_level"],
            "description": interpretation["description"],
            "classification": interpretation["classification"]
        }
        
        # 添加置信度（如果有）
        if return_confidence and "confidence" in sample_prediction:
            # 获取原始置信度（标准差）
            raw_confidence = float(sample_prediction["confidence"])
            
            # 归一化置信度到0-1区间
            # 标准差越低，置信度越高
            normalized_confidence = max(0.0, 1.0 - min(raw_confidence * 5, 1.0))
            
            # 保存归一化的置信度和原始标准差
            result["confidence"] = normalized_confidence
            result["std_dev"] = raw_confidence
            
            # 添加置信度解释
            if normalized_confidence >= 0.8:
                result["confidence_level"] = "非常高"
            elif normalized_confidence >= 0.6:
                result["confidence_level"] = "高"
            elif normalized_confidence >= 0.4:
                result["confidence_level"] = "中等"
            elif normalized_confidence >= 0.2:
                result["confidence_level"] = "低"
            else:
                result["confidence_level"] = "非常低"
        
        # 5. 清理临时文件
        if cleanup_features and temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        return result
        
    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        return {"error": str(e), "sequence": sequence}

def batch_predict(sequences, names=None, with_confidence=True, export_csv=None):
    """批量预测多个蛋白质序列的毒性
    
    Args:
        sequences: 蛋白质序列列表
        names: 蛋白质名称列表（可选），如果不提供则使用序列索引
        with_confidence: 是否返回置信度
        export_csv: 导出结果到CSV文件的路径（可选）
        
    Returns:
        预测结果列表
    """
    if names is None:
        names = [f"蛋白质_{i+1}" for i in range(len(sequences))]
    
    results = []
    print(f"\n==== 开始批量预测 {len(sequences)} 个序列的毒性 ====")
    
    for i, (name, seq) in enumerate(zip(names, sequences)):
        print(f"\n[{i+1}/{len(sequences)}] 预测 {name} ({len(seq)} aa) 的毒性")
        start_time = time.time()
        
        # 进行预测
        result = predict_toxicity(seq, return_confidence=with_confidence)
        
        # 记录执行时间
        elapsed = time.time() - start_time
        
        # 处理结果
        if 'error' in result:
            print(f"预测出错: {result['error']}")
            result_info = {
                'name': name,
                'sequence': seq[:20] + "..." if len(seq) > 20 else seq,
                'error': result['error']
            }
        else:
            print(f"毒性概率: {result['toxicity_probability']:.2f}")
            print(f"预测结果: {result['classification']}")
            print(f"风险级别: {result['risk_level']}")
            if with_confidence and 'confidence' in result:
                print(f"置信度: {result['confidence']:.2f} ({result['confidence_level']})")
                print(f"标准差: {result['std_dev']:.2f}")
            print(f"解释: {result['description']}")
            print(f"耗时: {elapsed:.2f}秒")
            
            result_info = {
                'name': name,
                'sequence': seq[:20] + "..." if len(seq) > 20 else seq,
                'toxicity_probability': result['toxicity_probability'],
                'is_toxic': result['is_toxic'],
                'risk_level': result['risk_level'],
                'description': result['description']
            }
            
            if with_confidence and 'confidence' in result:
                result_info['confidence'] = result['confidence']
                result_info['confidence_level'] = result['confidence_level']
                result_info['std_dev'] = result['std_dev']
        
        results.append(result_info)
    
    # 导出到CSV（如果需要）
    if export_csv:
        try:
            df = pd.DataFrame(results)
            df.to_csv(export_csv, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已导出到: {export_csv}")
        except Exception as e:
            print(f"导出CSV失败: {str(e)}")
    
    return results