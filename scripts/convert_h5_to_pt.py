#!/usr/bin/env python3
"""
H5 to PT Converter Script
将H5文件中的features提取并转换为PyTorch .pt文件

使用方法:
python convert_h5_to_pt.py --input_dir /path/to/h5/files --output_dir /path/to/output
"""

import os
import h5py
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging(log_file='convert_h5_to_pt.log', verbose=False):
    """设置日志配置"""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.CRITICAL,  # 只显示严重错误
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    return logging.getLogger(__name__)

def extract_sample_name(h5_filename):
    """
    从H5文件名中提取样本名称
    例子: TCGA-2J-AAB4-01Z-00-DX1.480BBC89-87E5-4F6B-B0E9-A6A8A7F4DB5E.h5
    提取: TCGA-2J-AAB4-01Z-00-DX1
    """
    # 移除.h5扩展名
    name_without_ext = h5_filename.replace('.h5', '')
    
    # 按点分割，取第一部分（TCGA样本ID部分）
    parts = name_without_ext.split('.')
    sample_name = parts[0]
    
    return sample_name

def convert_h5_to_pt(input_file_path, output_file_path, verbose):
    """
    将单个H5文件转换为PT文件
    
    Args:
        input_file_path (str): 输入H5文件路径
        output_file_path (str): 输出PT文件路径
        verbose (bool): 是否输出详细日志
    
    Returns:
        bool: 转换是否成功
        dict: 包含特征信息的字典
    """
    try:
        # 打开H5文件
        with h5py.File(input_file_path, 'r') as h5_file:
            # 检查是否包含features键
            if 'features' not in h5_file.keys():
                if verbose:
                    logging.warning(f"文件 {input_file_path} 中没有找到 'features' 键")
                    logging.info(f"可用键: {list(h5_file.keys())}")
                return False, {}
            
            # 提取features数据
            features = h5_file['features'][:]
            
            # 获取其他信息（如果存在）
            info = {
                'features': features,
                'original_file': os.path.basename(input_file_path)
            }
            
            # 尝试提取坐标信息
            if 'coords' in h5_file.keys():
                coords = h5_file['coords'][:]
                info['coords'] = coords
                if verbose:
                    logging.info(f"提取坐标信息: {coords.shape}")
            
            if 'coords_patching' in h5_file.keys():
                coords_patching = h5_file['coords_patching'][:]
                info['coords_patching'] = coords_patching
                if verbose:
                    logging.info(f"提取patching坐标信息: {coords_patching.shape}")
            
            if 'annots' in h5_file.keys():
                annots = h5_file['annots'][:]
                info['annots'] = annots
                if verbose:
                    logging.info(f"提取标注信息: {annots.shape}")
            
            if verbose:
                logging.info(f"特征形状: {features.shape}")
                logging.info(f"特征数据类型: {features.dtype}")
            
        # 转换为PyTorch张量
        features_tensor = torch.from_numpy(features)
        
        # 保存为PT文件
        torch.save(features_tensor, output_file_path)
        
        if verbose:
            logging.info(f"成功转换: {input_file_path} -> {output_file_path}")
            logging.info(f"保存的张量形状: {features_tensor.shape}")
        
        return True, {
            'features_shape': features_tensor.shape,
            'features_dtype': features_tensor.dtype,
            'coords_available': 'coords' in info,
            'file_size_mb': os.path.getsize(output_file_path) / (1024 * 1024)
        }
        
    except Exception as e:
        logging.error(f"转换失败 {input_file_path}: {str(e)}")
        return False, {}

def batch_convert_h5_to_pt(input_dir, output_dir, pattern="*.h5", verbose=False):
    """
    批量转换H5文件到PT文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        pattern (str): 文件匹配模式
        verbose (bool): 是否输出详细日志
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有H5文件
    h5_files = list(input_path.glob(pattern))
    
    if not h5_files:
        if verbose:
            logging.warning(f"在目录 {input_dir} 中没有找到匹配 {pattern} 的文件")
        return
    
    if verbose:
        logging.info(f"找到 {len(h5_files)} 个H5文件待转换")
    
    # 统计信息
    success_count = 0
    failed_count = 0
    total_size_mb = 0
    
    # 批量转换
    for h5_file in tqdm(h5_files, desc="转换H5文件"):
        # 提取样本名称
        # sample_name = extract_sample_name(h5_file.name)
        # pt_filename = f"{sample_name}.pt"
        pt_filename = h5_file.name
        
        output_file_path = output_path / pt_filename
        
        if verbose:
            logging.info(f"正在处理: {h5_file.name} -> {pt_filename}")
        
        # 执行转换
        success, info = convert_h5_to_pt(str(h5_file), str(output_file_path), verbose)
        
        if success:
            success_count += 1
            total_size_mb += info.get('file_size_mb', 0)
            if verbose:
                logging.info(f"✅ 转换成功: {pt_filename}")
                
                # 打印特征信息
                if 'features_shape' in info:
                    logging.info(f"   特征形状: {info['features_shape']}")
                    logging.info(f"   文件大小: {info['file_size_mb']:.2f} MB")
        else:
            failed_count += 1
            if verbose:
                logging.error(f"❌ 转换失败: {h5_file.name}")
    
    # 打印总结
    if verbose:
        logging.info("=" * 60)
        logging.info("转换完成！")
        logging.info("=" * 60)
        logging.info(f"总文件数: {len(h5_files)}")
        logging.info(f"成功转换: {success_count}")
        logging.info(f"转换失败: {failed_count}")
        logging.info(f"总输出大小: {total_size_mb:.2f} MB")
        logging.info(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将H5文件中的features提取并转换为PyTorch .pt文件')
    parser.add_argument('--input_dir', 
                        default='/home/scy/changhai_project/wsi/dsmil-wsi/datasets/wsi_features/UNI2-h_features/TCGA/TCGA-PAAD/',
                        help='输入H5文件目录路径')
    parser.add_argument('--output_dir', 
                        default='./converted_pt_files/',
                        help='输出PT文件目录路径')
    parser.add_argument('--pattern', 
                        default='*.h5',
                        help='文件匹配模式 (默认: *.h5)')
    parser.add_argument('--log_file', 
                        default='convert_h5_to_pt.log',
                        help='日志文件路径')
    parser.add_argument('--test_single', 
                        type=str,
                        help='测试单个文件转换 (提供H5文件路径)')
    parser.add_argument('--verbose', 
                        action='store_true',
                        help='输出详细的日志信息')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file, args.verbose)
    
    if args.verbose:
        logger.info("开始H5到PT转换任务")
        logger.info(f"输入目录: {args.input_dir}")
        logger.info(f"输出目录: {args.output_dir}")
    
    if args.test_single:
        # 测试单个文件
        if args.verbose:
            logger.info(f"测试单个文件: {args.test_single}")
        
        if not os.path.exists(args.test_single):
            logger.error(f"测试文件不存在: {args.test_single}")
            return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 提取样本名称并生成输出路径
        # sample_name = extract_sample_name(os.path.basename(args.test_single))
        sample_name = os.path.basename(args.test_single)
        output_file = os.path.join(args.output_dir, f"{sample_name}.pt")
        
        success, info = convert_h5_to_pt(args.test_single, output_file, args.verbose)
        
        if success:
            if args.verbose:
                logger.info("✅ 单文件测试成功!")
                logger.info(f"输出文件: {output_file}")
        else:
            if args.verbose:
                logger.error("❌ 单文件测试失败!")
    else:
        # 批量转换
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录不存在: {args.input_dir}")
            return
        
        batch_convert_h5_to_pt(args.input_dir, args.output_dir, args.pattern, args.verbose)

if __name__ == "__main__":
    main() 