import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_core_region_heatmap(expert, name, cognitive_system, save_dir="heatmaps/V16_core"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_core_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    plot_size = min(200, dim)
    partition_size = int(dim * 0.2)

    synapse_plot = synapse[:plot_size, :plot_size]

    neuron_activation_count = np.zeros(dim)
    # 存储：神经元ID -> 绑定的实体列表
    neuron_to_entities = defaultdict(list)

    # 🔥 调试：先打印实体总数，确认index里有数据
    all_entities = list(cognitive_system.cortex.index.entities.items())
    print(f"[DEBUG] [{name}] 实体索引中总实体数: {len(all_entities)}")
    matched_entities_count = 0

    # 遍历实体索引（适配实体架构）
    for entity_id, entity in all_entities:
        # 🔥 修复1：放宽过滤条件，优先匹配类型，再匹配metadata
        # 1. 先看实体的expert字段，或者entity_type是否匹配
        entity_expert = entity.metadata.get("expert", entity.entity_type)
        # 2. 兼容大小写和中英文（比如"identity"和"身份"）
        expert_name_map = {
            "identity": "身份", "visual": "视觉", "concept": "概念",
            "spatial": "空间", "abstract": "抽象"
        }
        entity_expert = expert_name_map.get(entity_expert.lower(), entity_expert)
        name_cn = expert_name_map.get(name.lower(), name)
        
        # 🔥 修复2：只要实体是当前专家的，或者实体类型匹配，就保留
        if entity_expert != name and entity.entity_type != name:
            continue
        
        matched_entities_count += 1
        # 获取实体SDR向量
        sdr = entity.sdr
        # 🔥 修复3：强制获取实体名称和内容，避免空值
        entity_name = entity.name if entity.name else entity_id[:8]
        # 补充实体内容（可选）
        entity_content = entity.latest_evidence.content if entity.latest_evidence else entity_name
        
        # 统计激活的神经元（🔥 修复4：降低激活阈值，避免全0SDR被过滤）
        active_neurons = torch.where(sdr > 0.05)[0].numpy()
        for neuron_id in active_neurons:
            if neuron_id < plot_size:
                neuron_activation_count[neuron_id] += 1
                # 存储实体信息（用于标注）
                neuron_to_entities[neuron_id].append({
                    "name": entity_name,
                    "content": entity_content[:30]  # 截断避免过长
                })

    print(f"[DEBUG] [{name}] 匹配到的实体数: {matched_entities_count} | 绑定实体的神经元数: {len(neuron_to_entities)}")

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 9

    mask = np.zeros((plot_size, plot_size))
    mask[:min(partition_size, plot_size), :min(partition_size, plot_size)] = 1
    heatmap_data = synapse_plot * mask * 2.5 + synapse_plot * (1 - mask) * 0.2
    im = plt.imshow(heatmap_data, cmap='coolwarm', vmin=-2.5, vmax=2.5)
    cbar = plt.colorbar(im, label='突触权重（强化局部连接）', shrink=0.8)

    # 获取激活度最高的前10个神经元（🔥 修复5：只取激活数>0的神经元）
    non_zero_neurons = np.where(neuron_activation_count > 0)[0]
    if len(non_zero_neurons) > 0:
        top_neurons = non_zero_neurons[np.argsort(neuron_activation_count[non_zero_neurons])[::-1][:10]]
    else:
        top_neurons = []
    
    top_neurons = sorted(top_neurons)
    
    # 去重实体，每个神经元只显示1个绑定实体
    cleaned_entities = {}
    for nid, entity_list in neuron_to_entities.items():
        unique_entities = list({e["name"]: e for e in entity_list}.values())[:1]
        cleaned_entities[nid] = unique_entities

    # 标注位置方向
    directions = [(5, 0, 'left'), (0, 5, 'center'), (-5, 0, 'right'), (0, -5, 'center')]
    used_labels = set()

    for idx, neuron_id in enumerate(top_neurons):
        if neuron_id >= plot_size or neuron_activation_count[neuron_id] == 0:
            continue
        
        # 绘制神经元标记点
        plt.scatter(neuron_id, neuron_id, s=90, c='white', marker='+', linewidths=2,
                   label=f'神经元{neuron_id}(激活:{int(neuron_activation_count[neuron_id])})')
        
        if neuron_id not in cleaned_entities or len(cleaned_entities[neuron_id]) == 0:
            continue

        # ===================== 🔥 核心：标注神经元绑定的实体 =====================
        bound_entity = cleaned_entities[neuron_id][0]
        # 醒目格式：【实体】+ 实体名称
        entity_label = f"【实体】{bound_entity['name']}"
        
        # 避免重复标签
        while entity_label in used_labels:
            entity_label += f"[{neuron_id}]"
        used_labels.add(entity_label)

        offset_x, offset_y, ha = directions[idx % 4]

        # 绘制实体标注（黄色背景，醒目显示）
        plt.text(
            neuron_id + offset_x,
            neuron_id + offset_y,
            entity_label,  # 直接标注绑定的实体
            fontsize=8,
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=1.0, edgecolor='red')
        )

    # 绘制核心区框线
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), min(partition_size, plot_size), min(partition_size, plot_size),
                                       linewidth=3, edgecolor='gold', linestyle='--', facecolor='none',
                                       label='核心功能区'))

    # 计算连接率
    core_connections = np.sum(np.abs(synapse[:partition_size, :partition_size]) > 0.1)
    local_rate = (core_connections / (partition_size * partition_size)) * 100
    
    # ===================== 标题优化：显示实体数量 =====================
    plt.title(
        f'[{name}] 核心区细节热力图（前200维）\n'
        f'局部连接率: {local_rate:.2f}% | 绑定实体数: {len(neuron_to_entities)}\n'
        f'时间: [{datetime.now().strftime("%Y:%m:%d %H:%M:%S")}]',
        fontsize=14
    )
    plt.xlabel('神经元索引（前200）', fontsize=12)
    plt.ylabel('神经元索引（前200）', fontsize=12)
    
    plt.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ [{name}] 核心区细节图已保存: {save_path}")
    return local_rate


def plot_local_connectivity_heatmap(expert, name, cognitive_system, save_dir="heatmaps/V16"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_blockwise_2048_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    full_dim = 2048

    block_size = 64
    n_blocks = full_dim // block_size

    block_heatmap = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            si, ei = i*block_size, (i+1)*block_size
            sj, ej = j*block_size, (j+1)*block_size
            block = synapse[si:ei, sj:ej]
            block_heatmap[i, j] = np.mean(np.abs(block))

    core_size = int(full_dim * 0.2)
    core_blocks = int(np.ceil(core_size / block_size))

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 10

    im = plt.imshow(block_heatmap, cmap='coolwarm', vmin=0, vmax=np.percentile(block_heatmap, 99))
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Average Synapse Strength | 块平均连接强度', fontsize=12)

    plt.gca().add_patch(plt.Rectangle(
        (-0.5, -0.5), core_blocks, core_blocks,
        linewidth=3, edgecolor='gold', linestyle='--',
        facecolor='none', label='Core Region | 核心区'
    ))

    plt.xticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.yticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.xlabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)
    plt.ylabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)

    total_connections = np.sum(np.abs(synapse) > 0.1)
    core_connections = np.sum(np.abs(synapse[:core_size, :core_size]) > 0.1)
    core_rate = (core_connections / total_connections) * 100 if total_connections > 0 else 0

    plt.title(
        f'[{name}] 2048维突触 · 分块聚合热力图\n'
        f'核心区连接占比: {core_rate:.1f}% | 总稀疏度: {1-total_connections/(2048*2048):.2%}\n'
        f'时间: [{datetime.now().strftime("%Y:%m:%d %H:%M:%S")}]',
        fontsize=14
    )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"\n✅ [{name}] 2048维完整热力图已保存: {save_path}")
    return core_rate