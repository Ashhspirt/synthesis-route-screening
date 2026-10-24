"""Generate plots, feature-collision sensitivity, and a Chinese experiment report."""
from pathlib import Path
import argparse
import json
import pilot
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

NAMES={'majority':'多数类基线','logistic':'逻辑回归','random_forest':'随机森林',
    'large_standard':'原网络','small_standard':'小网络',
    'large_regularized':'原网络＋强正则','small_regularized':'小网络＋强正则'}
ORDER=['logistic','random_forest','large_standard','small_standard','large_regularized','small_regularized']
ENGLISH=['Logistic','Random forest','Large','Small','Large + reg.','Small + reg.']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'experiment_20000_three_seeds')
    out=parser.parse_args().output
    summary=pd.read_csv(out/'summary.csv')
    runs=pd.read_csv(out/'metrics.csv')
    validation=pd.read_csv(out/'validation_metrics.csv')
    audit=json.loads((out/'data'/'data_audit.json').read_text())
    splits=json.loads((out/'split_checks.json').read_text())
    selection=json.loads((out/'validation_selection.json').read_text())
    completion=json.loads((out/'completion.json').read_text())
    verification=json.loads((out/'verification.json').read_text())
    assert verification['passed']
    sample=pd.read_csv(out/'data'/'pilot_sample.csv')
    features=np.load(out/'data'/'pilot_features.npz')['x']
    collisions=[]
    sensitivity=[]
    # Binary folded fingerprints can coincide for chemically distinct inputs.
    # Describe this limitation; do not retroactively change the primary evaluation set.
    for mode in ['random','scaffold']:
        idx=np.load(out/f'{mode}_indices.npz')
        lookup={}
        for i in idx['train']:
            lookup.setdefault(bytes(features[i]),[]).append(int(i))
        affected=[]
        for i in idx['internal_test']:
            for j in lookup.get(bytes(features[i]),[]):
                affected.append(int(sample.iloc[i].source_id))
                collisions.append({'split':mode,'training_id':int(sample.iloc[j].source_id),
                    'test_id':int(sample.iloc[i].source_id),'training_smiles':sample.iloc[j].canonical_smiles,
                    'test_smiles':sample.iloc[i].canonical_smiles,'training_label':int(sample.iloc[j].activity),
                    'test_label':int(sample.iloc[i].activity),
                    'same_scaffold':bool(sample.iloc[j].scaffold==sample.iloc[i].scaffold)})
        for row in runs[runs.split==mode].itertuples(index=False):
            saved=pd.read_csv(out/'runs'/mode/f'seed{row.seed}'/row.model/'internal_test_predictions.csv')
            reduced=saved[~saved.source_id.isin(affected)]
            sensitivity.append({'split':mode,'seed':row.seed,'model':row.model,'removed_test_rows':len(saved)-len(reduced),
                'auroc_without_seen_fingerprints':float(roc_auc_score(reduced.label,reduced.probability)),
                'delta_auroc':float(roc_auc_score(reduced.label,reduced.probability)-row.auroc),
                'ap_without_seen_fingerprints':float(average_precision_score(reduced.label,reduced.probability))})
    pd.DataFrame(collisions).to_csv(out/'fingerprint_overlap_examples.csv',index=False)
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(out/'fingerprint_overlap_sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric,title in zip(axes,['auroc','brier'],['AUROC (higher is better)','Brier score (lower is better)']):
        for j,mode in enumerate(['random','scaffold']):
            d=summary[summary.split==mode].set_index('model').loc[ORDER]
            ax.errorbar(np.arange(len(ORDER))+(j-.5)*.14,d[f'{metric}_mean'],
                yerr=d[f'{metric}_std'],fmt='o',capsize=4,label=mode.title())
        ax.set_xticks(np.arange(len(ORDER)),ENGLISH,rotation=30,ha='right')
        ax.set_title(title)
        ax.grid(axis='y',alpha=.25)
        ax.legend()
    fig.suptitle('20,000 molecules | Fixed partitions | Mean ± SD over 3 training seeds',fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'model_comparison.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for i,mode in enumerate(['random','scaffold']):
        for j,name in enumerate(['large_standard','small_regularized']):
            ax=axes[i,j]
            for seed,color in zip([42,43,44],['#2878b5','#e68128','#348d58']):
                folder=out/'runs'/mode/f'seed{seed}'/name
                h=pd.read_csv(folder/'history.csv')
                info=json.loads((folder/'validation.json').read_text())
                ax.plot(h.epoch,h.validation_loss,color=color,label=f'Validation seed {seed}')
                ax.plot(h.epoch,h.train_loss,color=color,linestyle=':',alpha=.55)
                ax.scatter([info['best_epoch']],[h.loc[h.epoch==info['best_epoch'],'validation_loss'].iloc[0]],color=color,s=25)
            ax.set(title=f'{mode.title()} / {"Large" if j==0 else "Small + regularization"}',
                xlabel='Epoch',ylabel='Binary cross-entropy')
            ax.legend(fontsize=8)
    fig.suptitle('Solid: validation | Dotted: training (dropout active) | Dots: selected epochs',fontsize=11)
    fig.tight_layout()
    fig.savefig(out/'learning_curves_comparison.png',dpi=150)
    plt.close(fig)

    def get(mode,model):
        return summary[(summary.split==mode)&(summary.model==model)].iloc[0]
    lines=['# 20,000 样本、模型缩小与正则化实验报告','',
        '日期：2026-10-04。所有数值来自本机 CPU 实际运行，原有 5,000 样本结果保留。', '',
        '## 结论','',
        '本轮完成 42 次模型拟合，其中 24 次是神经网络训练。原网络在随机划分下的平均 AUROC 最高；小网络以明显更少的参数达到接近的表现。加强正则化没有带来一致的 AUROC 改善，不能将“更强正则化”直接等同于“更好”。','',
        '## 预先固定的设计','',
        '- 从清理后的原训练池按标签分层抽取固定的 20,000 个分子，抽样种子为 42。',
        '- 随机划分与骨架划分各固定一套，划分种子为 42；各模型使用完全相同的数据。',
        '- 训练种子为 42、43、44，改变初始化、Dropout、批次顺序和随机森林随机性；没有重新抽样或重新划分。',
        '- 三次运行的标准差仅反映训练随机性，不是数据集不确定性或置信区间。逻辑回归和常数基线基本为确定性计算，重复分数相同。',
        '- 两种划分下均先按三次平均验证集交叉熵选择神经网络配置，再计算内部测试指标；正式测试集始终不参与性能评价。','',
        '## 网络与训练设置','',
        '| 配置 | 隐藏层 | 参数量 | Dropout | Adam L2 权重衰减 |',
        '|---|---|---:|---:|---:|',
        '| 原网络 | 256→128→64 | 303,617 | 0.2 | 0 |',
        '| 小网络 | 64→32→16 | 68,225 | 0.2 | 0 |',
        '| 原网络＋强正则 | 256→128→64 | 303,617 | 0.4 | 0.0001 |',
        '| 小网络＋强正则 | 64→32→16 | 68,225 | 0.4 | 0.0001 |','',
        '保持相同的 1024 位 Morgan 指纹、学习率 0.001、批大小 64、最多 50 轮、早停耐心值 5。小网络参数减少约 77.5%。加强正则化同时改变 Dropout 和 L2，因此本实验不能单独归因于其中一个因素。所有神经网络仍包含三个隐藏层。','',
        '## 数据与划分','',
        f'20,000 个分子正类比例 {audit["pilot_positive_fraction"]:.2%}，共有 {audit["pilot_unique_scaffolds"]:,} 个骨架；{audit["pilot_acyclic_rows"]} 个无环分子统一归入空骨架组。全量数据检查规则与上一阶段一致：去除 3 条重复记录，无非法结构、冲突标签或原训练/测试集规范结构交叉。','',
        '| 划分 | 训练 / 验证 / 内部测试 | 内部测试正类比例 | 训练与内部测试的骨架交集 |',
        '|---|---|---:|---:|']
    for mode in ['random','scaffold']:
        s=splits[mode]
        lines.append(f'| {mode} | {s["train"]["n"]:,} / {s["validation"]["n"]:,} / {s["internal_test"]["n"]:,} | {s["internal_test"]["positive_fraction"]:.2%} | {s["train_test_scaffold_overlap"]} |')
    lines += ['', '## 内部测试结果','', '所有 ± 为三个训练种子的样本标准差（ddof=1）；AUROC、AP 越高越好，Brier 越低越好。','',
        '| 模型 | 随机 AUROC | 骨架 AUROC | 随机 AP | 骨架 AP |','|---|---:|---:|---:|---:|']
    for model in ['majority']+ORDER:
        a,b=get('random',model),get('scaffold',model)
        lines.append(f'| {NAMES[model]} | {a.auroc_mean:.4f} ± {a.auroc_std:.4f} | {b.auroc_mean:.4f} ± {b.auroc_std:.4f} | {a.ap_mean:.4f} ± {a.ap_std:.4f} | {b.ap_mean:.4f} ± {b.ap_std:.4f} |')
    lines += ['', '| 模型 | 随机 Brier | 骨架 Brier | 随机平均训练秒数 | 骨架平均训练秒数 |','|---|---:|---:|---:|---:|']
    for model in ORDER:
        a,b=get('random',model),get('scaffold',model)
        lines.append(f'| {NAMES[model]} | {a.brier_mean:.4f} ± {a.brier_std:.4f} | {b.brier_mean:.4f} ± {b.brier_std:.4f} | {a.training_seconds_mean:.2f} | {b.training_seconds_mean:.2f} |')
    lines += ['', '## 如何解释这些结果','']
    a,b=get('random','large_standard'),get('random','small_standard')
    lines.append(f'1. 随机划分下，小网络 AUROC 相对原网络减少 {(a.auroc_mean-b.auroc_mean)*100:.2f} 个百分点，训练循环平均耗时减少 {(1-b.training_seconds_mean/a.training_seconds_mean)*100:.1f}%。它适合作为低成本候选，但不能称为统计等效。')
    lines.append('2. 强正则化没有稳定提升 AUROC。小网络在骨架划分下的 AUROC 均值略升、训练种子间波动减小，而其他组合没有同样收益；详细配对差值保存在 paired_contrasts.csv。不能从三个种子声称显著提升。')
    for mode in ['random','scaffold']:
        chosen=selection[mode]['selected_neural_config']
        row=get(mode,chosen)
        lines.append(f'3. {mode} 划分按验证损失选出的配置是“{NAMES[chosen]}”，其内部测试 AUROC 为 {row.auroc_mean:.4f} ± {row.auroc_std:.4f}。该选择未使用内部测试分数，因此可能不是内部测试 AUROC 最高者。')
    lines += ['4. 本次骨架划分的平均 AUROC 低于随机划分。但它们包含不同测试分子和不同类别比例，且只有一套固定划分，不能把差异全部归因于新骨架。需要更换划分种子才能评价划分不确定性。',
        '5. 与上一轮 5,000 样本相比，本轮原网络随机 AUROC 数值更高，但测试分子也改变了，这不是固定测试集的训练数据量消融，不能量化纯粹的数据增益。',
        '6. 神经网络仍多在较早轮次达到最小验证损失。已恢复各次最佳参数，不能简单增加训练轮数。不同网络在不同轮次早停，最佳轮次处的训练/验证差值并非完全受控的过拟合程度比较。','',
        '## 指纹重叠检查与敏感性分析','',
        '规范化分子在不同集合之间没有重复，但有限长度、无手性、二值指纹不能区分所有不同结构。随机划分有 3 个与训练集相同的指纹，骨架划分有 1 个。原始结构与标签已列入 fingerprint_overlap_examples.csv。不能把这种表示重叠说成相同分子泄漏，也不应忽略它。',
        f'保留预定主要结果，另做一次仅排除这些内部测试样本的敏感性复算，不重新训练：所有模型的 AUROC 变化绝对值最大为 {sensitivity.delta_auroc.abs().max():.6f}。详见 fingerprint_overlap_sensitivity.csv。','',
        '## 验证与运行成本','',
        f'全流程（数据检查、准备、42 次拟合及评价）内部计时约 {completion["elapsed_seconds"]:.1f} 秒；不包括后续独立核验、报告绘图和依赖安装。每次神经网络训练约 2—4 秒，当前计算成本可控。',
        '已重新加载全部 42 份模型并核验 84 组验证/内部测试预测，重新计算 AUROC、AP、Brier 和交叉熵，检查每份神经网络参数来自验证损失最小轮次，并核对三种子汇总与配置选择。全部通过，见 verification.json。',
        '多数类正例约占四分之三，不能单独用普通准确率证明价值。标签仍表示原规划器在原配置下是否找到路线，不表示真实实验成败。','',
        '## 建议的后续方向','',
        '保留原网络和小网络作为主要候选；不要默认采用更强正则化。下一步优先增加独立数据划分重复、错误分析和少量结构特征消融。若要分别判断 Dropout 和 L2 的作用，需要另设单因素实验。正式测试集应在方法冻结之后评价。','',
        '本轮内部测试结果已被查看，今后据此提出的新方法仍属于开发阶段。不能把这批内部结果称为最终盲测。','',
        '文件：summary.csv 为均值与标准差，metrics.csv 为 42 条逐次结果，validation_metrics.csv 为选择配置的验证结果，runs/ 保存模型、预测与学习曲线。']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report and plots saved. Max |AUROC change| after fingerprint sensitivity exclusion:',sensitivity.delta_auroc.abs().max())


if __name__=='__main__':
    main()
