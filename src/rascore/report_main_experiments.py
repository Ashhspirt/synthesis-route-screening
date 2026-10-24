"""Summarize split variation separately from training-seed variation."""
from pathlib import Path
import json
import hashlib
import itertools
import pilot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'main_models_and_splits'
LABELS={'majority':'多数类','logistic':'逻辑回归','random_forest':'随机森林',
    'large_standard':'原网络','small_standard':'小网络',
    'large_regularized':'原网络＋强正则','small_regularized':'小网络＋强正则',
    'validation_selected_nn':'验证集选出的网络'}
ORDER=['majority','logistic','random_forest','large_standard','small_standard','large_regularized','small_regularized','validation_selected_nn']
METRICS=['auroc','ap','brier','log_loss','balanced_accuracy','training_seconds']


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    frames=[]
    selected_frames=[]
    split_rows=[]
    sensitivity=[]
    feature_overlap=[]
    selections=[]
    test_ids={}
    for split_seed in protocol['split_seeds']:
        folder=OUT/f'split_{split_seed}'
        check=json.loads((folder/'verification.json').read_text())
        assert check['passed'] and check['models_checked']==42
        child_protocol=json.loads((folder/'protocol.json').read_text())
        assert child_protocol['split_seed']==split_seed
        for n,h in protocol['data_hashes'].items():
            assert hashlib.sha256((folder/'data'/n).read_bytes()).hexdigest()==h
        sample=pd.read_csv(folder/'data'/'pilot_sample.csv')
        x=np.load(folder/'data'/'pilot_features.npz')['x']
        data=pd.read_csv(folder/'metrics.csv')
        validation=pd.read_csv(folder/'validation_metrics.csv')
        data['split_seed']=split_seed
        frames.append(data)
        choices=json.loads((folder/'validation_selection.json').read_text())
        quality=json.loads((folder/'split_checks.json').read_text())
        for mode in ['random','scaffold']:
            choice=choices[mode]['selected_neural_config']
            selection=data[(data.split==mode)&(data.model==choice)].copy()
            selection['original_model']=choice
            selection['model']='validation_selected_nn'
            selected_frames.append(selection)
            selections.append({'split_seed':split_seed,'split':mode,'selected_config':choice,
                'mean_validation_log_loss':choices[mode]['mean_validation_log_loss'][choice]})
            for part in ['train','validation','internal_test']:
                split_rows.append({'split_seed':split_seed,'split':mode,'partition':part,**quality[mode][part]})
            indices=np.load(folder/f'{mode}_indices.npz')
            train,valid,test=[indices[n] for n in ['train','validation','internal_test']]
            expected=pilot.split_indices(sample,sample.activity.to_numpy(),mode,split_seed)
            assert all(np.array_equal(a,b) for a,b in zip(expected,[train,valid,test]))
            test_ids[(mode,split_seed)]=set(sample.iloc[test].source_id)
            fingerprints=set(map(bytes,x[train]))
            affected=[int(sample.iloc[i].source_id) for i in test if bytes(x[i]) in fingerprints]
            feature_overlap.append({'split_seed':split_seed,'split':mode,'affected_test_rows':len(affected),
                'internal_test_n':len(test),'train_test_scaffold_overlap':quality[mode]['train_test_scaffold_overlap']})
            for row in data[data.split==mode].itertuples(index=False):
                predictions=pd.read_csv(folder/'runs'/mode/f'seed{row.seed}'/row.model/'internal_test_predictions.csv')
                remaining=predictions[~predictions.source_id.isin(affected)]
                sensitivity.append({'split_seed':split_seed,'split':mode,'seed':row.seed,'model':row.model,
                    'removed_test_rows':len(affected),'auroc_without_overlap':roc_auc_score(remaining.label,remaining.probability),
                    'delta_auroc':roc_auc_score(remaining.label,remaining.probability)-row.auroc})
    runs=pd.concat(frames,ignore_index=True)
    assert len(runs)==126
    assert not runs.duplicated(['split_seed','split','seed','model']).any()
    all_runs=pd.concat([runs]+selected_frames,ignore_index=True)
    assert len(all_runs)==144
    runs.to_csv(OUT/'all_runs.csv',index=False)
    pd.concat(selected_frames).to_csv(OUT/'validation_selected_runs.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'validation_selections.csv',index=False)
    pd.DataFrame(split_rows).to_csv(OUT/'partition_statistics.csv',index=False)
    pd.DataFrame(feature_overlap).to_csv(OUT/'feature_overlap.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'feature_overlap_sensitivity.csv',index=False)
    between_overlap=[]
    for mode in ['random','scaffold']:
        for a,b in itertools.combinations(protocol['split_seeds'],2):
            between_overlap.append({'split':mode,'seed_a':a,'seed_b':b,
                'shared_test_molecules':len(test_ids[(mode,a)]&test_ids[(mode,b)])})
    pd.DataFrame(between_overlap).to_csv(OUT/'between_repetition_test_overlap.csv',index=False)
    grouped=all_runs.groupby(['split_seed','split','model'])
    assert (grouped.size()==3).all()
    split_means=grouped[METRICS].mean().reset_index()
    split_means.to_csv(OUT/'split_level_means.csv',index=False)
    within=grouped[METRICS].std(ddof=1).reset_index().groupby(['split','model'])[METRICS].mean()
    summary=split_means.groupby(['split','model'])[METRICS].agg(['mean','std'])
    summary.columns=[f'{metric}_{"mean" if stat=="mean" else "split_sd"}' for metric,stat in summary.columns]
    for metric in METRICS:
        summary[f'{metric}_mean_training_sd']=within[metric]
    summary=summary.reset_index()
    summary.to_csv(OUT/'main_summary.csv',index=False)
    # Check aggregations independently by indexing the underlying runs.
    for row in summary.itertuples(index=False):
        raw=all_runs[(all_runs.split==row.split)&(all_runs.model==row.model)]
        values=[float(raw[raw.split_seed==s].auroc.mean()) for s in protocol['split_seeds']]
        assert abs(np.mean(values)-row.auroc_mean)<1e-10
        assert abs(np.std(values,ddof=1)-row.auroc_split_sd)<1e-10
    differences=[]
    for name in ORDER:
        a=split_means[(split_means.split=='random')&(split_means.model==name)].set_index('split_seed')
        b=split_means[(split_means.split=='scaffold')&(split_means.model==name)].set_index('split_seed')
        for metric in METRICS[:5]:
            delta=b[metric]-a[metric]
            differences.append({'model':name,'metric':metric,'scaffold_minus_random_mean':delta.mean(),
                'difference_sd_across_split_seeds':delta.std(ddof=1),
                **{f'delta_split_{s}':delta.loc[s] for s in protocol['split_seeds']}})
    pd.DataFrame(differences).to_csv(OUT/'split_type_differences.csv',index=False)
    contrasts=[]
    for mode in ['random','scaffold']:
        for new,base in [('large_standard','logistic'),('small_standard','large_standard'),
            ('large_regularized','large_standard'),('small_regularized','small_standard'),('validation_selected_nn','logistic')]:
            a=split_means[(split_means.split==mode)&(split_means.model==new)].set_index('split_seed')
            b=split_means[(split_means.split==mode)&(split_means.model==base)].set_index('split_seed')
            delta=a.auroc-b.auroc
            contrasts.append({'split':mode,'model':new,'baseline':base,'mean_delta_auroc':delta.mean(),
                'sd_split_delta':delta.std(ddof=1),'splits_with_positive_delta':int((delta>0).sum())})
    pd.DataFrame(contrasts).to_csv(OUT/'model_contrasts.csv',index=False)
    names=ORDER[1:]
    en=['Logistic','Random forest','Large','Small','Large + reg.','Small + reg.','Val-selected NN']
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,metric,title in zip(axes,['auroc','brier'],['AUROC (higher is better)','Brier score (lower is better)']):
        for j,mode in enumerate(['random','scaffold']):
            data=summary[summary.split==mode].set_index('model').loc[names]
            ax.errorbar(np.arange(len(names))+(j-.5)*.18,data[f'{metric}_mean'],
                yerr=data[f'{metric}_split_sd'],fmt='o',capsize=4,label=mode.title())
        ax.set_xticks(np.arange(len(names)),en,rotation=32,ha='right')
        ax.set_title(title)
        ax.grid(axis='y',alpha=.25)
        ax.legend()
    fig.suptitle('3 repeated partitions × 3 training seeds | Error bars: SD of 3 partition means',fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/'main_model_comparison.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,style in [('large_standard','o-'),('small_standard','s--'),('logistic','^:')]:
        for mode,color in [('random','#2878b5'),('scaffold','#df7e2b')]:
            d=split_means[(split_means.model==name)&(split_means.split==mode)].sort_values('split_seed')
            ax.plot([str(s) for s in d.split_seed],d.auroc,style,color=color,label=f'{name} / {mode}')
    ax.set(xlabel='Partition seed',ylabel='AUROC averaged over 3 training seeds',title='How results change with the partition')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'partition_variation.png',dpi=160)
    plt.close(fig)
    verification={'passed':True,'fits_checked':126,'prediction_sets_checked':252,
        'data_hashes_identical':True,'requested_split_seeds_verified':True,
        'hierarchical_aggregation_verified':True,'official_test_excluded':True}
    pilot.save_json(OUT/'verification.json',verification)
    def result(mode,name):
        return summary[(summary.split==mode)&(summary.model==name)].iloc[0]
    lines=['# 主要模型与划分实验报告','',
        '日期：2026-10-04。固定 20,000 个分子，使用三个新的划分种子和三个训练种子完成主要模型比较。此前单划分的试验结果保留，不混入本次汇总。','',
        '## 实验范围与结论','',
        '完成 126 次拟合，其中 72 次为神经网络训练。每份模型都已重新加载并复算验证集与内部测试集指标，共核验 252 组预测。原始正式测试集始终未用于训练、调参或性能评分。','',
        '这是主要实验阶段的内部稳健性评价。新划分与之前的开发数据来自同一批分子，不能称为全新的盲测或临床验证。','',
        '## 实验设计','',
        '| 因素 | 设置 |','|---|---|',
        '| 分子样本 | 固定 20,000 个，从清理后的作者原训练集抽取，抽样种子 42 |',
        '| 划分方法 | 随机分层；Bemis–Murcko 骨架分组 |',
        '| 新划分种子 | 101、202、303 |',
        '| 每个划分的训练种子 | 42、43、44 |',
        '| 模型 | 常数基线、逻辑回归、随机森林、四种神经网络 |',
        '| 特征 | 半径 2、1024 位 Morgan 二值指纹，不含手性编码 |',
        '| 配置选择 | 仅依据三次训练的平均验证集交叉熵 |',
        '| 早停和阈值 | 最多 50 轮，连续 5 轮未改善则早停；阈值在验证集选择 |','',
        '原网络隐藏层为 256→128→64，参数 303,617；小网络为 64→32→16，参数 68,225，减少约 77.5%。普通正则配置为 Dropout 0.2、权重衰减 0；强正则配置为 Dropout 0.4、Adam L2 权重衰减 0.0001。其余固定为学习率 0.001、批大小 64、两个 CPU 线程。强正则同时改变两个因素，不能区分各自贡献。','',
        '每一划分内所有模型共享相同训练、验证和内部测试分子。规范结构无跨集合重复，骨架划分没有跨集合的相同骨架；无环分子作为一个整体骨架组处理。','',
        '## 样本划分','',
        '| 划分种子 | 方法 | 训练 / 验证 / 内部测试数量 | 内部测试正类比例 |',
        '|---:|---|---|---:|']
    stats=pd.DataFrame(split_rows)
    for seed in protocol['split_seeds']:
        for mode in ['random','scaffold']:
            d=stats[(stats.split_seed==seed)&(stats.split==mode)].set_index('partition')
            lines.append(f'| {seed} | {mode} | {d.loc["train","n"]:,} / {d.loc["validation","n"]:,} / {d.loc["internal_test","n"]:,} | {d.loc["internal_test","positive_fraction"]:.2%} |')
    lines += ['', '## 主要结果','',
        '先对每个划分内的三个训练种子求平均，再汇总三个划分均值。下表“±”是三个划分均值的样本标准差，不是九次运行的标准差，也不是置信区间。训练种子间波动另存于 main_summary.csv 的 mean_training_sd 列。','',
        '| 模型 | 随机 AUROC | 骨架 AUROC | 随机 AP | 骨架 AP |','|---|---:|---:|---:|---:|']
    for name in ORDER:
        a,b=result('random',name),result('scaffold',name)
        lines.append(f'| {LABELS[name]} | {a.auroc_mean:.4f} ± {a.auroc_split_sd:.4f} | {b.auroc_mean:.4f} ± {b.auroc_split_sd:.4f} | {a.ap_mean:.4f} ± {a.ap_split_sd:.4f} | {b.ap_mean:.4f} ± {b.ap_split_sd:.4f} |')
    lines += ['', '| 模型 | 随机 Brier | 骨架 Brier |','|---|---:|---:|']
    for name in ORDER[1:]:
        a,b=result('random',name),result('scaffold',name)
        lines.append(f'| {LABELS[name]} | {a.brier_mean:.4f} ± {a.brier_split_sd:.4f} | {b.brier_mean:.4f} ± {b.brier_split_sd:.4f} |')
    lines += ['', 'AUROC/AP 越高越好，Brier 越低越好。正类比例较高，普通准确率和 AP 均需结合常数基线理解。Brier 不能单独证明校准良好。','',
        '## 关键比较','']
    for mode in ['random','scaffold']:
        base=result(mode,'large_standard')
        small=result(mode,'small_standard')
        logistic=result(mode,'logistic')
        lines.append(f'- {mode}：原网络平均 AUROC 比逻辑回归高 {(base.auroc_mean-logistic.auroc_mean)*100:.2f} 个百分点；小网络相对原网络变化 {(small.auroc_mean-base.auroc_mean)*100:+.2f} 个百分点。')
        reg=result(mode,'large_regularized')
        smallreg=result(mode,'small_regularized')
        lines.append(f'- {mode}：加强正则化后，原网络 AUROC 变化 {(reg.auroc_mean-base.auroc_mean)*100:+.2f} 个百分点，小网络变化 {(smallreg.auroc_mean-small.auroc_mean)*100:+.2f} 个百分点。这些差值不表示统计显著性。')
    dif=pd.DataFrame(differences)
    for name in ['large_standard','small_standard','validation_selected_nn']:
        row=dif[(dif.model==name)&(dif.metric=='auroc')].iloc[0]
        d=[row[f'delta_split_{s}'] for s in protocol['split_seeds']]
        lines.append(f'- {LABELS[name]}：骨架划分减随机划分的平均 AUROC 差为 {row.scaffold_minus_random_mean*100:+.2f} 个百分点，三次划分中有 {sum(v<0 for v in d)}/3 次为负。两种方法的测试分子不同，此差值是描述性比较，不能解释为骨架变化的纯因果效应。')
    lines += ['', '## 按验证集选择模型','',
        '预先按平均验证交叉熵选出配置，然后才评价内部测试集。下表记录真实选择过程，不能事后换成内部测试分数最高的配置。','',
        '| 划分种子 | 方法 | 验证集选出的网络 | 平均验证交叉熵 |','|---:|---|---|---:|']
    for row in selections:
        lines.append(f'| {row["split_seed"]} | {row["split"]} | {LABELS[row["selected_config"]]} | {row["mean_validation_log_loss"]:.4f} |')
    lines += ['', '## 稳健性与边界','',
        '重复划分不是互不重叠的外部数据集。各重复内部测试集的交集已记录在 between_repetition_test_overlap.csv，因此没有把九次训练当作九个独立样本做显著性检验，也没有报告过度精确的置信区间。',
        '样本固定为此前的 20,000 个分子，结果不涵盖重新抽样的不确定性。原料库和底层规划器的限制仍随标签保留；标签不是实验室合成成功率。',
        f'有限长度二值指纹出现少量不同分子共用表示的情况。各划分已记录受影响内部测试样本数，并复算剔除这些样本后的 AUROC；126 次运行中最大绝对变化为 {max(abs(r["delta_auroc"]) for r in sensitivity):.6f}，详见 feature_overlap_sensitivity.csv。主要结果保持原预定测试集，未用事后排除替换。',
        '与先前 5,000 样本或单划分结果的数字变化不构成严格的数据量消融，因为评估分子并非固定。四种网络配置来源于之前开发阶段，故本轮应报告为内部验证而非完全未使用信息的盲测。','',
        '## 交付材料与下一阶段','',
        '- all_runs.csv：126 条逐次指标；split_level_means.csv：先按训练种子平均的结果。',
        '- main_summary.csv：总体均值、划分间标准差、平均训练种子标准差。',
        '- validation_selected_runs.csv：按验证集选出网络后的 18 条结果。',
        '- model_contrasts.csv、split_type_differences.csv：模型与划分方式的差值。',
        '- verification.json：126 份模型和 252 组预测的核验汇总。',
        '- split_101/、split_202/、split_303/：完整参数、预测、划分、训练曲线与日志。','',
        '主要模型比较与重复划分实验已完成这一阶段。后续优先开展错误分析和结构特征消融；概率校准需要另设校准验证子集，正式测试评价应在方法冻结之后进行。论文可以据此形成“模型比较”“划分稳健性”“资源成本”三部分，但整个课程项目仍需相关工作、最终报告和 Lectrace 展示。','',
        '数据和标签背景：[RAscore 官方仓库](https://github.com/reymond-group/RAscore)、[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8179384/)。']
    (OUT/'主要模型与划分实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[['split','model','auroc_mean','auroc_split_sd','auroc_mean_training_sd','brier_mean']].to_string(index=False))
    print('\nModel contrasts:\n',pd.DataFrame(contrasts).to_string(index=False))
    print('\nValidation selections:\n',pd.DataFrame(selections).to_string(index=False))
    print('Max fingerprint sensitivity |delta AUROC|:',max(abs(r['delta_auroc']) for r in sensitivity))
    print('All split checks and aggregate checks passed.')


if __name__=='__main__':
    main()
