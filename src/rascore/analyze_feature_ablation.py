"""Reload every model, verify preprocessing and predictions, then error analysis."""
import json
import argparse
from pathlib import Path
import feature_ablation as a
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

LABELS = {'fp_only':'仅指纹','descriptors_only':'仅五项结构特征','fp_all':'指纹＋全部五项',
          'fp_no_mass':'移除分子量','fp_no_topology':'移除环数与可旋转键','fp_no_hbond':'移除氢键供体与受体'}
METRICS = ['auroc','ap','brier','log_loss','balanced_accuracy']


def table(df):
    return '| ' + ' | '.join(map(str,df.columns)) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str,row)) + ' |' for row in df.itertuples(index=False,name=None))


def verify(out, sample, fp, y, d, results, val):
    protocol = json.loads((out/'protocol.json').read_text())
    assert a.sha(a.DATA/'pilot_sample.csv') == protocol['sample_sha256']
    assert a.sha(a.DATA/'pilot_features.npz') == protocol['fingerprint_sha256']
    assert a.sha(a.ROOT/'feature_ablation.py') == protocol['source_sha256']
    main = json.loads((a.MAIN/'protocol.json').read_text())
    for name, h in protocol['baseline_sources'].items():
        assert a.sha(a.ROOT/name) == h == main['code_hashes'][name]
    assert (sample.source_split == 'train').all()
    assert len(results) == len(val) == 108
    assert not results.duplicated(['split_seed','split','seed','condition']).any()
    np.testing.assert_allclose(a.descriptors(sample), d, rtol=1e-12, atol=1e-12)
    checked = 0
    for ss in a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            splitfile = a.MAIN/f'split_{ss}'/f'{mode}_indices.npz'
            assert a.sha(splitfile) == protocol['split_files'][f'{ss}/{mode}']
            idx = np.load(splitfile)
            tr, va, te = [idx[k] for k in ['train','validation','internal_test']]
            for got, expected in zip([tr,va,te],a.pilot.split_indices(sample,y,mode,ss)):
                assert np.array_equal(got,expected)
            assert len(np.unique(np.concatenate([tr,va,te]))) == len(sample)
            assert not set(sample.iloc[tr].canonical_smiles) & set(sample.iloc[te].canonical_smiles)
            if mode == 'scaffold':
                assert not set(sample.iloc[tr].scaffold) & set(sample.iloc[te].scaffold)
            part = out/f'split_{ss}'/mode
            sc = json.loads((part/'scaler.json').read_text())
            np.testing.assert_allclose(sc['mean'], d[tr].mean(axis=0),atol=1e-10)
            np.testing.assert_allclose(sc['scale'], d[tr].std(axis=0),atol=1e-10)
            assert sc['n_samples_seen'] == len(tr)
            assert sc['features'] == a.FEATURES
            assert sc['train_indices_sha256'] == a.hashlib.sha256(tr.tobytes()).hexdigest()
            z = (d-np.array(sc['mean']))/np.array(sc['scale'])
            for seed in a.TRAIN_SEEDS:
                for condition in a.CONDITIONS:
                    folder = part/f'seed{seed}'/condition
                    row = results[(results.split_seed==ss)&(results.split==mode)&(results.seed==seed)&(results.condition==condition)].iloc[0]
                    vr = val[(val.split_seed==ss)&(val.split==mode)&(val.seed==seed)&(val.condition==condition)].iloc[0]
                    if condition == 'fp_only':
                        ref = json.loads((folder/'baseline_reference.json').read_text())
                        old = Path(ref['path'])
                        assert a.sha(old/'model.pt') == ref['sha256']
                        cp = torch.load(old/'model.pt',map_location='cpu',weights_only=True)
                        model = a.old_network(cp['config'])
                        hist = pd.read_csv(old/'history.csv')
                    else:
                        cp = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                        model = a.network(cp['input_dim'])
                        hist = pd.read_csv(folder/'history.csv')
                    model.load_state_dict(cp['state_dict'])
                    assert sum(p.numel() for p in model.parameters()) == row.parameter_count
                    assert cp['best_epoch'] == int(hist.loc[hist.validation_loss.idxmin(),'epoch']) == row.best_epoch
                    x = a.inputs(fp,z,condition)
                    for name, indices, metricrow in [('validation',va,vr),('internal_test',te,row)]:
                        pred = pd.read_csv(folder/f'{name}_predictions.csv')
                        assert np.array_equal(pred.source_id,sample.iloc[indices].source_id)
                        assert np.array_equal(pred.label,y[indices])
                        p = a.predict(model,x[indices])
                        np.testing.assert_allclose(p,pred.probability,atol=1e-7,rtol=1e-6)
                        for metric,v in a.scores(y[indices],p,row.threshold).items():
                            np.testing.assert_allclose(v,metricrow[metric],atol=1e-7,rtol=1e-6)
                        if name == 'validation':
                            grid = np.arange(.05,.951,.01)
                            best = grid[np.argmax([balanced_accuracy_score(y[va],p>=t) for t in grid])]
                            np.testing.assert_allclose(best,row.threshold,atol=1e-12)
                        else:
                            assert np.array_equal(pred.prediction, (p>=row.threshold).astype(int))
                            if condition == 'fp_only':
                                orig = pd.read_csv(old/'internal_test_predictions.csv')
                                np.testing.assert_allclose(p,orig.probability,atol=1e-7,rtol=1e-6)
                    checked += 1
            print(f'Verified {ss}/{mode}',flush=True)
    choice = pd.read_csv(out/'validation_selection.csv')
    means = val.groupby(['split_seed','split','condition']).log_loss.mean()
    for _, r in choice.iterrows():
        np.testing.assert_allclose(r.log_loss,means.loc[(r.split_seed,r.split,r.condition)])
        assert bool(r.selected) == bool(r.log_loss == choice[(choice.split_seed==r.split_seed)&(choice.split==r.split)].log_loss.min())
    a.pilot.save_json(out/'verification.json',{'passed':True,'models_checked':checked,'new_models':90,
        'reused_models':18,'predictions_checked':'validation and internal test for every model',
        'descriptors_recomputed':True,'training_only_scalers_checked':True,'all_thresholds_recomputed':True,
        'official_test_used':False,'same_main_partitions':True})


def error_analysis(out,sample,d):
    observations, group_rows, bin_rows = [], [], []
    for ss in a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            idx = np.load(a.MAIN/f'split_{ss}'/f'{mode}_indices.npz')
            tr, te = idx['train'],idx['internal_test']
            groups = [('all','all',np.ones(len(te),dtype=bool))]
            seen = sample.iloc[te].scaffold.isin(set(sample.iloc[tr].scaffold)).to_numpy()
            groups += [('scaffold_seen','seen',seen),('scaffold_seen','unseen',~seen)]
            for j,feature in enumerate(a.FEATURES):
                cuts = np.unique(np.quantile(d[tr,j],[.25,.5,.75]))
                assignment = np.searchsorted(cuts,d[te,j],side='left')
                for k in range(len(cuts)+1):
                    bin_rows.append({'split_seed':ss,'split':mode,'feature':feature,'bin':f'B{k+1}',
                                     'lower_exclusive':float(cuts[k-1]) if k else '-inf',
                                     'upper_inclusive':float(cuts[k]) if k<len(cuts) else 'inf'})
                    groups.append((feature,f'B{k+1}',assignment==k))
            for seed in a.TRAIN_SEEDS:
                for condition in ['fp_only','fp_all']:
                    folder=out/f'split_{ss}'/mode/f'seed{seed}'/condition
                    pred=pd.read_csv(folder/'internal_test_predictions.csv')
                    wrong = pred.label.to_numpy()!=pred.prediction.to_numpy()
                    obs=sample.iloc[te][['source_id','canonical_smiles','scaffold']].reset_index(drop=True).copy()
                    obs[['label','probability','prediction']]=pred[['label','probability','prediction']]
                    obs['split_seed'],obs['split'],obs['seed'],obs['condition']=ss,mode,seed,condition
                    obs['error']=wrong.astype(int)
                    observations.append(obs)
                    for feature,b,mask in groups:
                        n0=int(np.sum(mask&(pred.label.to_numpy()==0)))
                        n1=int(np.sum(mask&(pred.label.to_numpy()==1)))
                        fp=int(np.sum(mask&wrong&(pred.label.to_numpy()==0)))
                        fn=int(np.sum(mask&wrong&(pred.label.to_numpy()==1)))
                        group_rows.append({'split_seed':ss,'split':mode,'seed':seed,'condition':condition,
                            'feature':feature,'bin':b,'n':int(mask.sum()),'n_negative':n0,'n_positive':n1,
                            'fp':fp,'fn':fn,'fpr':fp/n0 if n0 else np.nan,'fnr':fn/n1 if n1 else np.nan,
                            'sparse_negative':n0<30,'sparse_positive':n1<30})
    obs=pd.concat(observations,ignore_index=True)
    obs.to_csv(out/'error_observations.csv',index=False)
    gr=pd.DataFrame(group_rows)
    gr.to_csv(out/'error_groups_per_run.csv',index=False)
    pd.DataFrame(bin_rows).to_csv(out/'training_quantile_bins.csv',index=False)
    keys=['split','condition','feature','bin']
    numeric=['n','n_negative','n_positive','fp','fn','fpr','fnr']
    level=gr.groupby(['split_seed']+keys)[numeric].mean().reset_index()
    level.to_csv(out/'error_groups_split_means.csv',index=False)
    summary=level.groupby(keys)[numeric].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(v).rstrip('_') for v in summary.columns]
    summary.to_csv(out/'error_group_summary.csv',index=False)
    # One vote per partition after averaging seeds, then equal weight per held-out appearance.
    unique=obs.groupby(['split','source_id','condition','split_seed']).agg(
        label=('label','first'),probability=('probability','mean'),error_fraction=('error','mean')).reset_index()
    unique=unique.groupby(['split','source_id','condition']).agg(label=('label','first'),
        probability=('probability','mean'),error_fraction=('error_fraction','mean'),
        heldout_partitions=('split_seed','nunique')).reset_index()
    base=unique[unique.condition=='fp_only'].drop(columns='condition')
    aug=unique[unique.condition=='fp_all'][['split','source_id','probability','error_fraction']].rename(
        columns={'probability':'augmented_probability','error_fraction':'augmented_error_fraction'})
    base=base.merge(aug,on=['split','source_id'],validate='one_to_one')
    base=base.merge(sample[['source_id','canonical_smiles']],on='source_id',validate='many_to_one')
    desc=pd.DataFrame(d,columns=a.FEATURES);desc.insert(0,'source_id',sample.source_id)
    base=base.merge(desc,on='source_id',validate='many_to_one')
    base['wrong_class_probability']=np.where(base.label==0,base.probability,1-base.probability)
    base.to_csv(out/'unique_molecule_errors.csv',index=False)
    cases=[]
    for label in [0,1]:
        # Most stringent scaffold split; 3 unique examples per error direction.
        chosen=base[(base.split=='scaffold')&(base.label==label)&(base.error_fraction==1)].nlargest(3,'wrong_class_probability').copy()
        chosen['error_type']='FP' if label==0 else 'FN'
        cases.append(chosen)
    cases=pd.concat(cases,ignore_index=True)
    cases.to_csv(out/'error_case_studies.csv',index=False)
    legends=[f'{r.error_type} ID {r.source_id}\nlabel={r.label}, p={r.probability:.3f}\nheldout splits={r.heldout_partitions}' for r in cases.itertuples()]
    img=Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in cases.canonical_smiles],molsPerRow=3,
                            subImgSize=(440,370),legends=legends)
    img.save(str(out/'error_molecules.png'))
    return summary,cases,gr


def report(out,results,val,errors,cases,gr):
    splitmeans=results.groupby(['split_seed','split','condition'])[METRICS].mean().reset_index()
    splitmeans.to_csv(out/'split_level_means.csv',index=False)
    summary=splitmeans.groupby(['split','condition'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(v).rstrip('_') for v in summary.columns]
    summary.to_csv(out/'ablation_summary.csv',index=False)
    contrast=[]
    for ref in ['fp_only','fp_all']:
        baseline=splitmeans[splitmeans.condition==ref].set_index(['split_seed','split'])
        for r in splitmeans.itertuples():
            if r.condition==ref: continue
            contrast.append({'split_seed':r.split_seed,'split':r.split,'condition':r.condition,'reference':ref,
                             **{m:getattr(r,m)-baseline.loc[(r.split_seed,r.split),m] for m in METRICS}})
    contrasts=pd.DataFrame(contrast)
    contrasts.to_csv(out/'paired_split_differences.csv',index=False)
    cs=contrasts.groupby(['split','condition','reference'])[METRICS].agg(['mean','std']).reset_index()
    cs.columns=['_'.join(v).rstrip('_') for v in cs.columns]
    cs.to_csv(out/'paired_difference_summary.csv',index=False)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    names=list(a.CONDITIONS)
    for ax,mode,title in zip(axes,['random','scaffold'],['随机划分','骨架划分']):
        s=summary[summary.split==mode].set_index('condition').loc[names]
        ax.errorbar(s.auroc_mean,np.arange(len(names)),xerr=s.auroc_std,fmt='o',capsize=4,color='#176b87')
        ax.set_yticks(np.arange(len(names)),[LABELS[n] for n in names]);ax.invert_yaxis()
        ax.set_xlabel('AUROC（均值 ± 三次划分间标准差）');ax.set_title(title);ax.grid(axis='x',alpha=.2)
        ax.set_xlim(.72,.89)
    fig.savefig(out/'ablation_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,feature in zip(axes.flat,a.FEATURES):
        for mode,color in [('random','#176b87'),('scaffold','#cf6b32')]:
            sub=errors[(errors.split==mode)&(errors.condition=='fp_only')&(errors.feature==feature)].sort_values('bin')
            mode_label={'random':'随机','scaffold':'骨架'}[mode]
            ax.plot(sub.bin,sub.fpr_mean,'o-',color=color,label=mode_label+' FPR')
            ax.plot(sub.bin,sub.fnr_mean,'s--',color=color,label=mode_label+' FNR')
        ax.set_title(feature);ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.set_xlabel('按训练集分位点划分的特征区间（从低到高）')
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center')
    axes.flat[-1].text(.05,.15,'FPR：真实负类被误报为正类\nFNR：真实正类被漏报为负类\n区间边界与样本数见配套数据',transform=axes.flat[-1].transAxes)
    fig.savefig(out/'error_by_structure.png',dpi=180);plt.close(fig)
    tab=[]
    for name in names:
        row={'输入':LABELS[name]}
        for mode,label in [('random','随机'),('scaffold','骨架')]:
            s=summary[(summary.split==mode)&(summary.condition==name)].iloc[0]
            row[label+' AUROC']=f'{s.auroc_mean:.4f} ± {s.auroc_std:.4f}'
            row[label+' Brier']=f'{s.brier_mean:.4f}'
        tab.append(row)
    lines=['# 错误分析与结构特征消融','', '日期：2026-10-04。此轮完成 90 次新训练，复用 18 个原网络基线，共比较 108 个模型结果。所有模型与预处理已重新加载核验。','',
           '**本轮发现：**五项整体结构描述符可以补充分子指纹的信息。全部加入后，两种划分的平均 AUROC、Brier 和 log loss 均改善；每种划分的三个重复中 AUROC 均提高。移除氢键供体与受体后的平均 AUROC 降幅最大。与此同时，选定阈值下的骨架测试漏报率上升，六个最极端的错误案例也未被修正；总体排序能力的提升不等于所有错误类型都改善。','',
           '## 实验问题与设计','',
           '在固定 Morgan 指纹和原网络的条件下，加入分子量、环数、可旋转键数、氢键供体数、氢键受体数，能否改善对历史路径搜索标签的预测？随后按特征组移除，检查增益来自哪里。', '',
           '继续使用同一 20,000 分子样本；随机与骨架两种划分，各采用 101、202、303 三个划分种子，每个设置用 42、43、44 三个训练种子。网络隐藏层为 256/128/64，Dropout 0.2，Adam 学习率 0.001，批大小 64，最多 50 轮，验证损失连续 5 轮不改善时停止。只变输入；参数量随输入维度略变。指纹加五项的参数量为 304,897，原模型为 303,617，增幅约 0.42%。仅描述符模型参数量为 42,753。', '',
           '描述符按各训练子集的均值和标准差标准化，验证及内部测试只应用该变换；移除特征后不重新搜索网络。阈值继续由验证集平衡准确率选取。输入维度不同会改变随机初始化序列，因此相同种子并不代表完全相同初始权重。', '',
           '## 消融结果','',table(pd.DataFrame(tab)),'',
           'AUROC 越高越好，Brier 越低越好。先平均同一划分的三个训练种子，再对三个划分均值求均值与样本标准差。标准差描述划分波动，不是置信区间；重复划分有重叠，未作独立样本显著性检验。','',
           '![消融比较](ablation_comparison.png)','', '### 全部特征相对仅指纹的配对变化','']
    for mode,label in [('random','随机'),('scaffold','骨架')]:
        r=cs[(cs.split==mode)&(cs.condition=='fp_all')&(cs.reference=='fp_only')].iloc[0]
        raw=contrasts[(contrasts.split==mode)&(contrasts.condition=='fp_all')&(contrasts.reference=='fp_only')]
        lines.append(f'- {label}划分：AUROC 变化 {r.auroc_mean:+.4f}（三个配对差值标准差 {r.auroc_std:.4f}），Brier 变化 {r.brier_mean:+.4f}，log loss 变化 {r.log_loss_mean:+.4f}；AUROC 在 {int((raw.auroc>0).sum())}/3 个划分上提高。')
    lines+=['','### 移除哪组特征后发生变化','',
            '下表为移除某组后相对“指纹＋全部五项”的 AUROC 变化；负数意味着移除后变差。组间可能有冗余，不能解读为化学因素的因果效应。','']
    removal=[]
    for name in ['fp_no_mass','fp_no_topology','fp_no_hbond']:
        rr={'移除组':LABELS[name]}
        for mode,label in [('random','随机'),('scaffold','骨架')]:
            r=cs[(cs.split==mode)&(cs.condition==name)&(cs.reference=='fp_all')].iloc[0]
            rr[label+' ΔAUROC']=f'{r.auroc_mean:+.4f} ± {r.auroc_std:.4f}'
        removal.append(rr)
    lines+=[table(pd.DataFrame(removal)),'','### 验证集选型记录','',
            '全部特征方案不是预先保证的赢家。以下选择依据三个训练种子的平均验证 log loss，在新模型内部测试评分前保存；还应结合原有试验与新特征试验的开发性质理解。','',
            table(pd.read_csv(out/'validation_selection.csv').query('selected == True')[['split_seed','split','condition','log_loss']].round(4)),'',
            '## 错误分析','',
            '错误分析以固定的原网络（仅指纹）为主，并保存增加全部特征后的对应记录。FP 表示原标签为“未找到路径”却预测为“找到”，FN 表示原标签为“找到路径”却预测为“未找到”。阈值依验证集确定，并不固定为 0.5。','']
    overall=errors[errors.feature=='all']
    er=[]
    for r in overall.itertuples():
        er.append({'划分':r.split,'输入':LABELS[r.condition],'每次测试平均负类数':f'{r.n_negative_mean:.1f}',
                   '每次测试平均正类数':f'{r.n_positive_mean:.1f}',
                   'FPR':f'{r.fpr_mean:.1%} ± {r.fpr_std:.1%}','FNR':f'{r.fnr_mean:.1%} ± {r.fnr_std:.1%}'})
    lines+=[table(pd.DataFrame(er)),'',
            '骨架划分中，增加特征后 FPR 从 24.6% 降至 20.1%，FNR 从 20.3% 升至 22.2%。两种模型各自用验证集选择阈值，因此这是完整预测流程的比较，不能全部归因为特征表示本身。AUROC 衡量跨阈值的排序能力，它提高并不保证当前阈值下的误报和漏报同时下降。','',
            '### 结构分组','',
            '每个划分用训练集的 25%、50%、75% 分位点设边界，重复边界合并；B1 至 B4 表示从低到高的区间，离散计数的区间不一定各含四分之一分子。边界详见 training_quantile_bins.csv；错误率分母分别为真实负类与真实正类。原始表标记分母不足 30 的组，稀疏组仅作描述，不作稳健结论。跨划分同一 B 编号是相对区间，实际边界可能不同。','',
            '![结构分组错误率](error_by_structure.png)','',
            '各描述符在骨架划分下最低与最高非空区间的比较如下。正负类别比例以及描述符之间的相关性都会影响图形，不能据此断言某个化学结构导致错误。','']
    extrema=[]
    for feature in a.FEATURES:
        sub=errors[(errors.condition=='fp_only')&(errors.split=='scaffold')&(errors.feature==feature)&(errors.n_mean>0)].sort_values('bin')
        for r in [sub.iloc[0],sub.iloc[-1]]:
            extrema.append({'特征':feature,'区间':r['bin'],'平均正/负类数':f'{r.n_positive_mean:.0f}/{r.n_negative_mean:.0f}',
                            'FPR':f'{r.fpr_mean:.1%}','FNR':f'{r.fnr_mean:.1%}'})
    lines+=[table(pd.DataFrame(extrema)),'',
            '较高环数、供体数、受体数区间的漏报更突出：骨架划分中，环数最高组的 FNR 为 30.9%，最低组为 16.8%；供体数最高组为 31.2%，最低组为 17.4%。分子量也呈类似方向，可旋转键数则没有明显的单调趋势。这提示原模型在部分整体结构特征较高的分子上更容易给出负预测，但尚不能区分训练覆盖、标签机制和特征表示的作用。','',
            '### 六个可人工复核的分子','',
            '从骨架划分的内部测试预测中，按 source_id 去重；先平均同一划分的三个训练种子，再平均该分子出现过的测试划分。要求基线在所有实际被留出的预测中均出错，分别展示三个最极端 FP 与 FN。没有把该分子在训练集时的预测混入。选择最极端案例会放大错误印象，它们不是随机抽取的代表性样本。模型概率未经额外校准，不能直接视作化学事实的置信度。','',
            '![错误分子结构](error_molecules.png)','']
    ct=cases[['source_id','error_type','label','probability','augmented_probability','heldout_partitions','MolWt','RingCount','RotatableBonds','HBD','HBA']].copy()
    lines+=[table(ct.round(3)),'',
            '增加全部特征后，这六个案例在各自所有留出预测中仍然出错。三个 FN 的分子量约为 468–658、环数为 4–7，与分组统计中的漏报倾向相呼应；但这是挑选出的极端案例，不能推广为所有多环分子的规律。','',
            '这些结构图与统计可支持课程展示中的个案讨论，但当前数据不含具体失败搜索日志、反应模板覆盖与实验验证，不能把误报归因为某一步反应，也不能据此宣称找到了可行合成路线。','',
            '## 边界与后续使用','',
            '- 这是同一训练来源上的开发阶段消融。内部测试已在之前比较中使用并影响后续研究方向；正式测试集仍未用于性能评估。',
            '- 模型学习的是特定规划器、模板库、原料库和搜索限制下的历史“找到路径”标签；负标签不等于化学上不可合成。',
            '- 本轮控制了网络、划分和训练流程，但移除组的效果仍包含优化随机性与特征冗余；未逐一分离环数与旋转键、供体与受体。',
            '- 正式测试前需固定完整方法和评价计划；若报告声称对新骨架泛化，必须另明确相应骨架隔离评估，官方随机划分成绩不能直接支持该声称。','',
            '## 文件与核验','',
            'protocol.json 记录预先固定的条件、标签范围、哈希和评价流程；verification.json 记录全部 108 个模型的重新加载、验证及内部测试预测复算、阈值复算、描述符复算和仅训练集标准化检查。ablation_summary.csv 为总表；paired_split_differences.csv 为逐划分配对差值；error_groups_per_run.csv 为带正负类分母的错误统计；unique_molecule_errors.csv 为按分子去重后的记录；error_case_studies.csv 为图中六个案例。','',
            'AI 辅助记录：本轮由 AI 辅助实现并在本机执行特征消融、复核及报告生成。提交作业前，学生应能解释描述符、分组消融、FPR/FNR、开发集复用和标签边界。']
    (out/'错误分析与结构特征消融报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[['split','condition','auroc_mean','brier_mean']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=a.OUT)
    out=parser.parse_args().output
    torch.set_num_threads(2)
    sample=pd.read_csv(a.DATA/'pilot_sample.csv')
    cache=np.load(a.DATA/'pilot_features.npz');fp,y=cache['x'].astype(np.float32),cache['y']
    desc=pd.read_csv(out/'descriptors.csv');assert np.array_equal(desc.source_id,sample.source_id)
    d=desc[a.FEATURES].to_numpy()
    results,val=pd.read_csv(out/'metrics.csv'),pd.read_csv(out/'validation_metrics.csv')
    verify(out,sample,fp,y,d,results,val)
    errors,cases,gr=error_analysis(out,sample,d)
    report(out,results,val,errors,cases,gr)


if __name__=='__main__':
    main()
