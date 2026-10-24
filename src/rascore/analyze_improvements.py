"""Independent checkpoint verification, paired error transitions and report."""
import argparse
import json
from pathlib import Path
import improvement_experiments as e
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from analyze_feature_ablation import table

LABELS={'binary_standard':'二值指纹＋原正则','binary_regularized':'二值指纹＋强正则',
        'count_standard':'计数指纹＋原正则','count_regularized':'计数指纹＋强正则'}
METRICS=['auroc','ap','brier','log_loss','balanced_accuracy','fpr','fnr']


def read(path):return pd.read_csv(path,float_precision='round_trip')


def row_for(frame,ss,mode,condition,seed,policy):
    rows=frame[(frame.split_seed==ss)&(frame.split==mode)&(frame.condition==condition)&(frame.seed==seed)&(frame.policy==policy)]
    assert len(rows)==1
    return rows.iloc[0]


def verify(out,sample,binary,counts,d,y,val,results):
    protocol=json.loads((out/'protocol.json').read_text())
    assert e.a.sha(e.a.ROOT/'improvement_experiments.py')==protocol['source_sha256']
    for file,h in protocol['dependency_sha256'].items():assert e.a.sha(e.a.ROOT/file)==h
    assert e.a.sha(e.a.DATA/'pilot_sample.csv')==protocol['sample_sha256']
    assert e.a.sha(e.a.DATA/'pilot_features.npz')==protocol['fingerprint_sha256']
    assert (sample.source_split=='train').all() and sample.source_id.is_unique
    assert np.array_equal(y,sample.activity)
    assert np.array_equal(counts,e.count_features(sample)) and np.array_equal(counts>0,binary>0)
    np.testing.assert_allclose(d,e.a.descriptors(sample),atol=1e-10,rtol=1e-12)
    assert len(results)==len(val)==192
    keys=['split_seed','split','condition','seed','policy']
    assert not results.duplicated(keys).any() and not val.duplicated(keys).any()
    info=read(out/'training_info.csv');checked=0
    for ss in e.a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            splitpath=e.a.MAIN/f'split_{ss}'/f'{mode}_indices.npz'
            assert e.a.sha(splitpath)==protocol['split_sha256'][f'{ss}/{mode}']
            idx=np.load(splitpath);tr,va,te=[idx[k] for k in ['train','validation','internal_test']]
            for got,expected in zip([tr,va,te],e.a.pilot.split_indices(sample,y,mode,ss)):assert np.array_equal(got,expected)
            assert len(np.unique(np.concatenate([tr,va,te])))==len(sample)
            if mode=='scaffold':assert not set(sample.iloc[tr].scaffold)&set(sample.iloc[te].scaffold)
            part=out/f'split_{ss}'/mode;sc=json.loads((part/'scaler.json').read_text())
            assert sc['n_train']==len(tr)
            np.testing.assert_allclose(sc['mean'],d[tr].mean(0),atol=1e-10)
            np.testing.assert_allclose(sc['scale'],d[tr].std(0),atol=1e-10)
            scaled=(d-np.array(sc['mean']))/np.array(sc['scale'])
            for condition in e.CONDITIONS:
                x=e.inputs(binary,counts,scaled,condition);vp_list=[];tp_list=[]
                for seed in e.a.TRAIN_SEEDS:
                    folder=part/f'seed{seed}'/condition;model,cp,path=e.load_model(folder,condition)
                    assert cp['seed']==seed
                    if condition!='binary_standard':assert cp['condition']==condition
                    hist=read(path.parent/'history.csv')
                    assert cp['best_epoch']==hist.loc[hist.validation_loss.idxmin(),'epoch']
                    inf=info[(info.split_seed==ss)&(info.split==mode)&(info.condition==condition)&(info.seed==seed)].iloc[0]
                    assert inf.best_epoch==cp['best_epoch'] and inf.parameter_count==sum(p.numel() for p in model.parameters())
                    vp,tp=e.a.predict(model,x[va]),e.a.predict(model,x[te]).astype(np.float64)
                    vp_list.append(vp.astype(np.float64));tp_list.append(tp)
                    for name,indices,p,metrics_frame in [('validation',va,vp,val),('internal_test',te,tp,results)]:
                        saved=read(folder/f'{name}_predictions.csv')
                        assert np.array_equal(saved.source_id,sample.iloc[indices].source_id)
                        assert np.array_equal(saved.label,y[indices])
                        np.testing.assert_allclose(p,saved.probability,atol=1e-7,rtol=1e-6)
                        for policy in e.POLICIES:
                            t=e.threshold(y[va],vp,policy);row=row_for(metrics_frame,ss,mode,condition,seed,policy)
                            np.testing.assert_allclose(t,row.threshold,atol=1e-14,rtol=1e-14)
                            for m,v in e.metrics(y[indices],p,t).items():np.testing.assert_allclose(v,row[m],atol=1e-7,rtol=1e-6)
                        if condition=='binary_standard':
                            prior=read(path.parent/f'{name}_predictions.csv')
                            np.testing.assert_allclose(p,prior.probability,atol=1e-7,rtol=1e-6)
                    checked+=1
                vp,tp=np.mean(vp_list,0),np.mean(tp_list,0)
                for name,indices,p,frame in [('validation',va,vp,val),('internal_test',te,tp,results)]:
                    saved=read(part/'ensemble'/condition/f'{name}_predictions.csv')
                    assert np.array_equal(saved.source_id,sample.iloc[indices].source_id) and np.array_equal(saved.label,y[indices])
                    np.testing.assert_allclose(p,saved.probability,atol=1e-12,rtol=1e-12)
                    for policy in e.POLICIES:
                        t=e.threshold(y[va],vp,policy);row=row_for(frame,ss,mode,condition,-1,policy)
                        np.testing.assert_allclose(t,row.threshold,atol=1e-14,rtol=1e-14)
                        for m,v in e.metrics(y[indices],p,t).items():np.testing.assert_allclose(v,row[m],atol=1e-7,rtol=1e-6)
            print(f'Verified {ss}/{mode}',flush=True)
    choices=read(out/'validation_selection.csv')
    for r in choices.itertuples():
        vr=row_for(val,r.split_seed,r.split,r.condition,-1,'balanced')
        np.testing.assert_allclose(r.log_loss,vr.log_loss,atol=1e-12)
        block=val[(val.split_seed==r.split_seed)&(val.split==r.split)&(val.kind=='ensemble')&(val.policy=='balanced')]
        assert r.selected==(vr.log_loss==block.log_loss.min())
    assert (val[val.policy=='recall90'].recall>=.9).all()
    e.a.pilot.save_json(out/'verification.json',{'passed':True,'models_checked':checked,'new_models':54,'reused_models':18,
        'ensembles_checked':24,'validation_and_test_metric_rows_checked':384,'raw_counts_recomputed':True,
        'descriptors_recomputed':True,'training_only_scalers_checked':True,'thresholds_and_selection_recomputed':True,
        'original_partitions_verified':True,'official_test_used':False})


def errors(out,sample,d,y,results,choices):
    records,groups,transitions=[],[],[]
    for ss in e.a.SPLIT_SEEDS:
        for mode in ['random','scaffold']:
            idx=np.load(e.a.MAIN/f'split_{ss}'/f'{mode}_indices.npz');tr,te=idx['train'],idx['internal_test']
            chosen=choices[(choices.split_seed==ss)&(choices.split==mode)&choices.selected].condition.iloc[0]
            part=out/f'split_{ss}'/mode
            base=read(part/'ensemble'/'binary_standard'/'internal_test_predictions.csv')
            selected=read(part/'ensemble'/chosen/'internal_test_predictions.csv')
            bins=[('all','all',np.ones(len(te),bool),np.nan)]
            for f in ['MolWt','RingCount','HBD','HBA']:
                j=e.a.FEATURES.index(f);cut=float(np.quantile(d[tr,j],.75));high=d[te,j]>cut
                bins.extend([(f,'high',high,cut),(f,'rest',~high,cut)])
            for policy in e.POLICIES:
                br=row_for(results,ss,mode,'binary_standard',-1,policy)
                sr=row_for(results,ss,mode,chosen,-1,policy)
                bp=base.probability.to_numpy()>=br.threshold;sp=selected.probability.to_numpy()>=sr.threshold
                bw,sw=bp!=y[te],sp!=y[te]
                frame=sample.iloc[te][['source_id','canonical_smiles']].reset_index(drop=True).copy()
                frame['split_seed'],frame['split'],frame['policy'],frame['selected_condition']=ss,mode,policy,chosen
                frame['label']=y[te];frame['baseline_probability']=base.probability
                frame['selected_probability']=selected.probability;frame['baseline_wrong']=bw;frame['selected_wrong']=sw
                frame['transition']=np.select([bw&~sw,~bw&sw,bw&sw],['corrected','new_error','persistent_error'],default='both_correct')
                records.append(frame)
                transitions.append({'split_seed':ss,'split':mode,'policy':policy,'selected_condition':chosen,
                    'n':len(te),'corrected':int((bw&~sw).sum()),'new_errors':int((~bw&sw).sum()),
                    'persistent_errors':int((bw&sw).sum()),'baseline_errors':int(bw.sum()),'selected_errors':int(sw.sum())})
                for feature,b,mask,cut in bins:
                    for role,pred in [('baseline',bp),('selected',sp)]:
                        n0=int(((y[te]==0)&mask).sum());n1=int(((y[te]==1)&mask).sum())
                        fp=int(((y[te]==0)&pred&mask).sum());fn=int(((y[te]==1)&~pred&mask).sum())
                        groups.append({'split_seed':ss,'split':mode,'policy':policy,'role':role,'feature':feature,'group':b,
                            'training_q75':cut,'n_negative':n0,'n_positive':n1,'fp':fp,'fn':fn,
                            'fpr':fp/n0 if n0 else np.nan,'fnr':fn/n1 if n1 else np.nan,
                            'sparse_negative':n0<30,'sparse_positive':n1<30})
    obs=pd.concat(records,ignore_index=True);obs.to_csv(out/'error_observations.csv',index=False)
    gr=pd.DataFrame(groups);gr.to_csv(out/'structure_group_errors.csv',index=False)
    trans=pd.DataFrame(transitions);trans.to_csv(out/'error_transitions_per_split.csv',index=False)
    unique=obs.groupby(['split','policy','source_id']).agg(label=('label','first'),canonical_smiles=('canonical_smiles','first'),
        baseline_probability=('baseline_probability','mean'),selected_probability=('selected_probability','mean'),
        baseline_error_fraction=('baseline_wrong','mean'),selected_error_fraction=('selected_wrong','mean'),
        heldout_partitions=('split_seed','nunique')).reset_index()
    unique['gain_wrong_probability']=np.where(unique.label==1,unique.selected_probability-unique.baseline_probability,
                                               unique.baseline_probability-unique.selected_probability)
    unique.to_csv(out/'unique_molecule_errors.csv',index=False)
    old=read(e.a.OUT/'error_case_studies.csv')
    case=unique[(unique.split=='scaffold')&(unique.policy=='balanced')]
    follow=old[['source_id','error_type','probability','augmented_probability']].merge(case,on='source_id',validate='one_to_one')
    follow.to_csv(out/'previous_six_case_followup.csv',index=False)
    blocks=[]
    filters={
        'corrected':(case.baseline_error_fraction==1)&(case.selected_error_fraction==0),
        'new_error':(case.baseline_error_fraction==0)&(case.selected_error_fraction==1),
        'persistent':(case.baseline_error_fraction==1)&(case.selected_error_fraction==1)}
    for name,mask in filters.items():
        sub=case[mask].copy()
        if name=='corrected':sub=sub.nlargest(2,'gain_wrong_probability')
        elif name=='new_error':sub=sub.nsmallest(2,'gain_wrong_probability')
        else:
            sub['wrong_probability']=np.where(sub.label==0,sub.selected_probability,1-sub.selected_probability)
            sub=sub.nlargest(2,'wrong_probability')
        sub['example_type']=name;blocks.append(sub)
    examples=pd.concat(blocks,ignore_index=True);examples.to_csv(out/'error_examples.csv',index=False)
    legends=[f'{r.example_type} ID {r.source_id}\nlabel={r.label}, splits={r.heldout_partitions}\np: {r.baseline_probability:.3f} -> {r.selected_probability:.3f}' for r in examples.itertuples()]
    img=Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in examples.canonical_smiles],molsPerRow=2,
                            subImgSize=(530,370),legends=legends)
    img.save(str(out/'error_examples.png'))
    # Algebraic consistency for paired changes and unique-source examples.
    assert (trans.baseline_errors-trans.selected_errors==trans.corrected-trans.new_errors).all()
    assert not examples.source_id.duplicated().any() and len(follow)==6
    return gr,trans,follow,examples


def summarize(out,results,choices):
    level=results.groupby(['split_seed','split','kind','condition','policy'])[METRICS].mean().reset_index()
    level.to_csv(out/'split_level_means.csv',index=False)
    summary=level.groupby(['split','kind','condition','policy'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
    summary.to_csv(out/'summary.csv',index=False)
    contrasts=[]
    for (ss,mode,kind,policy),block in level.groupby(['split_seed','split','kind','policy']):
        values=block.set_index('condition')
        comparisons=[('count_standard','binary_standard'),('count_regularized','binary_regularized'),
                     ('binary_regularized','binary_standard'),('count_regularized','count_standard')]
        for target,ref in comparisons:
            contrasts.append({'split_seed':ss,'split':mode,'kind':kind,'policy':policy,'condition':target,'reference':ref,
                              **{m:float(values.loc[target,m]-values.loc[ref,m]) for m in METRICS}})
    pd.DataFrame(contrasts).to_csv(out/'factorial_paired_differences.csv',index=False)
    selected=results.merge(choices[choices.selected][['split_seed','split','condition']],on=['split_seed','split','condition'])
    selected=selected[selected.kind=='ensemble'];selected.to_csv(out/'validation_selected_ensemble_metrics.csv',index=False)
    delta=[]
    for r in selected.itertuples():
        baseline=row_for(results,r.split_seed,r.split,'binary_standard',-1,r.policy)
        delta.append({'split_seed':r.split_seed,'split':r.split,'policy':r.policy,'selected_condition':r.condition,
                      **{m:getattr(r,m)-baseline[m] for m in METRICS}})
    pd.DataFrame(delta).to_csv(out/'selected_vs_baseline_ensemble_differences.csv',index=False)
    return summary,selected


def report(out,summary,selected,gr,trans,follow,examples):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    names=list(e.CONDITIONS)
    for ax,mode,title in zip(axes,['random','scaffold'],['随机划分','骨架划分']):
        for kind,offset,color,label in [('single',-.08,'#176b87','单模型（三种子均值）'),('ensemble',.08,'#cf6b32','三个种子平均预测')]:
            sub=summary[(summary.split==mode)&(summary.kind==kind)&(summary.policy=='balanced')].set_index('condition').loc[names]
            ax.errorbar(sub.auroc_mean,np.arange(4)+offset,xerr=sub.auroc_std,fmt='o',capsize=3,color=color,label=label)
        ax.set_yticks(np.arange(4),[LABELS[n] for n in names]);ax.invert_yaxis();ax.set_xlim(.85,.90)
        ax.set_xlabel('AUROC（均值 ± 三次划分间标准差）');ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9)
    fig.savefig(out/'improvement_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,mode,title in zip(axes,['random','scaffold'],['随机划分','骨架划分']):
        for role,label,color in [('baseline','原二值集成','#176b87'),('selected','验证集选出方案','#cf6b32')]:
            sub=gr[(gr.split==mode)&(gr.role==role)&(gr.feature=='all')].groupby('policy')[['fpr','fnr']].mean()
            ax.plot(sub.fpr,sub.fnr,'o-',color=color,label=label)
            for policy,r in sub.iterrows():ax.annotate('召回90%' if policy=='recall90' else '平衡阈值',(r.fpr,r.fnr),xytext=(3,5),textcoords='offset points',fontsize=8)
        ax.set_xlabel('误报率 FPR');ax.set_ylabel('漏报率 FNR');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
        ax.set_xlim(.19,.37);ax.set_ylim(.07,.23)
    fig.savefig(out/'threshold_tradeoff.png',dpi=180);plt.close(fig)
    rows=[]
    for name in names:
        r={'输入与正则化':LABELS[name]}
        for mode,title in [('random','随机'),('scaffold','骨架')]:
            s=summary[(summary.split==mode)&(summary.kind=='single')&(summary.condition==name)&(summary.policy=='balanced')].iloc[0]
            r[title+' AUROC']=f'{s.auroc_mean:.4f} ± {s.auroc_std:.4f}';r[title+' Brier']=f'{s.brier_mean:.4f}'
        rows.append(r)
    choices=read(out/'validation_selection.csv').query('selected == True')
    lines=['# 第二轮改进、重复实验与错误分析','',
           '本轮完成 54 次新训练，复用上一轮 18 个“二值指纹＋五项结构特征”模型，形成 24 个三种子等权集成；全部模型、集成预测、阈值与指标均已核验。正式测试集未使用。','',
           '**主要发现：计数指纹的增益比单纯加强正则化更一致。**在两种正则化水平、两种划分的三个重复中，计数指纹均提高单模型平均 AUROC。验证集在六个划分中全部选择了计数方案，但原正则与强正则各有入选，因此没有证据支持一律加大正则。集成和阈值调整改善了部分错误，仍保留明确的误报/漏报取舍。','',
           '## 改进设计','',
           '原方案对一些高环数、高供体/受体数的正类分子更容易漏报。本轮测试一个可验证的假设：记录子结构出现次数，是否比只记录有无更有帮助？使用同一 1024 位、半径 2、不含手性的 Morgan 哈希，计数作 log(1+count) 变换。程序检查 20,000 个分子的非零位置与原二值指纹逐位一致，因此主要变化是计数信息；哈希碰撞和不含手性的限制仍存在。', '',
           '四种设置为二值/计数指纹 × 原正则/强正则，均加入分子量、环数、可旋转键、供体数、受体数。原正则为 Dropout 0.2、权重衰减 0；强正则为 Dropout 0.4、权重衰减 0.0001。两项一起变化，不能分离它们各自的作用。所有设置均为 1029→256→128→64→1，304,897 个参数，同种子有相同初始权重。', '',
           '沿用 20,000 个样本、随机/骨架两种划分、101/202/303 三个划分种子和 42/43/44 三个训练种子。训练设置仍为 Adam、学习率 0.001、batch 64、最多 50 轮，验证 BCE 连续 5 轮不改善则早停；描述符标准化只拟合训练子集。', '',
           '## 单模型重复结果','',table(pd.DataFrame(rows)),'',
           '先平均同一划分的三个训练种子，再统计三次划分间均值和样本标准差；标准差不是置信区间。相同分子可出现在多个重复的内部测试中，因此不将九个运行视为九份独立测试数据。','',
           '![改进比较](improvement_comparison.png)','',
           '在原正则下，计数相对二值的配对 AUROC 增益为随机 +0.0057、骨架 +0.0043；在强正则下分别为 +0.0053、+0.0055。四组对比各自的三个划分差值均为正。相反，二值输入上单纯加强正则只使随机平均 AUROC 增加 0.0011，骨架平均值下降 0.0003。以上为描述性重复结果，未进行独立样本显著性检验。','',
           '## 集成与验证集选型','',
           '集成是三个训练种子预测概率的等权平均，没有训练额外权重。其推理和模型存储成本约为单模型的三倍。每个划分只根据集成验证 log loss 从四种设置选择，选型和阈值在本轮内部测试评分前保存。下表展示选择记录，不能依据内部测试事后更换方案。','',
           table(choices[['split_seed','split','condition','log_loss']].round(4)),'']
    st=[]
    for mode,title in [('random','随机'),('scaffold','骨架')]:
        for role in ['baseline','selected']:
            if role=='baseline':
                s=summary[(summary.split==mode)&(summary.kind=='ensemble')&(summary.condition=='binary_standard')&(summary.policy=='balanced')].iloc[0]
                auc,auc_sd,brier,loss=s.auroc_mean,s.auroc_std,s.brier_mean,s.log_loss_mean
            else:
                s=selected[(selected.split==mode)&(selected.policy=='balanced')];auc,auc_sd,brier,loss=s.auroc.mean(),s.auroc.std(),s.brier.mean(),s.log_loss.mean()
            st.append({'划分':title,'方案':'原二值集成' if role=='baseline' else '验证集选出方案',
                       'AUROC':f'{auc:.4f} ± {auc_sd:.4f}','Brier':f'{brier:.4f}','log loss':f'{loss:.4f}'})
    lines+=[table(pd.DataFrame(st)),'',
            '验证集选出方案相对原二值集成，在全部六个配对划分中均提高 AUROC，并降低 Brier 和 log loss；但平衡准确率并非六次均提高。集成的比较使用相同的三个模型预算，不能将单模型与集成之间的全部差异归因于指纹改进。新训练的单次平均耗时约为 3.7–4.0 秒，54 次新训练计时合计约 206 秒，不含特征计算、复核与报告生成；计时受本机当时负载影响。','',
            '## 减少漏报的代价','',
            '比较两种只用验证集选择的阈值：①验证平衡准确率最大；②保留至少 90% 验证正类的最高阈值。第二种是探索性取舍，不是用户设定的应用要求，也不保证内部测试召回率达到 90%。同一模型只调整阈值不会改变 AUROC、Brier 或 log loss。','']
    et=[]
    for (mode,policy,role),block in gr[gr.feature=='all'].groupby(['split','policy','role']):
        et.append({'划分':mode,'阈值':'平衡' if policy=='balanced' else '验证召回90%',
                   '方案':'原二值集成' if role=='baseline' else '验证集选出方案',
                   '误报率':f'{block.fpr.mean():.1%}','漏报率':f'{block.fnr.mean():.1%}'})
    lines+=[table(pd.DataFrame(et)),'',
            '对选出方案，骨架划分由平衡阈值转为验证召回 90% 阈值后，漏报率从 18.9% 降至 10.7%，误报率从 22.0% 升至 32.6%。因此可以减少漏掉历史正标签分子的比例，但需要对更多负标签分子继续检查；不能只报告召回率而省略代价。','',
            '![阈值取舍](threshold_tradeoff.png)','',
            '## 配对错误变化','',
            '在同一测试分子、同一种阈值政策下，将验证集选出方案与原二值集成比较。修正数指原方案错而新方案对，新增错误数指原方案对而新方案错。以下为每次划分的平均分子数，不把跨划分重复的分子当作独立个体累计。','',
            table(trans.groupby(['split','policy'])[['n','corrected','new_errors','persistent_errors']].mean().round(1).reset_index()),'',
            '骨架划分的平衡阈值下，每次平均修正 101.7 个错误、新增 75.7 个错误，净减少约 26 个，仍有约 506.7 个错误被两种方案共同保留。总体改善主要来自一部分样本发生改变，没有消除系统性的困难样本。','',
            '### 高分子量及高计数组的漏报','',
            '高组定义为超过该训练子集的第 75 百分位；其余为对照组。离散计数的高组未必恰占 25%。表中只列骨架划分的平衡阈值结果，完整误报率、漏报率和分母见 structure_group_errors.csv，分母不足 30 的组已标记。','']
    g=gr[(gr.split=='scaffold')&(gr.policy=='balanced')&(gr.feature!='all')]
    gt=g.groupby(['feature','group','role'])[['n_positive','n_negative','fpr','fnr']].mean()
    gt_rows=[]
    for feature,label in [('MolWt','分子量'),('RingCount','环数'),('HBD','氢键供体数'),('HBA','氢键受体数')]:
        before,after=gt.loc[(feature,'high','baseline')],gt.loc[(feature,'high','selected')]
        rest=gt.loc[(feature,'rest','selected')]
        gt_rows.append({'特征':label,'高组平均正类数':f'{before.n_positive:.1f}',
                        '高组原漏报率':f'{before.fnr:.1%}','高组新漏报率':f'{after.fnr:.1%}',
                        '其余组新漏报率':f'{rest.fnr:.1%}'})
    lines+=[table(pd.DataFrame(gt_rows)),'',
            '高环数组漏报率从 40.0% 降至 36.4%，仍高于其余组的 16.0%；高供体数和高受体数组也存在类似差距。这些分组是相关性描述，特征间相互关联，不能解释为某个化学因素导致模型失败。这里的基线是“二值指纹＋五项特征”的集成，与上一轮仅指纹单模型的错误分组数值不能直接混用。','',
            '### 上一轮六个极端错误的追踪','',
            '下表保留上一轮的六个分子。error_fraction 为该分子实际被留出时的错误比例，0 表示全部修正，1 表示始终错误；只比较本轮平衡阈值下的集成。不能将极端个案推广为总体规律。','',
            table(follow[['source_id','error_type','label','baseline_probability','selected_probability','baseline_error_fraction','selected_error_fraction','heldout_partitions']].round(3)),'',
            '六个旧案例在本轮平衡阈值下全部仍然错误。它们不应因总体分数提高而被删除；后续若要解释具体失败原因，需要规划器日志、模板覆盖或可靠的外部核查，当前数据不足以给出路线层面的结论。','',
            '### 修正、新增与持续错误的结构例子','',
            '每类展示两个骨架内部测试分子，按预测变化或错误程度挑选。分子按 source_id 去重；只汇总它被留出时的预测。挑选极端案例会放大局部表现，模型概率也不等于实验事实的置信度。','',
            '![错误变化案例](error_examples.png)','',
            table(examples[['source_id','example_type','label','baseline_probability','selected_probability','heldout_partitions']].round(3)),'',
            '## 研究边界','',
            '这是在已经看过的内部测试结果上提出的第二轮改进，属于开发过程；验证选型仍有必要，但不能把本轮结果称为完全盲测。所有正式结论需要在固定方法和评价方案后再用保留数据检验。', '',
            '标签表示特定历史规划器配置是否找到路径。错误可能与模型表示、样本覆盖、规划器模板/原料库和搜索限制有关；当前数据没有失败路线日志，不能归因为具体反应，也不能把预测标签当作实验合成结论。', '',
            '本轮不继续根据内部测试成绩无限增加超参数搜索。建议结合验证选型、重复差值、误报/漏报成本和电脑资源固定候选方法，再完成报告与保留测试计划。','',
            '## 复现材料','',
            'protocol.json 保存预先固定的设计和来源哈希；verification.json 为核验记录；summary.csv 为汇总；factorial_paired_differences.csv 分离在不同正则水平下的指纹变化，以及不同指纹下的正则变化；selected_vs_baseline_ensemble_differences.csv 保存逐划分配对差值；error_transitions_per_split.csv、structure_group_errors.csv 和 previous_six_case_followup.csv 保存错误分析。','',
            'AI 辅助记录：本轮代码、重复训练、结果复核和报告由 AI 辅助完成，均在本机实际运行。学生提交前应理解计数指纹、正则化、集成、阈值取舍和开发数据复用的边界。']
    (out/'第二轮改进与错误分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(st).to_string(index=False),flush=True)
    print(pd.DataFrame(et).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=e.OUT)
    out=parser.parse_args().output;torch.set_num_threads(2)
    sample=read(e.a.DATA/'pilot_sample.csv');cache=np.load(e.a.DATA/'pilot_features.npz')
    binary,y=cache['x'].astype(np.float32),cache['y'];counts=np.load(out/'count_features.npz')['counts']
    dd=read(out/'descriptors.csv');assert np.array_equal(dd.source_id,sample.source_id)
    d=dd[e.a.FEATURES].to_numpy();results,val=read(out/'metrics.csv'),read(out/'validation_metrics.csv')
    verify(out,sample,binary,counts,d,y,val,results)
    choices=read(out/'validation_selection.csv');gr,trans,follow,examples=errors(out,sample,d,y,results,choices)
    summary,selected=summarize(out,results,choices)
    report(out,summary,selected,gr,trans,follow,examples)


if __name__=='__main__':main()
