"""Charts from real validation states and saved official results."""
from pathlib import Path
import csv, shutil, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TEAL,BLUE,ORANGE,INK='#087e8b','#4059ad','#c66b22','#172b45'

def render_figures(audit):
    out=HERE/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.titleweight':'bold','savefig.facecolor':'white'})
    def save(fig,name):
        fig.savefig(out/name,dpi=170,bbox_inches='tight',pad_inches=.22)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,2.8)); ax.set(xlim=(-.2,11.2),ylim=(0,2.8)); ax.axis('off')
    boxes=[('1  Molecules','SMILES + historical label'),('2  Features','1,024 counts\n+ 5 descriptors'),('3  Neural scores','Three frozen MLPs'),('4  Screening','Mean score + threshold')]
    for i,(title,body) in enumerate(boxes):
        left=.08+i*2.8
        ax.add_patch(FancyBboxPatch((left,.65),2.35,1.35,boxstyle='round,pad=.12',facecolor='#edf4f8',edgecolor=TEAL,lw=1.5))
        ax.text(left+1.175,1.55,title,ha='center',weight='bold',fontsize=13)
        ax.text(left+1.175,1.04,body,ha='center',fontsize=10)
        if i<3: ax.annotate('',xy=(left+2.68,1.3),xytext=(left+2.49,1.3),arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
    ax.text(5.5,2.5,'Follow the same three validation molecules',ha='center',fontsize=17,weight='bold')
    ax.text(5.5,.1,'Development selects the method; the reserved test evaluates it.',ha='center',fontsize=12)
    save(fig,'pipeline.png')

    state=audit['states'][1]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[.9,1.3]})
    vals=[state['binary_value'],state['bin_count'],state['log1p_value']]
    axes[0].bar(['Binary','Raw count','log(1 + count)'],vals,color=[BLUE,'#b7c8cf',TEAL],width=.6)
    axes[0].set_ylim(0,12); axes[0].set_ylabel('Encoded value'); axes[0].set_title(f"Real hashed bin {state['example_hashed_bin']}",fontsize=14)
    for i,v in enumerate(vals): axes[0].text(i,v+.3,f'{v:g}',ha='center',weight='bold')
    axes[0].tick_params(axis='x',labelsize=10)
    axes[1].axis('off'); axes[1].set_title('First molecule · global descriptors',fontsize=14,pad=16)
    raw=state['first_molecule_raw_descriptors']; scaled=state['first_molecule_scaled_descriptors']
    labels=['Mol. weight','Rings','Rotatable bonds','H-bond donors','H-bond acceptors']
    rows=[[label,f'{v:.3f}',f'{scaled[k]:.3f}'] for label,(k,v) in zip(labels,raw.items())]
    tab=axes[1].table(cellText=rows,colLabels=['Descriptor','Raw','Standardized'],cellLoc='center',colWidths=[.44,.22,.34],bbox=[0,.08,1,.84])
    tab.auto_set_font_size(False); tab.set_fontsize(11)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#d8e1e8'); cell.set_facecolor('#edf4f8' if r==0 else 'white')
        if r==0: cell.set_text_props(weight='bold')
    fig.suptitle('Preserve repetition and align descriptor scales',fontsize=17,weight='bold',y=1.03)
    fig.tight_layout(); save(fig,'feature_encoding.png')

    layers=audit['states'][2]['seed42_hidden_layers']
    fig=plt.figure(figsize=(11,4.8)); grid=fig.add_gridspec(2,1,height_ratios=[1,2],hspace=.45)
    ax=fig.add_subplot(grid[0]); ax.axis('off'); ax.set(xlim=(0,10),ylim=(0,1))
    for i,width in enumerate([1029,256,128,64,1]):
        x=.2+i*2
        ax.add_patch(FancyBboxPatch((x,.15),1.35,.7,boxstyle='round,pad=.08',facecolor='#edf4f8',edgecolor=TEAL))
        ax.text(x+.675,.5,str(width),ha='center',va='center',weight='bold',fontsize=17)
        if i<4: ax.annotate('',xy=(x+1.9,.5),xytext=(x+1.5,.5),arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
    ax.text(4.9,1.15,'Input → three ReLU layers → output logit → sigmoid',ha='center',fontsize=14)
    ax=fig.add_subplot(grid[1]); data=np.array([l['first_molecule_first_6_activations'] for l in layers])
    im=ax.imshow(data,cmap='YlGnBu',aspect='auto',vmin=0)
    ax.set_xticks(range(6),[f'Unit {i+1}' for i in range(6)])
    ax.set_yticks(range(3),[f"Layer {i+1}\n{l['nonzero_units']}/{l['width']} active" for i,l in enumerate(layers)])
    for (r,c),v in np.ndenumerate(data): ax.text(c,r,f'{v:.3f}',ha='center',va='center',color='white' if v>data.max()*.5 else INK,fontsize=12)
    fig.colorbar(im,ax=ax,label='ReLU activation',fraction=.025,pad=.04)
    fig.suptitle('Actual hidden activations · seed 42 · first molecule',fontsize=17,weight='bold',y=1.04)
    save(fig,'network_activations.png')

    probs=np.array(audit['states'][2]['three_seed_probabilities']); decisions=audit['states'][3]
    fig,ax=plt.subplots(figsize=(11,4))
    for j,seed in enumerate([42,43,44]): ax.scatter(probs[:,j],np.arange(3)+(j-1)*.12,s=70,label=f'Seed {seed}',color=[BLUE,ORANGE,'#8aa3b2'][j],zorder=3)
    means=[r['mean_probability'] for r in decisions]; threshold=decisions[0]['frozen_threshold']
    ax.scatter(means,np.arange(3),marker='D',s=105,color=TEAL,edgecolor='white',label='Mean',zorder=5)
    ax.axvline(threshold,color=INK,ls='--',lw=1.7,label=f'Threshold {threshold:.4f}')
    ax.set_yticks(range(3),[f"ID {r['source_id']} · label {r['historical_label']}" for r in decisions])
    for y,r in enumerate(decisions): ax.text(1.02,y,f"Mean {r['mean_probability']:.3f}\nPriority: {'yes' if r['predicted_label'] else 'no'}",va='center',fontsize=12)
    ax.set(xlim=(0,1),ylim=(-.6,2.6),xlabel='Probability of historical route discovery')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.18); ax.set_title('Three scores become one screening decision',pad=18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=5,fontsize=10,frameon=False)
    save(fig,'ensemble_decisions.png')

    with (ROOT/'results/official_holdout_v1/metrics.csv').open(encoding='utf-8-sig') as f: metrics=list(csv.DictReader(f))
    chosen=[next(r for r in metrics if r['model']=='count_standard' and r['policy']==policy) for policy in ['balanced','recall90']]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title in zip(axes,['fn','fp'],['Missed historical positives','Prioritized historical negatives']):
        vals=[int(r[key]) for r in chosen]; ax.bar(['Balanced','Recall-oriented'],vals,color=[BLUE,TEAL],width=.52)
        for i,v in enumerate(vals): ax.text(i,v+65,f'{v:,}',ha='center',weight='bold',fontsize=16)
        ax.set_ylim(0,3400); ax.set_title(title,fontsize=14); ax.set_ylabel('Molecules'); ax.grid(axis='y',alpha=.12); ax.set_axisbelow(True)
    fig.suptitle('Same count model · different fixed operating points',fontsize=17,weight='bold')
    fig.text(.5,.01,'Recall 81.41% → 90.24%    |    1,323 fewer misses; 573 extra false positives',ha='center',fontsize=13)
    fig.tight_layout(rect=[0,.07,1,.95]); save(fig,'threshold_tradeoff.png')

    with (ROOT/'results/official_holdout_v1/structure_subgroups.csv').open(encoding='utf-8-sig') as f: groups=list(csv.DictReader(f))
    rings=[next(r for r in groups if r['feature']=='RingCount' and r['model']=='count_standard' and r['policy']=='recall90' and r['group']==g) for g in ['rest','high']]
    vals=[100*(1-float(chosen[1]['recall']))]+[100*float(r['fnr']) for r in rings]
    assert round(vals[2],2)==26.51
    fig,ax=plt.subplots(figsize=(10,3.4))
    ax.barh(['All test molecules','Rings ≤ 4','Rings > 4'],vals,color=['#8aa3b2',TEAL,ORANGE],height=.55)
    for i,v in enumerate(vals): ax.text(v+.6,i,f'{v:.2f}%',va='center',weight='bold',fontsize=16)
    ax.set(xlim=(0,33),xlabel='False-negative rate among historical positives (%)')
    ax.invert_yaxis(); ax.set_title('Overall recall hides difficult ring-rich structures',pad=17)
    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True); save(fig,'structural_errors.png')
    shutil.copy2(ROOT/'report/representation_comparison.png',out/'representation_comparison.png')
    (out/'figure_sources.json').write_text(json.dumps({'validation_figures':'audit_demo(): first three saved validation rows and frozen count models','threshold_tradeoff':'results/official_holdout_v1/metrics.csv','structural_errors':'results/official_holdout_v1/structure_subgroups.csv','representation_comparison':'report/representation_comparison.png','new_official_test_inference':False},indent=2),encoding='utf-8')
