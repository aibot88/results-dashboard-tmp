"""Monthly observed-characteristic encoder and NN3, with chronological evaluation."""
import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import spearmanr
from torch import nn

FEATURES = ['log_me', 'ep', 'log_bm_float', 'roa', 'roe', 'roic', 'rd_sales',
            'current_ratio', 'leverage', 'asset_growth', 'return_1m',
            'momentum_3m', 'momentum_6m', 'momentum_12_1', 'volatility_12m', 'beta_12m']
SEEDS = [42, 43, 44]
TRAIN_END, VALID_END, PRIMARY_END = '2021-12', '2022-12', '2024-11'


def read_csv(path):
    return pd.read_csv(path, dtype={'Stkcd': str, 'asset_id': str}, encoding='utf-8-sig')


def annual_table(path, columns):
    frame = read_csv(path)
    frame = frame[frame.Accper.str.endswith('12-31')]
    if 'Source' in frame:
        frame = frame[frame.Source == 0]
    if 'Typrep' in frame:
        frame = frame[frame.Typrep == 'A']
    frame['report_year'] = frame.Accper.str[:4].astype(int)
    frame = frame.rename(columns={'Stkcd': 'asset_id', **columns})
    assert not frame.duplicated(['asset_id', 'report_year']).any()
    return frame[['asset_id', 'report_year', *columns.values()]]


def build_panel(root, output):
    raw = root / 'data/raw/ashare/market_financials'
    market = read_csv(raw / '月个股回报率文件/TRD_Mnth.csv')
    market = market.rename(columns={'Stkcd': 'asset_id', 'Trdmnt': 'month', 'Mretnd': 'actual'})
    market['cap'] = market.Msmvosd * 1000
    co = read_csv(raw / '基本面数据_update/公司文件1990_2026/TRD_Co.csv')
    co = co[co.Markettype.isin([1, 4, 16, 32])].rename(columns={'Stkcd': 'asset_id'})
    market = market.merge(co[['asset_id', 'Listdt']], on='asset_id', validate='many_to_one')
    assert not market.duplicated(['asset_id', 'month']).any()
    grid = market.pivot(index='month', columns='asset_id', values='actual')
    grid = grid.reindex(pd.period_range('2016-01', '2025-04', freq='M').astype(str))
    logs = np.log1p(grid.where(grid > -1))
    histories = {'return_1m': grid, 'momentum_3m': np.expm1(logs.rolling(3).sum()),
                 'momentum_6m': np.expm1(logs.rolling(6).sum()),
                 'momentum_12_1': np.expm1(logs.shift(1).rolling(11).sum()),
                 'volatility_12m': grid.rolling(12).std()}
    reference = read_csv(root / 'sparkEval/ch3_proxy_20261003/ch3_reference.csv').set_index('month')
    market_factor = reference.mktrf.reindex(grid.index)
    histories['beta_12m'] = grid.rolling(12).cov(market_factor).div(market_factor.rolling(12).var(), axis=0)
    panel = market[['asset_id', 'month', 'cap', 'Listdt']].copy()
    for name, values in histories.items():
        stacked = values.rename_axis(index='month', columns='asset_id').stack().rename(name).reset_index()
        panel = panel.merge(stacked, on=['month', 'asset_id'], how='left', validate='one_to_one')
    chars = read_csv(root / 'sparkEval/ch3_proxy_20261003/monthly_characteristics.csv')
    chars = chars.rename(columns={'formation_month': 'month', 'ep_price_adjusted_proxy': 'ep'})
    panel = panel.merge(chars[['asset_id', 'month', 'ep']], how='left', on=['asset_id', 'month'], validate='one_to_one')
    date = pd.PeriodIndex(panel.month, freq='M')
    panel['report_year'] = date.year - np.where(date.month >= 4, 1, 2)
    tables = [
        ('基本面数据_update/盈利能力2015_2025/FI_T5.csv', {'F050201B':'roa','F050501B':'roe','F051201B':'roic','F053401B':'rd_sales'}),
        ('基本面数据_update/偿债能力2015_2025/FI_T1.csv', {'F010101A':'current_ratio','F011201A':'leverage'}),
        ('发展能力update/FI_T8.csv', {'F080601A':'asset_growth'}),
        ('资产负债表/FS_Combas.csv', {'A003000000':'book_equity'}),
    ]
    for path, cols in tables:
        panel = panel.merge(annual_table(raw/path, cols), how='left', on=['asset_id','report_year'], validate='many_to_one')
    panel['log_me'] = np.log(panel.cap.where(panel.cap > 0))
    panel['log_bm_float'] = np.log((panel.book_equity / panel.cap).where(panel.book_equity > 0))
    panel['outcome_month'] = (date + 1).astype(str)
    labels = market[['asset_id','month','actual']].rename(columns={'month':'outcome_month'})
    panel = panel.merge(labels, on=['asset_id','outcome_month'], how='left', validate='one_to_one')
    age = date.astype('int64') - pd.PeriodIndex(panel.Listdt.str[:7], freq='M').astype('int64')
    panel = panel[(panel.month >= '2016-05') & (age >= 6) & (panel.cap > 0)].copy()
    for key in FEATURES:
        panel[key] = panel[key].replace([np.inf,-np.inf], np.nan)
    panel.to_csv(output/'monthly_panel.csv', index=False)
    return panel, reference


def normalize_monthly(panel):
    values, flags = [], []
    for _, group in panel.groupby('month', sort=True):
        frame = group[FEATURES]
        missing = frame.isna().astype(np.float32)
        clipped = frame.clip(frame.quantile(.01), frame.quantile(.99), axis=1)
        filled = clipped.fillna(clipped.median()).fillna(0)
        normalized = ((filled-filled.mean()) / filled.std(ddof=0).replace(0, 1)).fillna(0)
        values.append(normalized)
        flags.append(missing)
    # Each contemporaneous cross-section uses no labels or future-month statistics.
    return np.concatenate([pd.concat(values).reindex(panel.index).to_numpy(),
                           pd.concat(flags).reindex(panel.index).to_numpy()], axis=1).astype(np.float32)


class ReturnNetwork(nn.Module):
    def __init__(self, input_size, embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size,64),nn.ReLU(),nn.Linear(64,32)) if embedding else nn.Identity()
        width = 32 if embedding else input_size
        layers = []
        for hidden in [64,32,16]:
            layers.extend([nn.Linear(width,hidden),nn.ReLU(),nn.Dropout(.1)])
            width = hidden
        self.head = nn.Sequential(*layers,nn.Linear(width,1))

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def train_network(data, embedding, seed, output):
    torch.manual_seed(seed)
    x, target, train, validation = data
    network = ReturnNetwork(x.shape[1], embedding)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001, weight_decay=.0001)
    center, scale = target[train].mean(), target[train].std()
    y = (target-center)/scale
    best, state, wait, history = float('inf'), None, 0, []
    for epoch in range(1,81):
        network.train()
        shuffled = train[torch.randperm(len(train))]
        for ids in shuffled.split(4096):
            optimizer.zero_grad()
            loss = (network(x[ids])-y[ids]).square().mean()
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            mse = (network(x[validation])-y[validation]).square().mean().item()
        history.append({'epoch':epoch,'validation_mse':mse*scale.item()**2})
        if mse < best-1e-6:
            best, state, wait = mse, copy.deepcopy(network.state_dict()), 0
        else:
            wait += 1
        if epoch % 10 == 0:
            print(f'embedding={embedding} seed={seed} epoch={epoch} val_mse={mse*scale.item()**2:.6f}',flush=True)
        if wait >= 10:
            break
    network.load_state_dict(state)
    network.eval()
    name = ('EMB32' if embedding else 'RAW')+f'_seed{seed}'
    torch.save({'state_dict':state,'input_features':FEATURES,'missing_flags':True,
                'target_mean':center.item(),'target_std':scale.item(),'embedding':embedding},output/f'{name}.pt')
    with torch.no_grad():
        prediction = torch.cat([network(chunk) for chunk in x.split(16384)])*scale+center
    return network, prediction.numpy(), history


def portfolio_series(frame, score):
    rows, previous = [], {'EW':{},'VW':{}}
    for month, group in frame.groupby('outcome_month', sort=True):
        ranked = group.sort_values([score,'asset_id'], kind='stable')
        count = max(1,len(ranked)//10)
        short, long = ranked.iloc[:count], ranked.iloc[-count:]
        for weighting in ['EW','VW']:
            ws = np.ones(count)/count if weighting=='EW' else short.cap.to_numpy()/short.cap.sum()
            wl = np.ones(count)/count if weighting=='EW' else long.cap.to_numpy()/long.cap.sum()
            positions = dict(zip(short.asset_id,-ws))
            positions.update(dict(zip(long.asset_id,wl)))
            old = previous[weighting]
            turnover = sum(abs(positions.get(k,0)-old.get(k,0)) for k in positions.keys()|old.keys())
            previous[weighting] = positions
            p1,p10 = float(ws @ short.actual),float(wl @ long.actual)
            rows.append({'month':month,'weighting':weighting,'p1':p1,'p10':p10,'spread':p10-p1,
                         'target_weight_turnover':turnover,'net_10bps':p10-p1-.001*turnover,'stocks':len(group)})
    return pd.DataFrame(rows)


def hac_statistics(series, factors=None):
    values = np.asarray(series)
    design = np.ones((len(values),1)) if factors is None else sm.add_constant(np.asarray(factors), has_constant='add')
    fit = sm.OLS(values,design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    return float(fit.params[0]),float(fit.tvalues[0])


def evaluate(frame, predictions, reference, output):
    models, monthly = [], []
    for name, predicted in predictions.items():
        current = frame.copy()
        current['prediction'] = predicted
        for period,end in [('primary',PRIMARY_END),('extended','2025-04')]:
            test = current[(current.outcome_month>'2022-12') & (current.outcome_month<=end)]
            series = portfolio_series(test,'prediction')
            ic = [float(spearmanr(g.prediction,g.actual).statistic) for _,g in test.groupby('outcome_month')]
            r2 = 1-float(np.sum((test.actual-test.prediction)**2)/np.sum(test.actual**2))
            for weighting,returns in series.groupby('weighting'):
                aligned = reference.loc[returns.month,['mktrf','SMB','VMG']]
                alpha,alpha_t = hac_statistics(returns.spread,aligned)
                mean,tstat = hac_statistics(returns.spread)
                models.append({'model':name,'period':period,'weighting':weighting,'rank_ic':float(np.nanmean(ic)),
                               'r2':r2,'mean_spread':mean,'spread_hac_t':tstat,'alpha':alpha,'alpha_hac_t':alpha_t,
                               'annual_sharpe':float(returns.spread.mean()/returns.spread.std()*np.sqrt(12)),
                               'mean_net_10bps':float(returns.net_10bps.mean()),'months':len(returns),
                               'test_rows':len(test),'turnover':float(returns.target_weight_turnover.mean())})
            if period=='extended':
                series['model'] = name
                monthly.extend(series.to_dict('records'))
        current[current.outcome_month>'2022-12'].to_csv(output/f'predictions_{name}.csv',index=False)
    pd.DataFrame(models).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(output/'portfolio_monthly.csv',index=False)
    return models,monthly


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/workspace/AssertBert'))
    args=parser.parse_args()
    output=args.root/'sparkEval/firm_characteristic_nn3_20261003/outputs'
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    panel,reference=build_panel(args.root,output)
    panel=panel.sort_values(['month','asset_id']).reset_index(drop=True)
    normalized=normalize_monthly(panel)
    available=panel.actual.notna().to_numpy() & np.isfinite(panel.actual.to_numpy())
    x=torch.from_numpy(normalized[available])
    panel=panel.loc[available].reset_index(drop=True)
    target=torch.from_numpy(panel.actual.to_numpy(dtype=np.float32))
    train=torch.from_numpy(np.flatnonzero(panel.outcome_month<=TRAIN_END))
    validation=torch.from_numpy(np.flatnonzero((panel.outcome_month>TRAIN_END)&(panel.outcome_month<=VALID_END)))
    assert len(train) and len(validation) and torch.isfinite(x).all()
    assert panel.loc[train.numpy(),'outcome_month'].max()<panel.loc[validation.numpy(),'outcome_month'].min()
    print(f'panel={len(panel)} input={x.shape[1]} train={len(train)} validation={len(validation)}',flush=True)
    predictions,histories={},{}
    for embedding in [False,True]:
        prefix='EMB32' if embedding else 'RAW'
        ensemble=[]
        for seed in SEEDS:
            network,predicted,history=train_network((x,target,train,validation),embedding,seed,output)
            predictions[f'{prefix}_seed{seed}']=predicted
            ensemble.append(predicted)
            histories[f'{prefix}_seed{seed}']=history
            if embedding:
                ids=np.flatnonzero(panel.outcome_month>'2022-12')
                with torch.no_grad():
                    vectors=network.encoder(x[ids]).numpy()
                assert vectors.shape==(len(ids),32) and np.isfinite(vectors).all()
                np.savez_compressed(output/f'embedding32_seed{seed}.npz',embedding=vectors,
                                    asset_id=panel.loc[ids,'asset_id'].to_numpy(dtype=str),
                                    formation_month=panel.loc[ids,'month'].to_numpy(dtype=str))
        predictions[prefix+'_ensemble']=np.mean(ensemble,axis=0)
    models,monthly=evaluate(panel,predictions,reference,output)
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'features':FEATURES,'seeds':SEEDS,
            'architecture':'encoder 32 input (16 features + 16 missing flags) → 64 → 32; NN3 64 → 32 → 16 → 1',
            'periods':{'train':'outcome 2016-06–2021-12','validation':'outcome 2022-01–2022-12',
                       'test':'outcome 2023-01–2024-11','extended':'outcome 2023-01–2025-04'},
            'panel_rows':len(panel),'train_rows':len(train),'validation_rows':len(validation),
            'models':models,'monthly':monthly,'histories':histories,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime':{'torch':torch.__version__,'device':'cpu','threads':4},
            'warnings':['Exploratory: test period may have been viewed in prior project experiments.',
                        'Annual reports assumed available by Apr 30; actual publication dates absent.',
                        'Returns exclude cash dividends; missing outcome returns excluded from evaluation.',
                        'Book equity / float capitalization and annual PE repricing are proxies.',
                        'Cost scenario uses target-weight changes, not drift-adjusted turnover; no borrow fees.',
                        'Alpha uses local CH3 reference, not independently reconstructed proxy factors.']}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps([m for m in models if m['model'].endswith('ensemble') and m['period']=='primary'],indent=2),flush=True)


if __name__=='__main__':
    main()
