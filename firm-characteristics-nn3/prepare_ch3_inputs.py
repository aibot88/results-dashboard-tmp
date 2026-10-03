import csv
import json
import math
import statistics as st
from pathlib import Path
from collections import defaultdict

ROOT = Path('/workspace/AssertBert')
OUT = ROOT / 'sparkEval/ch3_proxy_20261003'
RAW = ROOT / 'data/raw/ashare/market_financials'
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None

def write(name, rows, fields):
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def quantile(a, p):
    a = sorted(a)
    v = (len(a)-1)*p
    j = int(v)
    return a[j] + (a[min(j+1, len(a)-1)]-a[j])*(v-j)

companies = {r['Stkcd']: r for r in read(RAW/'基本面数据_update/公司文件1990_2026/TRD_Co.csv')}
pe = {}
for r in read(RAW/'基本面数据_update/相对价值指标2015_2025/FI_T10.csv'):
    if r['Accper'].endswith('12-31') and r['Source'] == '0':
        z = num(r['F100101B'])
        if z is not None and z > 0:
            pe[(r['Stkcd'], int(r['Accper'][:4]))] = z

market = defaultdict(dict)
for r in read(RAW/'月个股回报率文件/TRD_Mnth.csv'):
    code, month = r['Stkcd'], r['Trdmnt']
    co = companies.get(code, {})
    if co.get('Markettype') not in {'1','4','16','32'}:
        continue
    cap, ret = num(r['Msmvosd']), num(r['Mretnd'])
    if cap is not None and cap > 0:
        market[month][code] = {'cap':cap*1000, 'ret':ret, 'listed':co['Listdt']}

months = sorted(market)
reference = {r['month']: {k:float(r[k]) for k in ['rf_mon','mktrf','SMB','VMG']} for r in read(OUT/'ch3_reference.csv')}
# Relative adjusted price: each year's Dec price is the anchor for that annual PE.
# Observed Jan returns provide the price ratio to the preceding Dec, including 2015 Dec.
relative = defaultdict(lambda: 1.0)
chars = {}
char_rows = []
for month in months:
    year, mon = map(int, month.split('-'))
    report_year = year-1 if mon >= 4 else year-2
    for code, r in market[month].items():
        if r['ret'] is not None and r['ret'] > -1:
            for anchor in range(2015, year+1):
                if year > anchor:
                    relative[(code,anchor)] *= 1+r['ret']
        p = pe.get((code,report_year))
        # Only emit a signal when all monthly returns from its Dec anchor are available.
        expected = (year-report_year-1)*12+mon
        chain = []
        for yy in range(report_year+1,year+1):
            for mm in range(1,(mon if yy==year else 12)+1):
                rr = market.get(f'{yy:04d}-{mm:02d}',{}).get(code,{}).get('ret')
                if rr is not None and rr > -1:
                    chain.append(rr)
        valid = p is not None and report_year >= 2015 and len(chain)==expected
        ep = 1/p/relative[(code,report_year)] if valid else None
        row = {'asset_id':code,'formation_month':month,'market_cap_float':r['cap'],
               'log_market_cap_float':math.log(r['cap']),'annual_report_year':report_year,
               'annual_pe':p,'ep_price_adjusted_proxy':ep,
               'ep_static_proxy':1/p if valid else None}
        chars[(month,code)] = row
        char_rows.append(row)
write('monthly_characteristics.csv',char_rows,list(char_rows[0]))

factor_rows=[]
audit=[]
for prev, month in zip(months,months[1:]):
    y,m=map(int,month.split('-'))
    prior_y,prior_m=map(int,prev.split('-'))
    if y*12+m != prior_y*12+prior_m+1 or month not in reference:
        continue
    base=[]
    for code,r in market[prev].items():
        ret=market[month].get(code,{}).get('ret')
        if ret is None or ret <= -1:
            continue
        ly,lm=map(int,r['listed'][:7].split('-'))
        if prior_y*12+prior_m-(ly*12+lm)<6:
            continue
        base.append((code,r['cap'],ret))
    if not base:
        continue
    cutoff=quantile([v[1] for v in base],.3)
    eligible=[v for v in base if v[1]>cutoff]
    size_cut=quantile([v[1] for v in eligible],.5)
    rf=reference[month]['rf_mon']
    marketret=sum(c*r for _,c,r in eligible)/sum(c for _,c,_ in eligible)
    fullret=sum(c*r for _,c,r in base)/sum(c for _,c,_ in base)
    for mode,field in [('repriced','ep_price_adjusted_proxy'),('static','ep_static_proxy')]:
        priced=[(*v,chars[(prev,v[0])][field]) for v in eligible if chars[(prev,v[0])][field] is not None]
        if len(priced)<100:
            continue
        low,high=quantile([v[3] for v in priced],.3),quantile([v[3] for v in priced],.7)
        groups=defaultdict(list)
        for code,cap,ret,ep in priced:
            group=('S' if cap<=size_cut else 'B')+('G' if ep<=low else 'V' if ep>high else 'N')
            groups[group].append((cap,ret))
        if len(groups)!=6:
            continue
        returns={g:sum(c*r for c,r in vs)/sum(c for c,_ in vs) for g,vs in groups.items()}
        smb=sum(returns['S'+g]-returns['B'+g] for g in 'GNV')/3
        vmg=(returns['SV']+returns['BV']-returns['SG']-returns['BG'])/2
        row={'month':month,'method':mode,'rf_mon':rf,'mktrf':marketret-rf,'SMB':smb,'VMG':vmg,
             'mktrf_full_universe':fullret-rf,'eligible_stocks':len(eligible),'ep_available_stocks':len(priced)}
        row.update({k+'_reference':reference[month][k] for k in ['mktrf','SMB','VMG']})
        row.update({k+'_difference':row[k]-reference[month][k] for k in ['mktrf','SMB','VMG']})
        factor_rows.append(row)
        audit.append({'month':month,'method':mode,'size_cut':size_cut,'ep_30':low,'ep_70':high,
                      **{g+'_n':len(groups[g]) for g in ['SG','SN','SV','BG','BN','BV']}})
write('factors_comparison.csv',factor_rows,list(factor_rows[0]))
write('portfolio_audit.csv',audit,list(audit[0]))

def metrics(a,b):
    ma,mb=st.mean(a),st.mean(b)
    da,db=[x-ma for x in a],[x-mb for x in b]
    corr=sum(x*y for x,y in zip(da,db))/math.sqrt(sum(x*x for x in da)*sum(y*y for y in db))
    diff=[x-y for x,y in zip(a,b)]
    return {'months':len(a),'correlation':corr,'mean_difference_pp':st.mean(diff)*100,
            'mae_pp':st.mean(abs(x) for x in diff)*100,'rmse_pp':math.sqrt(st.mean(x*x for x in diff))*100,
            'proxy_sd_pp':st.stdev(a)*100,'reference_sd_pp':st.stdev(b)*100}
summary={}
for mode in ['repriced','static']:
    rows=[r for r in factor_rows if r['method']==mode]
    summary[mode]={'first_month':rows[0]['month'],'last_month':rows[-1]['month'],
                   'metrics':{k:metrics([r[k] for r in rows],[r[k+'_reference'] for r in rows]) for k in ['mktrf','SMB','VMG']},
                   'full_market_sensitivity':metrics([r['mktrf_full_universe'] for r in rows],[r['mktrf_reference'] for r in rows]),
                   'mean_ep_coverage':st.mean(r['ep_available_stocks']/r['eligible_stocks'] for r in rows)}
with open(OUT/'summary.json','w') as f:
    json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
