import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>公司特征 Embedding32 + NN3 | 结果报告</title><script src="echarts.min.js"></script>
<style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#142033;background:#f3f6fb}*{box-sizing:border-box}body{margin:0;padding:28px}main{max-width:1450px;margin:auto}header{display:flex;justify-content:space-between;gap:20px;align-items:center}h1{font-size:28px;margin:0}h2{font-size:19px;margin:26px 0 12px}p{line-height:1.7;margin:8px 0}.muted{color:#64748b;font-size:13px}.badge{background:#ffedd5;color:#7c2d12;padding:10px;border-radius:8px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}.kpi,.panel{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:16px}.value{font-size:25px;font-weight:750;margin:5px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.wide{grid-column:1/-1}.chart{height:360px}select{padding:8px;border:1px solid #cbd5e1;border-radius:6px;margin:5px 12px 5px 5px;background:white;color:#142033}label{font-size:13px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}th,td{text-align:right;padding:10px;border-bottom:1px solid #e5eaf1}th:first-child,td:first-child{text-align:left}th{background:#f8fafc;color:#475569}.table-wrap{overflow:auto}a{color:#2563eb}.method{font-size:14px;line-height:1.8}.positive{color:#15803d}.negative{color:#b45309}.architecture{padding:18px;background:#eff6ff;border-radius:8px;color:#1e40af;font-family:monospace;font-size:15px;line-height:1.8}.footer{margin:20px 0;font-size:12px;color:#64748b}button{background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:8px 12px;color:#1d4ed8;cursor:pointer}@media(max-width:850px){body{padding:14px}header{display:block}.badge{display:inline-block;margin-top:10px}.kpis{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}@media(max-width:480px){.kpis{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><main>
<header><div><h1>公司特征 Embedding32 + NN3</h1><p class="muted">月度公司特征 · 32 维收益监督嵌入 · 下一月预测 · EW/VW P10−P1 · CH3 Alpha</p></div><div class="badge">探索性样本外实验 · 不按测试收益选模型</div></header>
<section class="kpis" id="kpis"></section>
<h2>模型结构</h2><section class="panel"><div class="architecture">16 个公司特征 + 16 个缺失标记 → Encoder(64 → 32) → Embedding32 → NN3(64 → 32 → 16) → 下月收益<br>基线：同样的特征和缺失标记 → NN3(64 → 32 → 16) → 下月收益</div><p class="muted">编码器与 NN3 联合训练，目标为 MSE。32 维为非线性表示，不是信息维度增加的保证。固定种子 42 / 43 / 44，Ensemble 为预测值平均；各编码器分别导出 32 维向量。</p></section>
<h2>样本外结果</h2><section class="panel"><label>评估期间<select id="period"><option value="primary">2023-01–2024-11（同参考页日历）</option><option value="extended">2023-01–2025-04（扩展）</option></select></label><label>组合权重<select id="weight"><option>EW</option><option>VW</option></select></label><label>结果范围<select id="scope"><option value="ensemble">固定三种子 Ensemble</option><option value="all">全部种子与 Ensemble</option></select></label></section>
<section class="grid" style="margin-top:15px"><article class="panel"><div class="chart" id="spreadChart"></div></article><article class="panel"><div class="chart" id="alphaChart"></div></article></section>
<h2>模型对比表</h2><section class="panel table-wrap"><table id="comparison"></table><p class="muted">收益和 Alpha 为月度百分比；HAC 使用 3 个滞后。R² 为相对零预测的样本外未中心化 R²。净收益仅扣除 10bp × 目标权重交易量，不含借券费、滑点及权重漂移调整。</p></section>
<h2>单模型月度明细</h2><section class="panel"><label>模型<select id="model"></select></label><button id="download">下载当前模型月度收益 CSV</button></section>
<section class="grid" style="margin-top:15px"><article class="panel wide"><div class="chart" id="monthlyChart"></div></article><article class="panel wide"><div class="chart" id="cumulativeChart"></div></article></section>
<h2>实验口径与限制</h2><section class="panel method">
<p><strong>时间：</strong>按收益发生月份划分，训练 2016-06–2021-12，验证 2022-01–2022-12，主要测试 2023-01–2024-11，扩展至 2025-04。特征为前一月末值，标签为下一月收益。验证 MSE 用于早停，最多 80 轮；不按测试 P10−P1 调参数。</p>
<p><strong>特征：</strong>规模、E/P 代理、账面流通市值比、ROA、ROE、ROIC、研发费用率、流动比率、杠杆、资产增长、1 月收益、3/6 月动量、12−1 动量、12 月波动率及 Beta。特征逐月横截面去除 1%/99% 极值、填补中位数并标准化，加缺失标记。预处理不读取收益标签。</p>
<p><strong>股票池：</strong>沪深 A 股，含创业板、科创板，上市月份至少 6 个月、形成月流通市值为正。预测实验未剔除市值最小 30%，区别于 CH3 因子构建。无收益记录的股票无法纳入回测，历史公司表覆盖与停牌/退市处理尚未完整校验。</p>
<p><strong>财务时间：</strong>年度财务数据假设于次年四月底可获得、用于五月收益预测；没有实际公告日期。E/P 由正值年度 PE 按非分红收益率调整，缺少负 E/P；账面市值比使用流通市值。这些是代理口径，个别延期公告仍存在前视风险。</p>
<p><strong>组合：</strong>每月按预测收益排序，最低/最高 floor(N/10) 只分别为 P1/P10；EW 等权，VW 使用形成月流通市值。收益为不含现金红利再投资的收益。CH3 Alpha 使用本地参考文件真实因子，不是此前构建的近似因子；多空组合不再减一次无风险利率。</p>
<p><strong>解读：</strong>不同随机种子的结果全部保留。高收益不等于统计显著，也不等于能够执行做空。测试月份曾用于项目其他实验，因此报告为探索性结果。<a href="../w2v/">参考 W2V 页面</a>仅提供展示形式与测试日历参考，股票池、输入与模型不同，不作严格横向胜负判断。</p>
<p><a href="metrics.csv">下载指标</a> · <a href="portfolio_monthly.csv">下载全部月度组合收益</a> · <a href="results.json">下载结果与训练历史</a> · <a href="run.py">计算脚本</a> · <a href="README.md">复现说明</a></p>
</section><p class="footer" id="footer"></p></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2)+'%',num=x=>Number(x).toFixed(3);
const charts=Object.fromEntries(['spreadChart','alphaChart','monthlyChart','cumulativeChart'].map(id=>[id,echarts.init($(id))]));
const names=[...new Set(DATA.models.map(x=>x.model))];
$('model').innerHTML=names.map(x=>`<option ${x==='EMB32_ensemble'?'selected':''}>${x}</option>`).join('');
function rows(){return DATA.models.filter(x=>x.period===$('period').value&&x.weighting===$('weight').value&&($('scope').value==='all'||x.model.endsWith('ensemble')))}
function lineOptions(title,months,series){return {title:{text:title,textStyle:{fontSize:16}},tooltip:{trigger:'axis',valueFormatter:x=>pct(x)},legend:{top:30},grid:{left:65,right:25,top:75,bottom:65},xAxis:{type:'category',data:months},yAxis:{type:'value',axisLabel:{formatter:x=>(x*100).toFixed(0)+'%'}},dataZoom:[{type:'inside'},{type:'slider',bottom:5}],series:series.map(([name,data])=>({name,data,type:'line',showSymbol:false}))}}
function drawBars(id,title,items,key){charts[id].setOption({title:{text:title,textStyle:{fontSize:16}},tooltip:{trigger:'axis',valueFormatter:pct},grid:{left:65,right:25,top:65,bottom:75},xAxis:{type:'category',data:items.map(x=>x.model),axisLabel:{rotate:20}},yAxis:{type:'value',axisLabel:{formatter:x=>(100*x).toFixed(1)+'%'}},series:[{type:'bar',data:items.map(x=>({value:x[key],itemStyle:{color:x.model.startsWith('EMB')?'#2563eb':'#94a3b8'}}))}]},true)}
function currentMonthly(){const end=$('period').value==='primary'?'2024-11':'2025-04';return DATA.monthly.filter(x=>x.model===$('model').value&&x.weighting===$('weight').value&&x.month<=end)}
function renderDetails(){const r=currentMonthly(),months=r.map(x=>x.month);charts.monthlyChart.setOption(lineOptions($('model').value+' · 月度收益',months,[['P10',r.map(x=>x.p10)],['P1',r.map(x=>x.p1)],['P10−P1',r.map(x=>x.spread)],['净多空（10bp）',r.map(x=>x.net_10bps)]]),true);function cum(key){let v=1;return r.map(x=>{v*=1+x[key];return v-1})}charts.cumulativeChart.setOption(lineOptions('累计收益 · 每月再平衡的示意复利（非真实保证金账户）',months,[['P10',cum('p10')],['P1',cum('p1')],['多空毛收益',cum('spread')],['多空净收益（10bp）',cum('net_10bps')]]),true)}
function render(){const r=rows(),a=r.find(x=>x.model==='EMB32_ensemble')||r[0];$('kpis').innerHTML=[['Embedding32 月均多空',pct(a.mean_spread),`${a.months} 个月 · ${a.weighting}`],['Embedding32 月度 Rank IC',num(a.rank_ic),'每月 Spearman 平均'],['Embedding32 CH3 Alpha',pct(a.alpha),'HAC t = '+num(a.alpha_hac_t)],['Embedding32 净月收益',pct(a.mean_net_10bps),'10bp 交易成本情景']].map(([label,value,meta])=>`<article class="kpi"><div class="muted">${label}</div><div class="value">${value}</div><div class="muted">${meta}</div></article>`).join('');drawBars('spreadChart','月均 P10−P1（毛收益）',r,'mean_spread');drawBars('alphaChart','CH3 月度 Alpha',r,'alpha');$('comparison').innerHTML='<thead><tr><th>模型</th><th>OOS R²</th><th>Rank IC</th><th>月均 P10−P1</th><th>收益 HAC t</th><th>年化 Sharpe</th><th>CH3 Alpha</th><th>Alpha HAC t</th><th>净收益 10bp</th><th>月数</th></tr></thead><tbody>'+r.map(x=>`<tr><td>${x.model}</td><td>${pct(x.r2)}</td><td>${num(x.rank_ic)}</td><td>${pct(x.mean_spread)}</td><td>${num(x.spread_hac_t)}</td><td>${num(x.annual_sharpe)}</td><td>${pct(x.alpha)}</td><td>${num(x.alpha_hac_t)}</td><td>${pct(x.mean_net_10bps)}</td><td>${x.months}</td></tr>`).join('')+'</tbody>';renderDetails()}
['period','weight','scope'].forEach(id=>$(id).addEventListener('change',render));$('model').addEventListener('change',renderDetails);$('download').addEventListener('click',()=>{const keys=['month','weighting','p1','p10','spread','net_10bps','target_weight_turnover'];const csv=[keys.join(','),...currentMonthly().map(x=>keys.map(k=>x[k]).join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download=$('model').value+'_'+$('weight').value+'.csv';a.click();URL.revokeObjectURL(a.href)});$('footer').textContent='生成时间：'+DATA.generated_at+' · 训练样本 '+DATA.train_rows.toLocaleString()+' · 验证样本 '+DATA.validation_rows.toLocaleString()+' · '+DATA.runtime.device+' · '+DATA.runtime.torch;window.addEventListener('resize',()=>Object.values(charts).forEach(x=>x.resize()));render();
</script></body></html>'''


def build(output):
    data = json.loads((output/'results.json').read_text())
    payload = json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (output/'index.html').write_text(TEMPLATE.replace('__DATA__',payload))


if __name__=='__main__':
    build(Path(__file__).parent/'outputs')
