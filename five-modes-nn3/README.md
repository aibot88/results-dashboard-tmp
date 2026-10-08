# 五种模式 Embedding + NN3 结果报告

直接用浏览器打开 index.html；图表库及数据已内嵌，可离线使用。解压整个压缩包后，页面中的CSV与核验文件链接可用。

采用用户选定的首轮 outputs_tuning_20261008 / tuning_v1 结果。未重新训练、选模或修改原结果。历史测试为2023-04至2024-12，验证为2022-02至2023-03；均按收益实现月份展示。原始负值保留，全部结果属于探索性历史实验。

默认展示VW，可切换EW；单模型明细支持五模式、Test/Validation、毛收益/扣10bp换手成本后收益。包含P10−P1、未中心化样本外R²、P9714 FF3月度Alpha与HAC t（滞后3）、配对增量、月度收益与复利曲线、模型参数及历史强基线参考。

metrics.csv为20行区间汇总；monthly_returns.csv为350行月度收益；paired_differences.csv为12行配对差值；five_mode_summary.csv与原始文件完全相同。CSV中的收益、Alpha为小数（0.01代表1%），R²及t值非百分数。

report_audit.json记录冻结核验及复算误差，source_fingerprints.json记录原始输入SHA256，verification.json记录报告核验。

复现：在本目录运行 G:/aibot/AssertBert/.venv/Scripts/python.exe -B build_report.py。需要原项目、输入与冻结产物仍在原位置；生成器会先核验冻结文件。便携浏览无需Python或原始输入。

页面格式参考 https://aibot88.github.io/results-dashboard-tmp/rs-b-all-models/ 。Apache ECharts许可证见ECHARTS_LICENSE.txt。
