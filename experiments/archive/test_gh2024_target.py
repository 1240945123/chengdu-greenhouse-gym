import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from processing.chengdu_crop_observations import extract_crop_observations

SQL = r"E:/school/NKY/玻璃温室数据库信息/tomato_full_backup_20260724.sql"

# 方案 A：把 GH2024 设为目标棚
frame = extract_crop_observations(
    SQL,
    target_greenhouse_ids=[1],
    target_greenhouse_codes=["GH2024"],
    source_sites={
        "GH2024": "chengdu_xindu_experimental_base",
        "GH202602061448266452514": "chengdu_pidu_target_greenhouse",
    },
    timezone="Asia/Shanghai",
)

print(f"总行数: {len(frame)}")
print(f"target_eligible=True 行数: {int(frame['target_eligible'].sum())}")
print(f"greenhouse_code 分布: {frame['greenhouse_code'].value_counts().to_dict()}")
print(f"planting_code 分布: {frame['planting_code'].value_counts().to_dict()}")
print(f"cultivar: {frame['cultivar'].unique().tolist()}")
print(f"plant_density: {frame['plant_density_plants_m2'].unique().tolist()}")
print(f"日期范围: {frame['observation_date'].min()} ~ {frame['observation_date'].max()}")
print(f"去重日期数: {frame['observation_date'].nunique()}")
print(f"去重样本植株数: {frame['sample_plant_id'].nunique()}")

ripe_fw = frame["ripe_fruit_fresh_g_per_plant"].notna()
ripe_dw = frame["ripe_fruit_dry_g_per_plant"].notna()
print(f"\n成熟果鲜重非空: {int(ripe_fw.sum())}")
print(f"成熟果干重非空: {int(ripe_dw.sum())}")
print(f"鲜重+干重同时对: {int((ripe_fw & ripe_dw).sum())}")

paired = frame[ripe_fw & ripe_dw].copy()
if not paired.empty:
    dm = paired["ripe_fruit_dry_g_per_plant"] / paired["ripe_fruit_fresh_g_per_plant"]
    print(f"\n成熟果干物质比例(dry/fresh): mean={dm.mean():.4f}, median={dm.median():.4f}")
    # 单位面积成熟果产量估计（干重 × 密度 / 1000 = kg/m2）
    paired["fruit_dry_kg_m2"] = paired["ripe_fruit_dry_g_per_plant"] * paired["plant_density_plants_m2"] / 1000.0
    print(f"单株成熟果干重范围: {paired['ripe_fruit_dry_g_per_plant'].min():.1f} ~ {paired['ripe_fruit_dry_g_per_plant'].max():.1f} g")
    print(f"换算单位面积成熟果干重(kg/m2): mean={paired['fruit_dry_kg_m2'].mean():.3f}, max={paired['fruit_dry_kg_m2'].max():.3f}")

print(f"\n各日期成熟果干重观测数:")
date_counts = frame[ripe_dw].groupby(frame['observation_date'].dt.date)['ripe_fruit_dry_g_per_plant'].agg(['count','mean','max'])
for d, row in date_counts.iterrows():
    print(f"  {d}: n={int(row['count'])}, mean={row['mean']:.1f}g, max={row['max']:.1f}g")
