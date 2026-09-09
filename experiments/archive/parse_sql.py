import re
sql = open(r'E:/school/NKY/玻璃温室数据库信息/tomato_full_backup_20260724.sql', encoding='utf-8', errors='replace').read()

def rows_of(tbl):
    out = []
    for m in re.finditer(r'INSERT INTO `' + tbl + r'` VALUES \((.*?)\);', sql, re.S):
        r = m.group(1)
        vals, cur, i, in_str = [], '', 0, False
        while i < len(r):
            c = r[i]
            if c == "'":
                if in_str and i+1 < len(r) and r[i+1] == "'":
                    cur += "'"; i += 2; continue
                in_str = not in_str; cur += c
            elif c == ',' and not in_str:
                vals.append(cur.strip()); cur = ''
            else:
                cur += c
            i += 1
        vals.append(cur.strip())
        out.append(vals)
    return out

print('=== greenhouse_info 全部记录 ===')
for v in rows_of('greenhouse_info'):
    print(f'  id={v[0]}, code={v[6]}, site={v[7]}, area={v[8]}m2, type={v[9]}, mode={v[10]}')

print('\n=== planting_area 全部记录 ===')
for v in rows_of('planting_area'):
    print('  ' + str(v)[:300])

print('\n=== planting_info 全部记录（完整）===')
for v in rows_of('planting_info'):
    print(f'  code={v[1]}, 作物={v[2]}, 品种={v[8]}, 株数={v[9]}, 育苗={v[11]}, 定植={v[12]}, 始花={v[13]}, 初果={v[14]}, 成熟={v[15]}, 拉秧={v[16]}, 密度={v[17]}株/m2, 温室={v[19]}, 面积单元={v[29]}')

print('\n=== accumulate_temperature 表结构（积温）===')
m = re.search(r'CREATE TABLE `accumulate_temperature`.*?ENGINE[^\n]*', sql, re.S)
if m:
    for line in m.group(0).split('\n'):
        if 'COMMENT' in line and 'PRIMARY' not in line:
            print('  ' + line.strip()[:120])
print('  记录数:', len(rows_of('accumulate_temperature')))
