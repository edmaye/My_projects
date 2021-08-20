import xlrd,collections


table = {'PB':{}, 'PD':{}, 'PP':{}}
for key in table:
    table[key]['trust_count'] = collections.defaultdict(int)    # 观测总数
    table[key]['trust_true'] = collections.defaultdict(int)     # yes 的数量
    table[key]['trust_value'] = collections.defaultdict(int)    # yes占总数的比例

    table[key]['read_count'] = collections.defaultdict(int)
    table[key]['read_true'] = collections.defaultdict(int)
    table[key]['read_value'] = collections.defaultdict(int)



# 处理第一份数据
wb = xlrd.open_workbook('Prolific_Data_Cleaned.xlsx')
table_trust = wb.sheet_by_name('Trust')
table_read = wb.sheet_by_name('Readability')
for i in range(1,table_trust.nrows):
    line = table_trust.row_values(i)
    for j in range(12):
        img_id = line[j*2]
        table_id = img_id.split('-')[0]
        table[table_id]['trust_count'][img_id] += 1
        if line[j*2+1]=='yes':
            table[table_id]['trust_true'][img_id] += 1
for i in range(1,table_read.nrows):
    line = table_read.row_values(i)
    for j in range(12):
        img_id = line[j*2]
        table_id = img_id.split('-')[0]
        table[table_id]['read_count'][img_id] += 1
        if line[j*2+1]=='yes':
            table[table_id]['read_true'][img_id] += 1



# 处理第二份数据
wb = xlrd.open_workbook('2-Readability_One.xlsx')
table_read = wb.sheet_by_name('Sheet1')
for i in range(1,table_read.nrows):
    line = table_read.row_values(i)
    for j in range(12):
        img_id = line[j].split('/')[-1]
        table_id = img_id.split('-')[0]
        table[table_id]['read_count'][img_id] += 1
        if line[j+12]=='yes':
            table[table_id]['read_true'][img_id] += 1
        



# 处理第三份数据
wb = xlrd.open_workbook('2-Trust_One.xlsx')
table_trust = wb.sheet_by_name('Sheet1')
for i in range(1,table_trust.nrows):
    line = table_trust.row_values(i)
    for j in range(12):
        img_id = line[j].split('/')[-1]
        table_id = img_id.split('-')[0]
        table[table_id]['trust_count'][img_id] += 1
        if line[j+12]=='yes':
            table[table_id]['trust_true'][img_id] += 1









# 将计数结果写入txt中
for key in table:
    with open('data_'+key+'.txt', mode='w') as f:
        for img_id in table[key]['trust_count']:
            img_path = 'data/' + img_id
            print(img_id, ' 被观测次数：',table[key]['trust_count'][img_id])
            trust_value =str(float(table[key]['trust_true'][img_id])/table[key]['trust_count'][img_id])
            read_value =str(float(table[key]['read_true'][img_id])/table[key]['read_count'][img_id])
            txt = img_path+','+trust_value+','+read_value+'\n'
            f.write(txt) 
