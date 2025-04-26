def load_era_data():
    import os
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建out.txt的绝对路径
    out_file_path = os.path.join(script_dir, 'out.txt')
    
    era_dict = {}
    with open(out_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            era, years = line.split(':')
            # 处理年份范围
            if '—' in years:
                try:
                    start, end = years.split('—')
                    # 处理前年和后年
                    start = start.replace('前', '-')
                    end = end.replace('前', '-')
                    
                    # 提取年份，过滤掉非数字和负号
                    start_digits = ''.join(filter(lambda x: x.isdigit() or x == '-', start))
                    end_digits = ''.join(filter(lambda x: x.isdigit() or x == '-', end))
                    
                    # 确保有数字再转换
                    if start_digits and end_digits:
                        start_year = int(start_digits)
                        end_year = int(end_digits)
                        era_dict[era] = (start_year, end_year)
                except:
                    continue
    return era_dict

def convert_cn_num(cn_str):
    """转换中文数字为阿拉伯数字"""
    cn_num = {'一':1, '二':2, '三':3, '四':4, '五':5, 
              '六':6, '七':7, '八':8, '九':9, '十':10,
              '正':1, '元':1,'腊':12}
    
    if not cn_str:
        return 0
        
    if '十' in cn_str:
        if len(cn_str) == 1:
            return 10
        elif cn_str.startswith('十'):
            return 10 + cn_num.get(cn_str[-1], 0)
        elif cn_str.endswith('十'):
            return cn_num[cn_str[0]] * 10
        else:
            return cn_num[cn_str[0]] * 10 + cn_num.get(cn_str[-1], 0)
    else:
        return cn_num.get(cn_str, 0)

def convert_to_gregorian(era_year_str):
    """
    将年号纪年转换为公历年月日
    返回格式：年:月:日
    例如：'清乾隆十一年闰十二月五日' -> '1746:12:5'
    """
    # 加载年号数据
    era_dict = load_era_data()
    
    # 处理"清"字前缀
    if era_year_str.startswith('清'):
        era_year_str = era_year_str[1:]  # 去掉"清"字
    
    # 分解输入字符串
    for era in era_dict:
        if era_year_str.startswith(era):
            # 提取年数
            year_str = era_year_str[len(era):]
            if '年' not in year_str:
                return "格式错误，请包含'年'字"
            
            parts = year_str.split('年')
            year_num = parts[0]
            
            # 转换年份中文数字
            year_num = convert_cn_num(year_num)
            
            # 计算公历年份
            start_year = era_dict[era][0]
            gregorian_year = start_year + year_num - 1
            
            # 处理月份和日期
            month = 0
            day = 0
            
            if len(parts) > 1:
                month_day = parts[1]
                # 处理月份
                if '月' in month_day:
                    month_parts = month_day.split('月')
                    month_str = month_parts[0]
                    
                    # 处理闰月
                    is_leap = False
                    if month_str.startswith('闰'):
                        is_leap = True
                        month_str = month_str[1:]
                    
                    month = convert_cn_num(month_str)
                    if is_leap:
                        month = month + 0.5  # 闰月用小数表示
                    
                    # 处理日期
                    if len(month_parts) > 1 and '日' in month_parts[1]:
                        day_str = month_parts[1].replace('日', '')
                        day = convert_cn_num(day_str)
            
            return f"{gregorian_year}:{int(month)}:{day}"
            
    return "未找到对应年号"

# 添加测试代码
if __name__ == '__main__':
    test_cases = [
        "清乾隆十一年十二月",  # 测试带"清"字的情况
        "乾隆十一年闰十二月",  # 测试不带"清"字的情况
        "清康熙六十一年七月",  # 测试其他年号
        "清嘉庆二十五年正月",
        "清道光十一年",        # 只有年份
        "清乾隆十一年闰正月",  # 测试闰正月
        "清乾隆十一年元月五日", # 测试元月和具体日期
        "清乾隆十一年正月十五日", # 测试正月和两位数日期
        "清乾隆十一年闰二月二十日" # 测试闰月和具体日期
    ]

    print("测试年号转换:")
    for test in test_cases:
        result = convert_to_gregorian(test)
        print(f"{test} -> {result}")