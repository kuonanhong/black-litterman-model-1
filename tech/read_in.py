import pandas as pd
import matplotlib.pyplot as plt
fund_num=20
excel=pd.ExcelFile('xlsx/fund_data.xlsx')
fund_names=excel.sheet_names
def read_from_sheet(columns_needed,worksheet=0):
    result=pd.read_excel(excel,sheetname=worksheet,skiprows=0,header=1,index_col=0,usecols=columns_needed,na_values="#N/A N/A")
    result.columns=[fund_names[worksheet]]
    return result
def read_all(columns_needed):
    first=read_from_sheet(columns_needed)
    for i in range(1,fund_num):
        current=read_from_sheet(columns_needed,i)
        first=pd.merge(first,current,left_index=True,right_index=True)
    return first
excel_path='xlsx/fund_data.xlsx'
excel=pd.ExcelFile(excel_path)
price_col=["Date","PX_LAST"]
return_col=["Date","DAY_TO_DAY_TOT_RETURN_GROSS_DVDS"]
return_frame=read_all(return_col)
print(return_frame.head(5))
return_frame.to_pickle('plk/return_matrix.plk')
price_frame=read_all(price_col)
print(price_frame.head(5))
price_frame.to_pickle('plk/price_matrix.plk')


