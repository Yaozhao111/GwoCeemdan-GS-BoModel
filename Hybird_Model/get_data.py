import tushare as ts
def get_data(ts_code,start_date,end_date,fields,name)  :
    ts.set_token('667d224f69a317ce8609dac2644e05d5fb704224b7288ef2cca5eef7')
    pro=ts.pro_api()
    data=pro.index_daily(ts_code=ts_code,
                         start_date=start_date,
                         end_date=end_date,
                         fields=fields)
    data=data.reindex(index=data.index[::-1])
    data.to_csv(name+'.csv')
    return data

if __name__=='__main__':
    data=get_data('399300.SZ','20170101','20180101',
                  fields=['close'],name='CSI300')
    print(data)
