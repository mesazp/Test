#coding:utf-8
import pandas as pd
import numpy as np
import pymysql
import time
import math
import multiprocessing
from multiprocessing import Pool
import urllib.request
from statsmodels.tsa.arima_model import ARMA
import warnings

warnings.filterwarnings('ignore')

def get_data_from_url(url_str):
    result = urllib.request.urlopen(url_str)
    content = result.read().decode()
    return content
 
def proper_model(timeseries, maxLag):
    init_bic = 1000000000
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(timeseries, order=(p, q))
            try:
                results_ARMA = model.fit(disp = 0, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                model_return = results_ARMA
                init_bic = bic
    return model_return
def give_me_value(result_only_lost, result_list, time_list_seq, temp_list_seq, humi_list_seq, missed_value_num, det_time):
    if len(time_list_seq) < 100:
        time_list_seq = time_list_seq[: len(time_list_seq)]
        temp_list_seq = temp_list_seq[: len(temp_list_seq)]
        humi_list_seq = humi_list_seq[: len(humi_list_seq)]
    else:
        time_list_seq = time_list_seq[len(time_list_seq) - 100: len(time_list_seq)]
        temp_list_seq = temp_list_seq[len(temp_list_seq) - 100: len(temp_list_seq)]
        humi_list_seq = humi_list_seq[len(humi_list_seq) - 100: len(humi_list_seq)]  

    tidx = pd.DatetimeIndex(time_list_seq, freq = None)
    dta_temp = pd.Series(temp_list_seq, index = tidx)
    dta_humi = pd.Series(humi_list_seq, index = tidx)

    model_temp = proper_model(dta_temp, 9)
    model_humi = proper_model(dta_humi, 9)
    predict_temp = model_temp.forecast(missed_value_num)
    predict_humi = model_humi.forecast(missed_value_num)
    for num_i in range(missed_value_num):
        time_list_seq.append(time_list_seq[-1] + det_time)
        temp_list_seq.append(float(predict_temp[0][num_i]))
        humi_list_seq.append(float(predict_humi[0][num_i]))
        
        result_list.append({"id": result_list[-1]["id"] + 1, "time": time_list_seq[-1], "temp": temp_list_seq[-1], "humi": humi_list_seq[-1]})
        result_only_lost.append({"time": time_list_seq[-1], "temp": temp_list_seq[-1], "humi": humi_list_seq[-1]})
    return
        

def predict_temp_humi(data_str, det_time):
    data_list = data_str.split("\n")
    del data_list[-1]
    if len(data_list) < 2:
        return [], ["内容为空"]
    result_list = []
    result_only_lost = []
    time_list_seq = []
    temp_list_seq = []
    humi_list_seq = []
    missed_value_num = 0
    total_missed_value_num = 0
    for i in range(1, len(data_list)):
        tmp_dict = {}
        data_list_split = data_list[i].split(",")
        #构建字典
        tmp_dict["id"] = int(data_list_split[0])
        tmp_dict["time"] = int(data_list_split[1])
        tmp_dict["temp"] = float(data_list_split[2])
        tmp_dict["humi"] = float(data_list_split[3])
        if i == 1:
            result_list.append(tmp_dict)
            #时间 温度 湿度  list
            time_list_seq.append(int(data_list_split[1]))
            temp_list_seq.append(float(data_list_split[2]))
            humi_list_seq.append(float(data_list_split[3]))
        else:
            diff_tim = int(data_list_split[1]) - int(data_list[i - 1].split(",")[1])
            if int(diff_tim % det_time) < 20 and int(diff_tim / det_time) > 1:
                missed_value_num = int(diff_tim / det_time) - 1
            elif int(diff_tim % det_time) > (det_time - 20) and int(diff_tim / det_time) > 0:
                missed_value_num = int(diff_tim / det_time)
            if missed_value_num:
                give_me_value(result_only_lost, result_list, time_list_seq, temp_list_seq, humi_list_seq, missed_value_num, det_time)
                total_missed_value_num += missed_value_num
                missed_value_num = 0
            tmp_dict["id"] = result_list[-1]["id"] + 1
            result_list.append(tmp_dict)
            time_list_seq.append(int(data_list_split[1]))
            temp_list_seq.append(float(data_list_split[2]))
            humi_list_seq.append(float(data_list_split[3]))

    return result_only_lost, result_list, total_missed_value_num

def predict(url, det_time):
    content = get_data_from_url(url)
    try:
        result_only_lost, result, total_missed_value_num = predict_temp_humi(content, det_time)
    except:
        result_only_lost = []
        result = ["缺失值前训练数据不足"]
        total_missed_value_num = 0
    return result_only_lost, result, total_missed_value_num

# def update_task():
#     conn = pymysql.connect(
#         host='rm-uf61v0hgnqj2359n3195.mysql.rds.aliyuncs.com',
#         port=3306,
#         user='service_data_expect',
#         passwd='service_data_expect',
#         db='service_data_expect',
#     )
#     cur = conn.cursor()
#     cur1 = conn.cursor()
#     cur.execute("select * from filling_task where status = '0'")
#     for r in cur.fetchall():
#         id = r[0]
#         url = r[3]
#         span = r[6]
#         print(id)
#         start_time = int(time.time())
#         result_only_lost, result, total_missed_value_num = predict(url, span)
#         create_time = int(time.time())
#         computation_time = create_time - start_time
#         cur1.execute("update filling_task set create_time = %d where id = %d" % (create_time, id))
#         cur1.execute("update filling_task set computation_time = %d where id = %d" % (computation_time, id))
#         cur1.execute("update filling_task set filling_num = %d where id = %d" % (total_missed_value_num, id))
#         if len(result_only_lost):
#             data_str = str(result_only_lost).replace("'", "\"")
#             cur1.execute("update filling_task set lost_result= '%s' where id= %d" % (data_str, id))
#         elif len(result) == 1:
#             data_str_result = str(result).replace("'", "\"")
#             cur1.execute("update filling_task set lost_result= '%s' where id= %d" % (data_str_result, id))
#         else:
#             cur1.execute("update filling_task set lost_result= '%s' where id= %d" % ("", id))
#         if len(result) > 1:
#             cur1.execute("update filling_task set status=1 where id= %d" % id)
#         else:
#             cur1.execute("update filling_task set status=2 where id= %d" % id)
#         conn.commit()
#     cur.close()
#     cur1.close()
#     conn.close()

def update_task(start, end):
    conn = pymysql.connect(
        host='rm-uf61v0hgnqj2359n3bo.mysql.rds.aliyuncs.com',
        port=3306,
        user='service_data_expect',
        passwd='service_data_expect',
        db='service_data_expect',
    )
    cur = conn.cursor()
    cur1 = conn.cursor()
    cur.execute("select * from filling_task where status = '0' and id >= %d and id < %d " % (start, end))
    for r in cur.fetchall():
        id = r[0]
        url = r[3]
        span = r[6]
        print(id)
        start_time = int(time.time())
        result_only_lost, result, total_missed_value_num = predict(url, span)
        create_time = int(time.time())
        computation_time = create_time - start_time
        cur1.execute("update filling_task set create_time = %d where id = %d" % (create_time, id))
        cur1.execute("update filling_task set computation_time = %d where id = %d" % (computation_time, id))
        cur1.execute("update filling_task set filling_num = %d where id = %d" % (total_missed_value_num, id))
        if len(result_only_lost):
            data_str = str(result_only_lost).replace("'", "\"")
            cur1.execute("update filling_task set lost_result= '%s' where id= %d" % (data_str, id))
        elif len(result) == 1:
            data_str_result = str(result).replace("'", "\"")
            cur1.execute("update filling_task set lost_result= '%s' where id= %d" % (data_str_result, id))
        else:
            cur1.execute("update filling_task set lost_result= '%s' where id= %d" % ("", id))
        if len(result) > 1:
            cur1.execute("update filling_task set status=1 where id= %d" % id)
        else:
            cur1.execute("update filling_task set status=2 where id= %d" % id)
        conn.commit()
    cur.close()
    cur1.close()
    conn.close()

def filling_task():
    conn = pymysql.connect(
        host='rm-uf61v0hgnqj2359n3bo.mysql.rds.aliyuncs.com',
        port=3306,
        user='service_data_expect',
        passwd='service_data_expect',
        db='service_data_expect',
    )
    processor = 4
    cur = conn.cursor()
    cur.execute("select count(*) from filling_task where status = '0'")
    rst = cur.fetchone()
    total = rst[0]
    cur.execute("select * from filling_task where status = '0'")
    rst = cur.fetchone()
    if rst != None:
        start = rst[0]
        processes = list()
        for i in range(processor):
            size = math.ceil(total / processor)
            begin = start + size * i
            end = start + (i + 1) * size if (i + 1) * size < total else start + total
            print(begin, end)
            p = multiprocessing.Process(target=update_task, args=(begin, end,))
            print(str(i) + ' processor started !')
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print('Process end.')
        cur.close()
        conn.close()

if __name__ == "__main__":
    while True:
        filling_task()
        time.sleep(3600)










