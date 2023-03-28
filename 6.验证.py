import numpy as np
import xlrd
import time
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
import epamodule as em

#获取压力监测值
start =time.perf_counter()
book=xlrd.open_workbook(r'实测压力.xlsx')
sheet_pres=book.sheet_by_name('pres_m')#压力监测列表
pres_ncol = sheet_pres.ncols#压力监测列数
pres_name=np.asarray(sheet_pres.row_values(rowx=0))#获取压力监测点名字
pres_number=len(pres_name)
print(pres_name)
pres_c=np.asarray([[1],[1],[1]])#16个压力监测系数，无则为1:[[1],[1],[1]]
pres_m=[]
print(pres_c)
for i in range(pres_ncol):#获取压力监测值
    cols_valus = sheet_pres.col_values(colx=i)
    del (cols_valus[0])
    pres_m.append(cols_valus)
pres_m = np.asarray(pres_m)
pres_m = pres_m*pres_c
print(pres_m)
em.ENopen("Anytown.inp",  "Anytown.rpt", "")#打开文件
nnodes = em.ENgetcount(em.EN_NODECOUNT)  # 获取节点数量，节点数量减需减去水池，水库。共:3
print(nnodes)
nnodes = nnodes - 3
llinks = em.ENgetcount(em.EN_LINKCOUNT)  # 获取管道数量，需减去阀门，泵站数量。泵：3。
llinks = llinks - 3
pres_id=[]
for i in pres_name:
    i = int(i)
    i = str(i)
    pres_id.append(i)
print(pres_id,type(pres_id))
pres_index = []
for i in pres_id:
    a=em.ENgetnodeindex(i)
    pres_index.append(a)
print(pres_index)
flow_id = []
for i in range(1,llinks + 1):
    a = em.ENgetlinkid(i)
    a = a.decode('utf-8')
    flow_id.append(a)
print(flow_id,len(flow_id))
links = []
for i in flow_id:
    a  = em.ENgetlinkindex(i)
    links.append(a)
link=np.asarray(flow_id)
def schaffer(p):
    global v
    X = p
    for i in range(1, llinks + 1):
        b = X[i - 1]
        # print("type b:", type(b))
        em.ENsetlinkvalue(i, 2, b)

    pres = [[] for _ in range(pres_number)]
    # print(pres)
    em.ENopenH()
    em.ENinitH(0)
    ii=0
    iii=0
    while True:
        em.ENrunH()
        # time.append(t)
        t = em.ENsimtime()

        if t.seconds % 3600 == 0:
            for i in pres_index:
                p = em.ENgetnodevalue(i, 11)
                name = em.ENgetnodeid(i)
                pres[ii].append(p)
                ii=ii+1
            ii=0

        tstep = em.ENnextH()
        # print('tstep',tstep)
        if (tstep == 0):
            break
    em.ENcloseH()
    # for i in range(len(pres)):
    #     del(pres[i][-1])
    pres = np.asarray(pres)
    pres = pres * pres_c
    # print(pres)
    v = np.concatenate(np.square(abs(np.subtract(pres_m, pres)))).sum()

    return v

varbound = np.array([[60,140]]*llinks)
algorithm_param = {'max_num_iteration': 5000,
                   'population_size':340,
                   'mutation_probability':0.0294,
                   'elit_ratio': 0.05,
                   'crossover_probability': 0.9,
                   'parents_portion': 0.1,
                   'crossover_type':'two_point',
                   'max_iteration_without_improv':None}
model=ga(function=schaffer, dimension=llinks, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()
em.ENclose()
a=model.output_dict
print(a)
value=a['variable']
link1=pd.DataFrame(link)
value1=pd.DataFrame(value)
res = pd.concat([link1,value1],axis=1)
res.columns = ["ID", "VALUE"]
print(res)

path = 'C:\\Users\\Administrator\\Desktop\\小论文\\随机优化-C值测试\\'

res.to_excel(path + 'anytown优化后管道C值表-25.xlsx')