#   _*_coding_*_:utf-8

from epyt import epanet
import numpy as np
import pandas as pd
import time
import random

Dataframe_features1 = pd.DataFrame(columns=[i for i in range(1, 101)])
Dataframe_labels1 = pd.DataFrame(columns=[i for i in range(101, 426)])
for j in range(2):
    Dataframe_features = pd.DataFrame(columns=[i for i in range(1, 101)])
    Dataframe_labels = pd.DataFrame(columns=[i for i in range(101, 426)])
    for i in range(100):
        #   C值随机变量设置
        random_roughness = []
        for i in range(1,38):
            a = random.uniform(60,140)
            random_roughness.append(a)

        #   获取感兴趣的节点和流量
        d = epanet('Anytown.inp')
        my_node = ['30','90','120']
        del_node = ['30','90','120','500','501','502']
        my_link = ['2001','2002','2003']    #   用三台泵的输入流量和作为入口流量
        my_link_index = d.getLinkIndex(my_link)
        my_node_index = d.getNodeIndex(my_node)
        del_node_index = d.getNodeIndex(del_node)
        node_index = d.getNodeIndex()
        # print(del_node_index)
        # print(node_index)
        # print(my_link_index)
        node_index_feature = [2,8,11]
        node_index_label = [1,3,4,5,6,7,9,10,12,13,14,15,16]
        #
        for i in random_roughness:
            p = [i for i in range(1,38)]
            for j in p:
                d.setLinkRoughnessCoeff(j,i)
        # print(d.getLinkRoughnessCoeff(1))

        #   设置模拟持续时间
        hrs = 24
        d.setTimeSimulationDuration(hrs * 3600)

        #   执行水力模拟
        etstep = 3600
        d.setTimeReportingStep(etstep)
        d.setTimeHydraulicStep(etstep)
        start = time.time()
        d.openHydraulicAnalysis()
        # print(d.getLinkCount())
        d.initializeHydraulicAnalysis()
        link_flow_feature,node_pressure_feature,node_pressure_label=[],[],[]
        while True:
            t = d.runHydraulicAnalysis()
            tstep = d.nextHydraulicAnalysisStep()
            for i in my_link_index:
                link_flow_feature.append(d.getLinkFlows(i))
            for i in node_index_feature:
                node_pressure_feature.append(d.getNodePressure(i))

            for i in node_index_label:
                node_pressure_label.append(d.getNodePressure(i))

            if tstep <= 0:
                break
        d.closeHydraulicAnalysis()
        stop = time.time()

        #   生成DataFrame:1.特征共100列：为入口流量0-24小时，30，90，120节点的压力值（24小时）
        #                2.标签列:325列，剩余13个节点（24小时）的压力
        list1 = link_flow_feature[0:25]
        list2 = link_flow_feature[25:50]
        list3 = link_flow_feature[50:75]
        link_flow_feature=np.sum([list1,list2,list3],axis=0).tolist()
        features = link_flow_feature + node_pressure_feature
        labels = node_pressure_label

        #   获取合适的模拟情况
        if len(features) == 100 and len(labels) == 325:
            features = np.array(features).reshape(1,100)
            labels = np.array(labels).reshape(1,325)
            df_features = pd.DataFrame(features,columns=[i for i in range(1,101)])
            df_labels = pd.DataFrame(labels,columns=[i for i in range(101,426)])
            Dataframe_features = Dataframe_features.append(df_features,ignore_index=True)
            Dataframe_labels = Dataframe_labels.append(df_labels,ignore_index=True)

    Dataframe_features1 = Dataframe_features1.append(Dataframe_features,ignore_index=True)
    Dataframe_labels1 = Dataframe_labels1.append(Dataframe_labels, ignore_index=True)

shuju = pd.concat([Dataframe_features1,Dataframe_labels1],axis=1)
# print(shuju.shape[0],shuju.shape[1])

shuju.to_excel('./shuju/101.xlsx')
# print(Dataframe_features1)
# print(Dataframe_labels1)
print(shuju)