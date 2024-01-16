from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent 
from langchain.agents.agent_types import AgentType
import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st

import pandas as pd
import numpy as np
import copy
def inflow_df(a):
    ll=[]
    a.dropna(axis=0,how='all',inplace=True)
    a.dropna(axis=1,how='all',inplace=True)
    ar0=np.where([type(x)==str for x in list(a.iloc[:,4])])[0]
    ls=0
    ct=0
    for i in ar0:
        
        
        if ct==0:
            cl0=list(range(i,ar0[ct+1]))
            df=a.iloc[cl0,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            
            globals()["sum_IF"]=df_dn
            ll.append(sum_IF)

        elif ct==1:
            cl0=list(range(i,ar0[ct+1]))
            df=a.iloc[cl0,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')

            globals()["sum_IF_D"]=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            ll.append(sum_IF_D)

        elif ct==2:
            cl0=list(range(i,len(a)))
            df=a.iloc[cl0,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')

            globals()["sum_IF_C"]=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            ll.append(sum_IF_C)

        ct+=1
    return ll
def sum_Total(excel_file):

    excel_file.dropna(axis=0,how='all',inplace=True)
    excel_file.dropna(axis=1,how='all',inplace=True)
    tab_li=np.append(np.where([type(x)==str for x in list(excel_file.iloc[:,4])])[0],len(excel_file)-1)
    df_name_li=["sum_to_d","sum_to_ac","sum_to_tch","sum_to_tch_cm","sum_to_tch_m","sum_to_tch_BF","sum_to_w","sum_to_dc","sum_to_dd"]
    ct=0
    sl_loc=0
    ll=[]
    for i in tab_li:
    
        if ct==0:
            cl=list(range(1,6))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='any').dropna(axis=0,how='all')
            
            
            ll.append(df)
        elif ct==1:
            cl=list(range(sl_loc,sl_loc+4))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='all')
            dd=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            
            ll.append(dd)


        else:
            cl=list(range(sl_loc,i))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            ddd=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            
            ll.append(ddd)

        ct+=1
        sl_loc=i
    return ll
def sum_Dep(excel_file):
    excel_file.dropna(axis=0,how='all',inplace=True)
    excel_file.dropna(axis=1,how='all',inplace=True)
    
    tab_li=np.append(np.where([type(x)==str for x in list(excel_file.iloc[:,4])])[0],len(excel_file))
    df_name_li=["sum_De","sum_De_PM","sum_De_Cam","sum_De_Dev","sum_De_m","sum _De_w","sum_De_c","sum_De_d"]
    ct=0
    sl_loc=0
    ll=[]
    for i in tab_li:
    
        if ct==0:
            cl=list(range(ct,7))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='any').dropna(axis=0,how='any')
            
            globals()['{}'.format(df_name_li[ct])]=df_dn
            ll.append(df_dn)

        else:
            cl=list(range(sl_loc,i))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            ddd= df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            globals()['{}'.format(df_name_li[ct])]=ddd
            ll.append(ddd)

        ct+=1
        sl_loc=i
    return ll
def sum_Loan(excel_file):
    excel_file.dropna(axis=0,how='all',inplace=True)
    excel_file.dropna(axis=1,how='all',inplace=True)
    
    tab_li=np.append(np.where([type(x)==str for x in list(excel_file.iloc[:,4])])[0],len(excel_file))
    df_name_li=["sum_Lo","sum_Lo_PM","sum_Lo_Cam","sum_Lo_Dev","sum_Lo_m","sum _Lo_w","sum_Lo_c","sum_Lo_d"]
    ct=0
    sl_loc=0
    ll=[]
    for i in tab_li:
    
        if ct==0:
            cl=list(range(ct,7))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='any').dropna(axis=0,how='any')
            
            globals()['{}'.format(df_name_li[ct])]=df_dn
            ll.append(df_dn)

        else:
            cl=list(range(sl_loc,i))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            ddd=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            globals()['{}'.format(df_name_li[ct])]=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            ll.append(ddd)

        ct+=1
        sl_loc=i
    return ll
def camp_df(excel_file):
    excel_file.dropna(axis=0,how='all',inplace=True)
    excel_file.dropna(axis=1,how='all',inplace=True)
    
    tab_li=np.append(np.where([type(x)==str for x in list(excel_file.iloc[:,4])])[0],len(excel_file))
    df_name_li=['camp_sum','camp_tr','camp_m','camp_w','camp_d_c']
    ct=0
    sl_loc=0
    ll=[]
    for i in tab_li:
    
        if ct==0:
            cl=list(range(ct,6))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='all')
            
            
            ll.append(df_dn)
        elif ct==5:
            cl=list(range(sl_loc,i))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            df_dn1=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)
            d_d_cut=np.where([type(x)==str for x in list(df_dn1.iloc[:,0])])[0]
            d_d_name=df_dn1.iloc[:,0]
            d_name_li=list(d_d_name.iloc[list(d_d_cut)])
            c_loc=0
            ct1=0
            
            for zz in list(d_d_cut):
                
                cl1=list(range(c_loc,zz))
                
                res=df_dn1.iloc[cl1,:].reset_index(drop=True)
                
                
                ll.append(res)
                ct1+=1
                c_loc=zz
                   
        else:
            cl=list(range(sl_loc,i))
            df=excel_file.iloc[cl,:]
            df_dn=df.dropna(axis=1,how='all').dropna(axis=0,how='any')
            ddd=df_dn.rename(columns=df_dn.iloc[0]).drop(df_dn.index[0]).reset_index(drop=True)

            ll.append(ddd)

        ct+=1
        sl_loc=i
    return ll

# 액셀 파일 입력시 시트이름:데이터프레임 리스트 로 구성된 딕셔너리 반환
def tbwaxlsx(xl):
    a=pd.read_excel(xl,None, engine='openpyxl')
    sn=list(a.keys())
    if 'Summary_대출_n' in sn:
        sn.remove('Summary_대출_n')
    li=[]
    ct=0
    for i in sn:
        dd=pd.read_excel(xl,sheet_name=i)
        if ct==0:
            cc=inflow_df(dd)
            li.append(cc)
        elif ct==1:
            cc=sum_Total(dd)
            li.append(cc)
        elif ct==2:
            cc=sum_Dep(dd)
            li.append(cc)
        elif ct==3:
            cc=sum_Loan(dd)
            li.append(cc)
        else:
            cc=camp_df(dd)
            li.append(cc)


        ct+=1
    di=dict(zip(sn,li))
    return di


lll=tbwaxlsx('sample_4fin.xlsx')
dff=lii['DA_구글'][4]
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-4-0613'),        # 모델 정의
    dff,                                    # 데이터프레임
    verbose=True,                          # 추론과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
an1=agent.run('가장 낮은 CPA가 나온 행의 1열 값만 출력')
an2=agent.run('가장 낮은 CPA가 나온 행의 클릭수')
an3=agent.run('가장 낮은 CPA가 나온 행의 지표 전일 대비 증감률 중 20% 이상 변화한 지표만')

llm = ChatOpenAI

st.title('This is a title')

result= llm.predict('다음 문장을 정리해줘 '+an1+an2+an3)
st.button("Reset", type="primary")
if st.button('Say hello'):
    st.write(result)
else:
    st.write('Goodbye')


