from datetime import datetime
import multiprocessing
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np
from.forms import NameForm
from datetime import datetime
import multiprocessing
# Create your views here.
def index(request):
    df=pd.read_csv('D:\AB Folder\prototype\software\insight\compiled.csv',encoding='ISO-8859-1')
    return HttpResponse("Hello , world. about to make the prototype")
def trial(request):
    return render(request,"D:\AB Folder\prototype\software\insight\\templates\website\welcome.html",
    {"name":"Mohammad Abdul BASIT",'date':datetime.now()})
''' 
    get the data frame containing the word text

'''
df1=[pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]             #0-100,000
#pd.DataFrame()             #100,000-200,000
#df3=pd.DataFrame()             #200,000-300,000
#df4=pd.DataFrame()             #300,000-400,000
#df5=pd.DataFrame()             #400,000-len(df)
class findtweets(Thread):
    def __init__(self,df,text,my_df,j):
        Thread.__init__(self)
        self.df=df
        self.text=text
        self.my_df=my_df
        self.j=j
    def run(self):
        print(self.getName())
        for i in range(0,len(self.df)):
            if(type(self.df.iloc[i]['text'])==float):
                continue
            if(self.df.iloc[i]["text"].find(self.text)!=-1):
                self.my_df=self.my_df.append(self.df.iloc[i],ignore_index=True) 
        df1[self.j]=self.my_df
        print(len(self.my_df))
          
def get_totals(text):
    df=pd.read_csv('D:\AB Folder\prototype\software\insight\compiled.csv',encoding='ISO-8859-1')
    df.text.dropna(inplace=True)
    df_for1=df.iloc[0:100_000][:]
    df_for2=df.iloc[100_001:200_000][:]
    df_for3=df.iloc[200_001:300_000][:]
    df_for4=df.iloc[300_001:400_000][:]
    df_for5=df.iloc[400_001:][:]
    data=pd.DataFrame()
    r1=findtweets(df_for1,text,df1[0],0)
    r2=findtweets(df_for2,text,df1[1],1)
    r3=findtweets(df_for3,text,df1[2],2)
    r4=findtweets(df_for4,text,df1[3],3)
    r5=findtweets(df_for5,text,df1[4],4)
    r1.start()
    r2.start()
    r3.start()
    r4.start()
    r5.start()
    r1.join()
    r2.join()
    r3.join()
    r4.join()
    r5.join()
    print(len(df1[0]))
    data=data.append(df1[0])
    data=data.append(df1[1])
    data=data.append(df1[2])
    data=data.append(df1[3])
    data=data.append(df1[4])
    '''
    #old function
    for index,row in df.iterrows():
        if(type(row['text'])==float):
            continue
        if(row["text"].find(text)!=-1):
            data=data.append(row,ignore_index=True)
    '''
    print(len(data))
    data.text.drop_duplicates(inplace=True)
    return data
'''multithread class for '''
class pieChart(Thread):
    def __init__(self,positive,negative,neutral,pie_src):
        Thread.__init__(self)
        self.positive=positive
        self.negative=negative
        self.neutral=neutral
        self.pie_src=pie_src
    def run(self):
        print("no")
        plt.pie(np.array([self.negative,self.neutral,self.positive]),labels=['Sad','Neutral','Happy'],colors=["#0C3E3D","#4E8D8C","#C0D6D6"])
        plt.savefig(self.pie_src)
'''
generate word cloud
'''
class wordclou(Thread):
    def __init__(self,df,typ,text,src,stop):
        Thread.__init__(self)
        self.df_text=df['text']
        self.typ=typ
        self.text=text
        self.src=src
        self.stop=stop
    def run(self):
        freq=self.genFreq(self.df_text,self.text)
        self.make_cloud(freq)

    def make_cloud(self,count):
        wc = WordCloud(background_color="white", max_words=500)
    # generate word cloud
        wc.generate_from_frequencies(count)
        wc.to_file(f'{self.src}{self.typ}.jpg')

    def genFreq(self,temp,text):
        stopWords = self.stop
        count={}
        for row in temp:
            for word in row.split(" "):
                if len(word)==1:
                    continue
                if word=='fuck':
                    word='f**k'
                if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be|!|olympics|2020|olympic|this|it|so|i|me|my|myself\
        we|our|ours|ourselves|you|your|yours|yourself|yourselves|he|him|usa|india|teamgb|japan|china|basketball|swimming|us", word):
                    continue
                if word==self.text:
                    continue
                if word in stopWords:
                    continue
                if word not in count:
                    count[word]=0
                count[word]+=1
        return count


'''
class which makes the time series
'''
data0=pd.DataFrame()
data1=pd.DataFrame()
data2=pd.DataFrame()
class TimeSeries(Thread):
    def __init__(self,df,src):
        Thread.__init__(self)
        self.df=df
        self.src=src
    def run(self):
        df=self.df
        df.date.dropna()
        df["refined date"]=df["date"].apply(lambda x: x.split(" ")[0])
        count0_refined={}
        count1_refined={}
        count2_refined={}
        for index,row in df.iterrows():
            if(row["refined date"].find("Twitter")==-1 and row["date"].find("Tweet")==-1):
                if(row["Sentiment"]==0):
                    if(row["refined date"] not in count0_refined):
                        count0_refined[row["refined date"]]=0
                    count0_refined[row["refined date"]]+=1
                if(row["Sentiment"]==1):
                    if(row["refined date"] not in count1_refined):
                        count1_refined[row["refined date"]]=0
                    count1_refined[row["refined date"]]+=1
                if(row["Sentiment"]==2):
                    if(row["refined date"] not in count2_refined):
                        count2_refined[row["refined date"]]=0
                    count2_refined[row["refined date"]]+=1
        #negative cumulation
        dat0=[]
        vale0=[]
        for val in count0_refined:
            #s0 is date, s2 is time and s3 is am/pm  
            date=val
            temp=date.split("/")
            mm=int(temp[0])
            dd=int(temp[1])
            yyyy=int(temp[2])
                #----------date processing done------------- now list
            if datetime(yyyy,mm,dd) not in dat0:
                dat0.append(datetime(yyyy,mm,dd))
                vale0.append(count0_refined[val])
                continue
            for i in range(len(dat0)):
                if dat0[i]==datetime(yyyy,mm,dd):
                    vale0[i]+=count0_refined[val]
        data0["date"]=dat0
        data0["count"]=vale0
        data0.sort_values(by=['date'],inplace=True)
        #neutral cumulation
        dat1=[]
        vale1=[]
        for val in count1_refined:
            #s0 is date, s2 is time and s3 is am/pm  
            date=val
            temp=date.split("/")
            mm=int(temp[0])
            dd=int(temp[1])
            yyyy=int(temp[2])
                #----------date processing done------------- now list
            if datetime(yyyy,mm,dd) not in dat1:
                dat1.append(datetime(yyyy,mm,dd))
                vale1.append(count1_refined[val])
                continue
            for i in range(len(dat1)):
                if dat1[i]==datetime(yyyy,mm,dd):
                    vale1[i]+=count1_refined[val]
        data1["date"]=dat1
        data1["count"]=vale1
        print("****************from data 1*********************")
        print(count1_refined)
        data1.sort_values(by=['date'],inplace=True)
        #positive cumulation
        dat2=[]
        vale2=[]
        for val in count2_refined:
            #s0 is date, s2 is time and s3 is am/pm  
            date=val
            temp=date.split("/")
            mm=int(temp[0])
            dd=int(temp[1])
            yyyy=int(temp[2])
                #----------date processing done------------- now list
            if datetime(yyyy,mm,dd) not in dat2:
                dat2.append(datetime(yyyy,mm,dd))
                vale2.append(count2_refined[val])
                continue
            for i in range(len(dat2)):
                if dat2[i]==datetime(yyyy,mm,dd):
                    vale2[i]+=count2_refined[val]
        data2["date"]=dat2
        data2["count"]=vale2
        data2.sort_values(by=['date'],inplace=True)
            
'''
    the function takes in the search word from the user and operates on it
'''

def get_query(request):
    if(request.method=='POST'):
        form=NameForm(request.POST)
        if(form.is_valid()):
            text=form.cleaned_data['your_name']
            df=get_totals(text)
            #df=pd.read_csv('temp.csv',encoding='ISO-8859-1',index_col=0)
            #df.to_csv("temp.csv",index=False)
            ts_src='D:\AB Folder\prototype\software\insight\static\insight\diagrams\ts.jpg'
            ts=TimeSeries(df,ts_src)
            ts.start()
            
            df_positive=df[df['Sentiment']==2]
            df_neutral=df[df['Sentiment']==1]
            df_negative=df[df['Sentiment']==0]
            positive=len(df_positive)
            negative=len(df_negative)
            neutral=len(df_neutral)
            #make the pie charts
            pie_src='D:\AB Folder\prototype\software\insight\static\insight\diagrams\pie.jpg'
            pie=pieChart(positive,negative,neutral,pie_src)
            wc_src='D:\AB Folder\prototype\software\insight\static\insight\diagrams\wc-'
            stop=set(stopwords.words('english'))
            wc1=wordclou(df_negative,'negative',text,wc_src,stop)
            wc2=wordclou(df_neutral,'neutral',text,wc_src,stop)
            wc3=wordclou(df_neutral,'positive',text,wc_src,stop)
            pie.start()
            wc1.start()
            wc2.start()
            wc3.start()
            pie.join()
            wc1.join()
            wc2.join()
            wc3.join()
            ts.join()
            print("reached here")
            p1=multiprocessing.Process(target=plot,args=(data0,data1,data2,2))
            p1.start()
            p1.join()
            sample_positive,sample_negative,sample_neutral=get_sample(df_positive,df_neutral,df_negative)
            print(sample_positive)
            print("___________________")
            print(sample_neutral)
            print("___________________")
            print(sample_negative)
            print("___________________")
            total=positive+negative+neutral
            return render(request,'website\\results.html',{'text':text,'positive':positive,'negative':negative,'neutral':neutral,'sample_pos':sample_positive,'sample_neu':sample_neutral,'sample_neg':sample_negative,'total':total})
    else:
        form=NameForm()
    return render(request,'website\disp.html',{'form':form})
def get_sample(df_pos,df_neu,df_neg):
    p=[]
    df_pos=df_pos.sample(frac=1)
    i=0
    for index,row in df_pos.iterrows():
        p.append([f'{row["text"]}...',row['date']])
        i+=1
        if i==15:
            break
    neu=[]
    df_neu=df_neu.sample(frac=1)
    i=0
    for index,row in df_neu.iterrows():
        neu.append([f'{row["text"]}...',row['date']])
        i+=1
        if i==15:
            break
    neg=[]
    df_neg=df_neg.sample(frac=1)
    i=0
    for index,row in df_neg.iterrows():
        neg.append([f'{row["text"]}...',row['date']])
        i+=1
        if i==15:
            break
    return p,neg,neu
def plot(data0,data1,data2,c):
    #print(data1)
    tickx=[]
    for d in data1['date']:
        tickx.append(d)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(data0["date"],data0['count'],label='Negative(Sad)')
    plt.plot(data1["date"],data1['count'],color='#0D6E04',label='Neutral')
    plt.plot(data2["date"],data2['count'],color='#58046E',label='Positive(Happy)')
    plt.legend(loc='best')
    plt.xticks(tickx)
    plt.gcf().autofmt_xdate()
    plt.savefig("D:\AB Folder\prototype\software\insight\static\insight\diagrams\\timeseries.jpg")