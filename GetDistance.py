#获取距离
import numpy as np

def levenshtein_distance(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    return _levenshtein_distance_matrix(string1, string2)[n1, n2]

def damerau_levenshtein_distance(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    return _levenshtein_distance_matrix(string1, string2, True)[n1, n2]

def get_ops(string1, string2, is_damerau=False):
    dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
    i, j = dist_matrix.shape
    i -= 1
    j -= 1
    ops = list()
    while i != -1 and j != -1:
        if is_damerau:
            if i > 1 and j > 1 and string1[i-1] == string2[j-2] and string1[i-2] == string2[j-1]:
                if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
#                     ops.insert(0, ('transpose', i - 1, i - 2))
                    ops.insert(0, ( i - 1, i - 2,string1[i-1],string1[i-2],'transpose'))
                    i -= 2
                    j -= 2
                    continue
        index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
        
        if index == 0:
            if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
#                 ops.insert(0, ('replace', i - 1, j - 1))
                ops.insert(0, ( i - 1, j - 1,string1[i-1],string2[j-1],'replace'))
            i -= 1
            j -= 1
        elif index == 1:
#             ops.insert(0, ('insert', i - 1, j - 1))
            ops.insert(0, ( i - 1, j - 1,string1[i-1],string2[j-1],'insert'))
            j -= 1
        elif index == 2:
#             ops.insert(0, ('delete', i - 1, i - 1))
            ops.insert(0, ( i - 1, i - 1,string1[i-1],string1[i-1],'delete'))
            i -= 1
    return ops

def execute_ops(ops, string1, string2):
    strings = [string1]
    string = list(string1)
    shift = 0
    for op in ops:
        i, j = op[1], op[2]
        if op[0] == 'delete':
            del string[i + shift]
            shift -= 1
        elif op[0] == 'insert':
            string.insert(i + shift + 1, string2[j])
            shift += 1
        elif op[0] == 'replace':
            string[i + shift] = string2[j]
        elif op[0] == 'transpose':
            string[i + shift], string[j + shift] = string[j + shift], string[i + shift]
        strings.append(''.join(string))
    return strings

def _levenshtein_distance_matrix(string1, string2, is_damerau=False):
    n1 = len(string1)
    n2 = len(string2)
    d = np.zeros((n1 + 1, n2 + 1), dtype=int)
    for i in range(n1 + 1):
        d[i, 0] = i
    for j in range(n2 + 1):
        d[0, j] = j
    for i in range(n1):
        for j in range(n2):
            if string1[i] == string2[j]:
                cost = 0
            else:
                cost = 1
            d[i+1, j+1] = min(d[i, j+1] + 1, # insert
                              d[i+1, j] + 1, # delete
                              d[i, j] + cost) # replace
            if is_damerau:
                if i > 0 and j > 0 and string1[i] == string2[j-1] and string1[i-1] == string2[j]:
                    d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost) # transpose
    return d



# 改进的编辑距离算法-4D
def revised_levenshtein_distance(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    return _revised_levenshtein_distance_matrix(string1, string2)[0,0,n1, n2]
def revised_get_ops(string1, string2, s1s=0,s2s=0,s1e=2,s2e=2,is_damerau=False):
    dist_matrix=_revised_levenshtein_distance_matrix(string1, string2)
    i, j = s1e,s2e#这里修改成长度了
#     i -= 1
#     j -= 1
    ops = list()
    while i >= s1s-1 and j >= s1s-1 :
        if is_damerau:
            for p in range(1,i):
                for q in range(1,j):
                    if i-p-1 >= 0 and j-q-1 >= 0 and string1[i-1] == string2[j-q-1] and string1[i-p-1] == string2[j-1] and string2[j-q-1]!=string2[j-1]:
                        if dist_matrix[s1s,s2s,i-p-1, j-q-1]+dist_matrix[i-p,j-q,i-1, j-1]<dist_matrix[s1s,s2s,i, j]:
#                             ops.insert(0, ( i- 1, i-p-1,string1[i-1],string1[i-p-1],'transpose'))
                            myopt=( i- 1, i-p-1,string1[i-1],string1[i-p-1],'transpose')
                            res1,res2=[],[]
                            res1= revised_get_ops(string1, string2,s1s=s1s,s2s=s2s,s1e=i-p-1,s2e=j-q-1,is_damerau=is_damerau)
                            res2= revised_get_ops(string1, string2, s1s=i-p,s2s=j-q,s1e=i-2,s2e=j-2,is_damerau=is_damerau)
                            print(s1s,i-p-1,s2s,j-q-1,"..",i-p,i-2,j-q,j-2,string1[s1s:i-p-1],string2[s2s:j-q-1],string1[i-p:i-2],string2[j-q:j-2],res1,res2)
#                             print(s1s,i-p-1,s2s,j-q-1,"..",i-p,i-2,j-q,j-2,string1[s1s:i-p-1],string2[s2s,j-q-1],string1[i-p:i-2],string2[j-q:j-2],res1,res2)
                            ops=res1+[myopt]+res2+ops
                            return ops

        index = np.argmin([dist_matrix[s1s,s2s,i-1, j-1], dist_matrix[s1s,s2s,i, j-1], dist_matrix[s1s,s2s,i-1, j]])
        if len(string1[s1s:s1e])==0 or len(string2[s2s:s2e])==0:
            index = np.argmin([1000000, dist_matrix[s1s,s2s,i, j-1], dist_matrix[s1s,s2s,i-1, j]])
        if index == 0:
            if dist_matrix[s1s,s2s,i, j] > dist_matrix[s1s,s2s,i-1, j-1] and i-1 >=0 and j-1>=0:
#                 ops.insert(0, ('replace', i - 1, j - 1))
                ops.insert(0, ( i - 1, j - 1,string1[i-1],string2[j-1],'replace'))
            i -= 1
            j -= 1
        elif index == 1 :
#             ops.insert(0, ('insert', i - 1, j - 1))
            if  j-1>=0:
                ops.insert(0, ( i - 1, j - 1,-1,string2[j-1],'insert'))
            j -= 1
        elif index == 2 :
#             ops.insert(0, ('delete', i - 1, i - 1))
            if i-1>=0:
                ops.insert(0, ( i - 1, i - 1,string1[i-1],string1[i-1],'delete'))
            i -= 1
    return ops
def _revised_levenshtein_distance_matrix(string1, string2,is_damerau=True):
    n1 = len(string1)
    n2 = len(string2)
    d = np.zeros((n1+1 ,n2+1 ,n1 + 1, n2 + 1), dtype=int)#s1从n1+1开始，s2从n2+1开始，s1到n1+1结束，s2到n2+1结束
    #状态变化，相等cost=0，不相等cost=1，insert,delete,replace,transpose(遍历可以替换的，前一半cost+后一半cost+item_cost)
    #在某个end之前，所有start-end都要被赋值。要把小于它span粒度的都算出来。
    #初始化
    for s1s in range(n1 + 1):
        s1e=s1s
        for s2s in range(n2+1):
            for s2e in range(s2s,n2+1):
                d[s1s,s2s,s1e,s2e]=s2e-s2s

    for s2s in range(n2 + 1):
        s2e=s2s
        for s1s in range(n1+1):
            for s1e in range(s1s,n1+1):
                d[s1s,s2s,s1e,s2e]=s1e-s1s  

    for span1 in range(0,n1+1):
        for span2 in range(0,n2+1):
            for s1s in range(0,n1):
                for i in range(s1s,min(s1s+span1,n1)):
                    for s2s in range(0,n2):
                        for j in range(s2s,min(s2s+span2,n2)):
                            if string1[i] == string2[j]:
                                cost = 0
                            else:
                                cost = 1
                            d[s1s,s2s,i+1, j+1] = min(d[s1s,s2s,i, j+1] + 1, # insert
                                              d[s1s,s2s,i+1, j] + 1, # delete
                                              d[s1s,s2s,i, j] + cost) # replace
                            if is_damerau:
                                tcost=n1+n2
                                mcost=n1+n2
                                for p in range(1,i+1):
                                    for q in range(1,j+1):
                                        if i-p >= 0 and j-q >= 0 and string1[i] == string2[j-q] and string1[i-p] == string2[j] and string2[j-q]!=string2[j]:
                                            tcost=d[s1s,s2s,i-p, j-q]+d[i-p+1,j-q+1,i, j]+cost
                                            # if s1s==0 and s2s==0:
                                            #     print(i,j,p,q,tcost,d[s1s,s2s,i-p, j-q],d[i-p+1,j-q+1,i, j])
                                            if tcost<mcost:
                                                mcost=tcost
                                d[s1s,s2s,i+1, j+1] = min(d[s1s,s2s,i+1, j+1], mcost) # transpose
    # print("3#",d[1,1,1,1])
    return d 

def get_score(resultdf,idx,chdf,is_damerau=True,revised=True):
    t=resultdf.iloc[idx]
    string1=t['label']
    string2=chdf[chdf.ch==t['ch']].iloc[0].label
    word1=string1
    word=string2
    if revised:
        ops = revised_get_ops(string1, string2, s1s=0,s2s=0,s1e=len(string1),s2e=len(string2),is_damerau=is_damerau)
    else:
        dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
        ops = get_ops(string1, string2, is_damerau=is_damerau)
    res= len(ops)
    basescore=(len(word)+len(word1)-res)/(len(word)+len(word1))#莱温斯坦比
    predict=t['predict']
    scorelist=[i[0]['logit'] for i in predict]
    adjscore=np.mean(scorelist)
#     print(0.9*basescore+0.1*adjscore)
    return 0.9*basescore+0.1*adjscore
    
    
#     print("你的分数是：",round(score*100,1))
    
# get_score(resultdf,0,chdf)
def get_problems(string1,string2,is_damerau=True,revised=True):
    problems=[]
    if revised:
        ops = revised_get_ops(string1, string2, s1s=0,s2s=0,s1e=len(string1),s2e=len(string2), is_damerau=is_damerau)
    else:
        dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
        ops = get_ops(string1, string2, is_damerau=is_damerau)
    if len(string1)-len(string2)>0:
        problems.append("多笔画")
    elif len(string1)-len(string2)<0:
        problems.append("少笔画")
    result=ops
    comments=[]
    for i in result:
        if i[4]=="insert":
            t="增加"
            comment="你在第"+str(i[0]+1)+"笔后要"+t+str(i[3])
        elif i[4]=="delete":
            t="删除"
            comment="你的第"+str(i[0]+1)+"笔（"+i[2]+"）要"+t

        elif i[4]=="replace":
            t="替换"
            comment="【笔画错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要"+t+"成"+str(i[3])
            if "笔画错误" not in problems:
                problems.append("笔画错误")
        else:
            t="交换"
            comment="【笔顺错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要和"+"第"+str(i[1]+1)+"笔（"+str(i[3])+"）"+t
            if "笔顺错误" not in problems:
                problems.append("笔顺错误")
        comments.append(comment)
    return problems
def get_comments(string1,string2,is_damerau=True,revised=True):
    problems=[]
    if revised:
        ops = revised_get_ops(string1, string2, s1s=0,s2s=0,s1e=len(string1),s2e=len(string2), is_damerau=is_damerau)
    else:
        dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
        ops = get_ops(string1, string2, is_damerau=is_damerau)
    if len(string1)-len(string2)>0:
        problems.append("多笔画")
    elif len(string1)-len(string2)<0:
        problems.append("少笔画")
    result=ops
    comments=[]
    for i in result:
        if i[4]=="insert":
            t="增加"
            comment="你在第"+str(i[0]+1)+"笔后要"+t+str(i[3])
        elif i[4]=="delete":
            t="删除"
            comment="你的第"+str(i[0]+1)+"笔（"+i[2]+"）要"+t

        elif i[4]=="replace":
            t="替换"
            comment="【笔画错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要"+t+"成"+str(i[3])
            if "笔画错误" not in problems:
                problems.append("笔画错误")
        else:
            t="交换"
            comment="【笔顺错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要和"+"第"+str(i[1]+1)+"笔（"+str(i[3])+"）"+t
            if "笔顺错误" not in problems:
                problems.append("笔顺错误")
        comments.append(comment)
    return comments
def get_problems_and_comments(string1,string2,is_damerau=True,revised=True):
    problems=[]
    if revised:
        ops = revised_get_ops(string1, string2, s1s=0,s2s=0,s1e=len(string1),s2e=len(string2), is_damerau=is_damerau)
    else:
        dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
        ops = get_ops(string1, string2, is_damerau=is_damerau)
    if len(string1)-len(string2)>0:
        problems.append("多笔画")
    elif len(string1)-len(string2)<0:
        problems.append("少笔画")
    result=ops
    comments=[]
    for i in result:
        if i[4]=="insert":
            t="增加"
            comment="你在第"+str(i[0]+1)+"笔后要"+t+str(i[3])
        elif i[4]=="delete":
            t="删除"
            comment="你的第"+str(i[0]+1)+"笔（"+i[2]+"）要"+t

        elif i[4]=="replace":
            t="替换"
            comment="【笔画错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要"+t+"成"+str(i[3])
            if "笔画错误" not in problems:
                problems.append("笔画错误")
        else:
            t="交换"
            comment="【笔顺错误】你的第"+str(i[0]+1)+"笔（"+str(i[2])+"）要和"+"第"+str(i[1]+1)+"笔（"+str(i[3])+"）"+t
            if "笔顺错误" not in problems:
                problems.append("笔顺错误")
        comments.append(comment)
    return problems,comments