import pandas as pd
import numpy as np

# =======test grammer
# list1 = [3, 3, 1, 1, 3, 2]
# list2 = [0, 0, 0 ,0, 0, 0]
# dataFrame = pd.DataFrame({'A': np.array(list1),
#                           'B': np.array(list2)})
# print(dataFrame.info())
# print(dataFrame.describe())
# print(dataFrame)
# print('===========')
# print(dataFrame.A.value_counts())       # value_counts() 表示的是 对当前Series各个值，计算出他们各自的数目
# print('===========')
# print(dataFrame.B.value_counts())

df = pd.DataFrame({'passenger': [1, 2, 3, 4, 5, 6, 7],
                   'survive': [0, 1, 1, 0, 0, 0, 1],
                   'sip': [0, 3, 1, 0, 6, 8, 0]})
# print(df)
print(df.groupby(by=['sip', 'survive']).count())