import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# def read_data(path, is_train):
#     if is_train:
#         data = pd.read_csv(path)
#         y = data.values[:, 1]

# 处理缺失的年龄
def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]    # 把已有的【数值型】特征提取出来丢进随机森林中

    # 数据集分成已知年龄 和 未知年辆 两部分
    konwn_age= age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    x = konwn_age[:, 1:]
    y = konwn_age[:, 0]

    model = RandomForestRegressor( n_estimators=2000, # 表示的是 随机森林的个数  所用处理器的限制(-1表示没有限制)
                 n_jobs=-1,
                 random_state=1)
    model.fit(x, y)
    # 拟合开始
    x_predict = unknown_age[:, 1:]
    y_hat = model.predict(x_predict)    # 预测出来的年龄值

    # 将预测出来的年龄值填充到原来的dataFrame中去
    df.loc[(df.Age.isnull()), 'Age'] = y_hat
    return df, model    # 要返回model是因为后面预测测试集的年龄时，要用相同的模型

# 处理车厢有无
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'       # 先要处理非空缺的值，如果先处理非空的，那么就没有空的了
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    # df.Cabin[df.Cabin.notnull()] = 'Yes'
    # df.Cabin[df.Cabin.isnull()] = 'No'
    return df

# 处理出发地的有无
def set_Embarked_type(df):
    df.loc[(df.Embarked.isnull()), 'Embarked'] = 'S'
    return df

# 画学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(0.05, 1., 20), verbose=0, plot=True):
    '''
    # 画出data在某个模型上的学习曲线

    :param estimator: 所用的分类器
    :param title:   图标的标题
    :param X:       训练样本输入矩阵（m x n）
    :param y:       X对应的输出
    :param ylim:    是个元组，shape(ymin, ymax), 决定这yval的最小值和最大值
    :param cv:      做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)【其实就是我做几次学习曲线的一个流程，就是我训练-->验证-->训练-->验证。。。一整个训练集上是完整的一次，我循环多少次】
    :param n_jobs:  并行的任务数
    :param train_sizes:  【这个参数的意义就是，我每次 开始准备开始在一个训练集上做不断地训练，验证；
                            这个过程中，我每次训练的样本的大小是不一样的，而我每次我这个样本的大小占整个训练样本大小的多少，我不可能一个一个样本数量的增加来计算，
                            当遇到多分类的时候，这个比例最好要大写，以保证每次训练的时候，都能够选择出各个类别都有的样本】
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)     # 用的是元组的解析
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, cv=cv,   # train_sizes: 表示的是，每次训练的样本数，这个变量存储了这些值，shape:(n_unique_ticks,)
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    # 【注意】；train_scores, test_scores的形状是(n_ticks, n_cv_folds), 也就是说，同大小样本只是大小相同，但是内容是不同的，几种不同的内容，取决于k折（同样大小的训练样本，每次训练就有k个值）
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std, color='r', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, color='g', alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score', c='r')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation score', c='g')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # =========初探数据
    path_train = 'train.csv'
    data = pd.read_csv(path_train)
    # print(data.info())
    # # print(data.columns)
    # # print(data.index)
    # # print(data.values.shape[1]) # 多少列
    # print(data.describe())  # 展示出dataframe的详细信息（仅仅针对那些是数字的列）

    # ==========数据初步分析（各个特征的分布图）【单个特征】
    # 画出各个分布图
    # mpl.rcParams['font.sans-serif'] = [u'SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    # #
    # plt.figure(figsize=(18, 9), facecolor='w')
    # # 获救情况（柱状图） 【横坐标表示的是这个特征下数字的类别，纵坐标就是各个类别的频数（多少而不是百分比）】
    # plt.subplot2grid((2, 3), (0, 0))        # 新的分图方法，这种方法分下来的图大小不一定是大小一样的，可以自由设置它的大小
    # data.Survived.value_counts().plot(kind='bar')   # 用pandas画柱状图
    # plt.title(u'获救情况(1为获救)')
    # plt.ylabel(u'人数')
    # # plt.show()                              # 【被救的人三百多点，不到半数】
    #
    # # 阶层情况（柱状图）
    # plt.subplot2grid((2, 3), (0, 1))
    # data.Pclass.value_counts().plot(kind='bar') # value_counts可以计算出频数
    # plt.title(u'阶层情况')
    # plt.ylabel(u'人数')
    # plt.show()                                # 【三等仓的人非常多】
    #
    # # 年龄 和 存活 的情况（散点图）
    # plt.subplot2grid((2, 3), (0, 2))
    # plt.scatter(data.Survived, data.Age)
    # plt.ylabel(u'年龄')
    # plt.title(u'存活年龄情况')
    # # plt.show()                              # 【获救 和 遇难的 年龄跨度都蛮大的】

    # 存活的年龄分布是怎么样的(分布图)
    # plt.subplot2grid((2, 3), (0, 2))
    # data.Age[data.Survived==1].plot(kind='kde')
    # data.Age[data.Survived==0].plot(kind='kde')
    # plt.xlabel('Age')
    # plt.ylabel('mu')
    # plt.xlim(data.Age.min(), data.Age.max())
    # plt.title(u'存活和死亡 年龄的分布')
    # plt.legend((u'survive', u'died'), loc='upper right')
    # plt.show()

    #
    # # 各个等级的乘客年龄分布【画的是分布图，注意这里的kind是‘kde’】
    # plt.subplot2grid((2, 3), (1, 0), colspan=2)
    # data.Age[data.Pclass==1].plot(kind='kde')
    # data.Age[data.Pclass==2].plot(kind='kde')
    # data.Age[data.Pclass==3].plot(kind='kde')
    # plt.xlabel(u'年龄')
    # plt.ylabel(u'密度')
    # plt.title(u'各等级的乘客年龄分布')
    # plt.legend((u'头等舱', u'二等舱', u'三等舱'), loc='upper right') # 图例的新方法，按上面的顺序来的
    # # plt.show()                              # 【三等仓的人数最多】
    #
    # # 各船的登口岸人数
    # plt.subplot2grid((2, 3), (1, 2))
    # data.Embarked.value_counts().plot(kind='bar')
    # plt.ylabel(u'人数')
    # plt.title(u'各口岸登陆人数')
    # plt.show()                                # 【S的人数最多】

    #============特征与获救结果的关联分析
    # plt.figure()
    # 等级的获救情况
    # Survived_0 = data.Pclass[data.Survived == 0].value_counts() # 挑选出 去世的 等级情况， 之后计算出等级的统计情况【排列情况是 人数由大到小】
    # Survived_1 = data.Pclass[data.Survived == 1].value_counts() # 挑选出 存活的 等级情况， 之后计算出等级的统计情况（对应的各个等级的人数）
    # df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})  # 尽管排列顺序不一样，但是存进一个dataFrame之后，还是自动进行处理的
    # df.plot(kind='bar', stacked=True)   # stacked = True表示会重叠输出
    # plt.title(u"各乘客等级的获救情况")
    # plt.xlabel(u"乘客等级")
    # plt.ylabel(u"人数")
    # plt.show()                                # 【发现等级1的乘客获救的概率高很多，这个一定是影响最后结果的一个特征】

    # 性别的获救情况
    # Survived_0_sex = data.Sex[data.Survived == 0].value_counts() # 外面的最终会表示为 横坐标，即Sex
    # Survived_1_sex = data.Sex[data.Survived == 1].value_counts()
    # df1 = pd.DataFrame({u'获救': Survived_1_sex,
    #                     u'未获救': Survived_0_sex})
    # # df1.plot(kind='bar', stacked=True)
    # df1.plot(kind='bar')                # 不堆叠画出
    # plt.title(u'各乘客性别获救情况')
    # plt.xlabel(u'乘客性别')
    # plt.ylabel(u'人数')
    # plt.show()                                  # 【发现 女性乘客活的概率高些， 显然这个也是影响最后结果的一个特征】

    # cabin有无【也许感觉和上面的相反，其实因为后面用了转置】
    # Survived_cabin = data.Survived[pd.notnull(data.Cabin)].value_counts()
    # Survived_nocabin = data.Survived[pd.isnull(data.Cabin)].value_counts()
    # df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()     # 加了转置
    # df.plot(kind='bar', stacked=True)
    # plt.title(u"按Cabin有无看获救情况")
    # plt.xlabel(u"Cabin有无")
    # plt.ylabel(u"人数")
    # plt.show()

    #===============一、数据的预处理、特征工程
    # 1、处理缺失值
    data, model = set_missing_ages(data)
    data = set_Cabin_type(data)
    data = set_Embarked_type(data)
    # print(data.info())    # 处理完毕，没有缺失值
    # print(data)

    # 2、离散值的one_hot编码
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    # print(dummies_Cabin)
    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
    df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # print(df)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # print(type(df))

    # 4、部分特征的标准化，用sklearn.preprocessing里面的standscaler
    df['Age_scaled'] = StandardScaler().fit_transform(df['Age'])    # 原来dataFrame可以直接添加列
    # print(df)
    df['Fare_scaled'] = StandardScaler().fit_transform(df['Fare'])
    # print(df.info())

    #===============二、需要的feature字段取出来，转成Numpy格式，进行建模拟合
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*') # 还是dataFrame类型的
    train_np = train_df.as_matrix()     # 矩阵类型的

    x_train = train_np[:, 1:]
    y_train = train_np[:, 0]

    # print(x_train.shape)
    # print(y_train.shape)
    lr = LogisticRegression(penalty='l2', C=1.0)    # 很明显，这是用的 批量梯度下降法
    lr.fit(x_train, y_train)

    #======================三、进行预测
    # ********数据的预处理
    # 1、读取数据
    path_test = 'test.csv'
    data_test = pd.read_csv(path_test)
    # print(data_test.info())

    # 2、处理缺失值
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

    # 年龄的预测应该使用和训练集上的预测相同的模型
    predict_Age_features = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_predict = predict_Age_features[predict_Age_features.Age.isnull()].as_matrix()
    unknown_predict_x = unknown_predict[:, 1:]
    predict_Age = model.predict(unknown_predict_x)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predict_Age

    # Cabin的填充
    data_test = set_Cabin_type(data_test)

    # 3、one_hot编码
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


    # 4、连续特征标准化
    df_test['Age_scaled'] = StandardScaler().fit_transform(df_test['Age'])  # 原来dataFrame可以直接添加列
    df_test['Fare_scaled'] = StandardScaler().fit_transform(df_test['Fare'])

    # **********数据准备完毕，开始预测
    test_data = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')    # 有个想法就是为什么不在上面直接drop掉呢？因为我在drop之后再对部分特征进行标准化的，如果我在之前就标准化，也许就可以直接drop了
    predictions = lr.predict(test_data)     # 返回的就是预测出来的真正结果 0 还是 1
    result = pd.DataFrame({
        'PassengerId': data_test.PassengerId.as_matrix(),
        'Survived': predictions.astype(np.int32)
    })
    result.to_csv('logistic_regression_predictions.csv', index=False)
    result.to_csv('logistic_regression_predictions_2.csv')  #如果不加index=False，那么在表的第一列就会加上从0开始的序列号，并且列标那一行不算，从真实的数字行开始计算


    # =================四、判定一下当前模型所处的状态（欠拟合 、过拟合）
    plot_learning_curve(lr, 'learning curve', x_train, y_train)     # 显然，这个模型没有过拟合，可以再做些特征工程的工作，添加一些新产出的特征，或者组合特征到模型当中

    # =================五、就是看看该如何优化我们的系统了（不换模型的情况下，还是有特征是可以挖掘的）

    # 1、看看模型现在得到的模型的系数，因为系数和它们最终的判定能力强弱是正相关的
    print(pd.DataFrame({'columns': list(train_df.columns)[1:],      # 注意看看这边是对哪个数据集来看的，是对我用模型来拟合的训练集，处理过后的，【就是看 我的特征系数大小 和 对应特征 之间的关系】
                        'coef': list(lr.coef_.T)}))
    # 【通过这一步有了自己的一些对于特征的想法之后，怎么知道效果好不好呢=======》》》 交叉验证】







    # ==================六、交叉验证
    # 1、简单看看打分情况【看看这个模型在数据集上的表现情况】
    clf = LogisticRegression(penalty='l2', C=1.0)
    all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.as_matrix()[:, 1:]
    y = all_data.as_matrix()[:, 0]
    print(cross_val_score(clf, X, y, cv=5))     #[ 0.81564246  0.81005587  0.79213483  0.78651685  0.81355932]

    # 2、然后我想看看我这个模型预测错误的那些样本 到底有哪些共性 帮助我们进行特征的进一步挖掘
    

