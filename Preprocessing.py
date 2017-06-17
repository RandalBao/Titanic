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
    #===============一、数据的预处理、特征工程
    # 1、处理缺失值
    data, model = set_missing_ages(data)
    data = set_Cabin_type(data)
    data = set_Embarked_type(data)

    # 2、离散值的one_hot编码
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
    df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # 4、部分特征的标准化，用sklearn.preprocessing里面的standscaler
    df['Age_scaled'] = StandardScaler().fit_transform(df['Age'])    # 原来dataFrame可以直接添加列
    df['Fare_scaled'] = StandardScaler().fit_transform(df['Fare'])
    print(df.info())
    #===============二、需要的feature字段取出来，转成Numpy格式，进行建模拟合
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*') # 还是dataFrame类型的
    train_np = train_df.as_matrix()     # 矩阵类型的

    x_train = train_np[:, 1:]
    y_train = train_np[:, 0]

    lr = LogisticRegression(penalty='l2', C=1.0)    # 很明显，这是用的 批量梯度下降法
    lr.fit(x_train, y_train)

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
    # 把样本集分割成 训练集 和 测试集 ，之后对训练集进行拟合， 之后在对测试集进行预测， 看看测试集中错掉的样本特征有哪些异同
    split_train, split_val = train_test_split(df, train_size=0.7, random_state=0)    # 原来 train_test_split 是可以将dataFrame分开来的
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')    #【去掉了没有有经过标准化的特征】
    clf = LogisticRegression(penalty='l2', C=1.0)
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

    # 对cross_validation上的数据进行预测
    val_df = split_val.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(val_df.as_matrix()[:, 1:])

    # 查看测试样本中 被预测错误的那些样本的情况【这就是为什么上面 分割成训练集和测试集时， 用的是原来的df，而不是经过筛选之后的dataFrame，因为想查看原本样本情况】
    print(split_val[predictions != val_df.as_matrix()[:, 0]])
    print(split_val[predictions != val_df.as_matrix()[:, 0]].info())

    # 去除预测错误的case，看原始dataFrame的情况【这个才是真正的原始数据集中出错样本的情况，上面的df也是经过处理之后的，不是真正的原始数据集的样子】
    origin_data_train = pd.read_csv("Train.csv")
    bad_cases = origin_data_train.loc[
        origin_data_train['PassengerId'].isin(split_val[predictions != val_df.as_matrix()[:, 0]]['PassengerId'].values)]
    print(bad_cases.info())