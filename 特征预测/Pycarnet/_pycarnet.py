#%% 备用空间
from matplotlib import pyplot
#help(pandas.read_csv)
#dir(pandas) #查看所有库函数
dir(pyplot)
#%% 导入数据集
#from pycaret.datasets import get_data
#diabetes = get_data('diabetes')#错误原因是上不了国外的网
import pandas as pd
diabetes = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\diabetes.csv')#只能先下载好数据集


# 初始化参数
from pycaret.classification import *
clf = setup(data = diabetes , target = 'Class variable',train_size=0.7,sampling= True,sample_estimator= None,silent= False,verbose= True ,session_id= None,
            numeric_imputation = 'mean',categorical_imputation = 'constant',numeric_features= None,categorical_features= None,
            ignore_features= None,date_features= None,feature_selection= True ,feature_selection_threshold= 0.8)


# 返回指定的训练模型评价参数表
'''
best = compare_models()#所有模型表
top3 = compare_models(n_select = 3)#因为默认参数按预测准确率，也就是返回准确率最高的3个模型
bottom3 = compare_models(n_select = -3,sort = 'AUC')%也可以返回按AUC排序的最后3名
'''
#%%
top3 = compare_models(n_select = 3,sort = 'AUC')
#%%

#创建模型 lr是逻辑回归模型
LogicalRegression = create_model(estimator= 'lr',ensemble= False,method= None ,fold=10,round=4,cross_validation= True,verbose= True,system= True )


#校准模型 calibrate model
calibrated_LR = calibrate_model(LogicalRegression,method= 'sigmoid',fold= 10,verbose= True ,round=4 )


#保存/加载模型  save/load model
save_model(calibrated_LR , '校准的逻辑回归模型')
'''LR_saved = load_model('校准的逻辑回归模型') #配套的加载模型'''


# 预测模型(给出各类评价参数accurcy,auc,f1,kappa,mcc)，与compare model区别在于这个只预测某个模型，而不是都列出来
pre = predict_model(LogicalRegression,platform= None,authentication= None) # 返回模型的得分和预测标签


# 预测其他数据的准备工作
pre_final = finalize_model(LogicalRegression)


#预测新数据
Pre = predict_model(pre_final ,data=diabetes,probability_threshold= None,platform= None ,authentication=None,verbose= None)#新数据的预测标签和得分


# 全部样本的预测准确率
import numpy as np
Predict = np.array(Pre.values[:,9])#或Predict = np.array(Pre.Label)
Real = np.array(diabetes.values[:,8]) #最后1列是标签，或Real= np.array(Pre.Label[:,8])
j = 0
for i in range(len(Predict)):
    if Predict[i] == Real[i]:
        j=j+1
print('预测准确率为：'+str(j/len(Predict)*100)+'%')


# 预测单一样本diabetes_1.csv
diabetes_1 = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\diabetes_1.csv')
Pre1 = predict_model(pre_final ,data=diabetes_1 )
print('该数据的实际标签为：'+str(diabetes_1.values[:,8])+'\n该数据的预测标签为：'+str(Pre1.values[:,9]))


#评估模型 evaluate model
evaluate_model(LogicalRegression)#可以返回该模型可以显示的图表类型


#可视化模型
plot_model(estimator=LogicalRegression ,plot='error',verbose=True ,system = True,save = False )


#%%
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' 以下程序不运行，只用于学习和调试用'''


#解释模型 interpret model
interpret_model(LogicalRegression,observation= None , feature= None ,plot='summary') #错误原因：pip install shap to use interpret_model function
'''此函数采用训练有素的模型对象，并根据测试/保持集返回解释图。 它仅支持基于树的算法。
        此功能基于SHAP（SHapley Additive exPlanations）实现，SHAP（SHapley Additive exPlanations）是一种博弈论方法，用于解释任何机器学习模型的输出。
         它使用博弈论中的经典Shapley值及其相关扩展将最佳信用分配与本地解释联系起来
       interpret_model不支持多类问题
       For more information : https://shap.readthedocs.io/en/latest/'''


#分配模型 assign model
from pycaret.clustering import *
import pandas as pd
jewellery = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\jewellery.csv')
clu1 = setup(data = jewellery)
kmeans = create_model('kmeans')# Clustering Example
kmeans_results = assign_model(model= kmeans,transformation= False ,verbose= True )
'''此函数使用经过训练的模型对象作为模型参数，将在设置阶段传递的数据集中的每个数据点分配给一个集群。
       在使用assign_model（）之前，必须先调用create_model（）函数。
       这将使用训练后的模型返回具有推断集群的数据框
在执行无监督实验（例如聚类，异常检测或自然语言处理）时，您通常会对模型生成的标签感兴趣，例如 数据点所属的群集标识是“群集”实验中的标签。 
同样，哪个观察值是异常值，是“异常检测”实验中的二进制标记，而哪个主题文档所属的则是“自然语言处理”实验中的标记。 
这可以在PyCaret中使用assign_model函数实现，该函数将训练有素的模型对象作为单个参数。
此功能仅在pycaret.clustering，pycaret.anomaly和pycaret.nlp模块中可用。'''


# 校准模型 calibrate model
calibrated_LR = calibrate_model(LogicalRegression,method= 'sigmoid',fold= 10,verbose= True ,round=4 )
'''此功能仅在pycaret.classification模块中可用
在执行分类实验时，您不仅经常要预测类别标签，而且还希望获得预测的可能性。 
 校准良好的分类器是概率分类器，其概率输出可以直接解释为置信度。
  这些功能采用训练有素的模型对象和通过方法参数进行校准的方法。 
  方法可以是对应于Platt方法的“ Sigmoid”，也可以是非参数方法的“等渗”。 
  不建议对等渗校准使用很少的校准样品（<< 1000），因为它倾向于过拟合。 
  此函数返回一个表格，其中包含经过分类验证指标（准确性，AUC，召回率，精度，F1和Kappa）的k倍交叉验证得分以及受过训练的模型对象。'''
'''此函数将输入经过训练的估计量，并通过S型或等张回归进行概率校正。
 输出将打印一个分数网格，以折叠形式显示准确性，AUC，召回率，精度，F1，Kappa和MCC（默认= 10折叠）。
  原始估算器和经过校准的估算器（使用此函数创建）的输出可能相差不大。 
  为了查看校准差异，请在plot_model中使用“ calibration”图来查看前后的差异。'''


#优化阈值  Optimize Threshold
optimize_threshold(LogicalRegression , true_negative = 1500, false_negative = -5000)
'''此功能仅在pycaret.classification模块中可用
   在分类问题中，误报的成本几乎永远不会与误报的成本相同。 
   这样，如果您正在优化类型1和类型2错误产生不同影响的业务问题，则可以通过定义正值，负值，误报的成本来优化分类器的概率阈值，以优化自定义损失函数。
和假阴性分别。 在PyCaret中优化阈值就像编写optimize_threshold一样简单。
   它需要一个训练有素的模型对象（一个分类器），并且损失函数仅由真阳性，真阴性，假阳性和假阴性表示。
   此函数返回一个交互图，其中损失函数（y轴）表示为x轴上不同概率阈值的函数。 然后显示一条垂直线，代表该特定分类器的概率阈值的最佳值。 
   然后，可以将使用optimize_threshold优化的概率阈值用于predict_model函数，以使用自定义概率阈值生成标签。 
   通常，所有分类器都经过训练可以预测50％的阳性分类
   此功能使用自定义成本函数为经过训练的模型优化概率阈值，该函数可以使用正肯定，正负，假正（也称为I型错误）和假负（II型错误）的组合进行定义。
   此函数返回优化成本作为概率阈值在0到100之间的函数的图'''


#部署模型 deploy model
lr = create_model('lr')
final_lr = finalize_model(lr)
deploy_model(final_lr, model_name = 'lr_aws', platform = 'aws', authentication = { 'bucket'  : 'pycaret-test' })
'''使用finalize_model确定模型后，即可进行部署。 
可以使用save_model功能在本地使用经过训练的模型，该功能将转换管道和经过训练的模型保存为最终用户应用程序可以作为二进制pickle文件使用。 
或者，可以使用PyCaret将模型部署在云上'''


# save model保存模型
save_model(calibrated_LR , '校准的逻辑回归模型')
LR_saved = load_model('校准的逻辑回归模型') #配套的加载模型
'''在PyCaret中保存训练好的模型就像编写save_model一样简单。 
该函数采用经过训练的模型对象，并将整个转换管道和经过训练的模型对象保存为可传输的二进制pickle文件，以备后用'''


# 调节超参数
tune_LR = tune_model(LogicalRegression ,optimize= 'Accuracy',round= 4,fold= 10,n_iter= 50,custom_grid= None,choose_better= True )
#如果想要按照自定义的超参数进行设置，可继续使用以下命令
'''
params = {"max_depth": np.random.randint(1, (len(data.columns)*.85),20),
          "max_features": np.random.randint(1, len(data.columns),20),
          "min_samples_leaf": [2,3,4,5,6],
          "criterion": ["gini", "entropy"]
          }
tuned_LR_custom = tune_model(tune_LR , custom_grid = params)
'''


# Ensemble Model以指定的方法method创建综合决策树模型，模型可以是直接创建的模型也可以是调优后的模型，下边是用的直接创建的模型LogicalRegression
'''组装训练好的模型就像编写ensemble_model一样简单。 它仅采用一个强制性参数，即经过训练的模型对象。
此函数返回一个表，该表具有k倍的通用评估指标的交叉验证分数以及训练有素的模型对象。'''
bagged_LR = ensemble_model(LogicalRegression , method = 'Bagging')
'''Bagging，也称为Bootstrap聚合，是一种机器学习集成元算法，旨在提高统计分类和回归中使用的机器学习算法的稳定性和准确性。
它还可以减少差异并有助于避免过度拟合。 尽管它通常应用于决策树方法，但可以与任何类型的方法一起使用。 Bagging是模型平均方法的特例。'''
boosted_LR = ensemble_model(LogicalRegression , method = 'Boosting', n_estimators = 100,fold= 10,round=4,optimize= 'Accuracy',verbose= True )
'''Boosting是一种集成元算法，主要用于减少监督学习中的偏见和差异。提升属于机器学习算法系列，可将弱学习者转换为强学习者。
弱学习者被定义为仅与真实分类略相关的分类器（与随机猜测相比，它可以更好地标记示例）。
相反，学习能力强的人是与真实分类任意相关的分类器。'''


# Blend Model ：相当于是在所有模型中选择更具广泛认同性的预测结果生成最终预测 ，method控制投票方法
'''混合模型是一种集合方法，它使用估算器之间的共识来生成最终预测。
融合背后的想法是结合不同的机器学习算法，并在分类的情况下使用多数投票或平均预测概率来预测最终结果。
在PyCaret中混合模型就像编写blend_models一样简单。
此函数可用于混合可以使用blend_models中的estimator_list参数传递的特定训练模型，或者如果未传递任何列表，它将使用模型库中的所有模型。
在分类的情况下，方法参数可用于定义“软”或“硬”，其中软使用预测的概率进行投票，而硬使用预测的标签。 
此函数返回一个表，该表具有k倍的通用评估指标的交叉验证分数以及训练有素的模型对象。'''
blender = blend_models()
#如果想结合指定的模型可以使用如下语句，利用estimator_list传递
dt = create_model('dt')
rf = create_model('rf')
adaboost = create_model('ada')
blender_specific = blend_models(estimator_list = [dt,rf,adaboost], method = 'soft',fold= 10,round=4,optimize= 'Accuracy',
        verbose= True ,turbo= True ,choose_better= True )#将turbo设置为True时，它将使用径向内核的估算器列入黑名单
# 直接选择compare_models中的top3也是可以的
blender_specific1 = blend_models(estimator_list = compare_models(n_select = 5), method = 'hard')


# Stack Models 堆叠模型 meta_model如果设置为None，则默认将逻辑回归LR用作元模型
'''堆叠模型是使用元学习的整合方法。 堆叠背后的想法是建立一个元模型，该模型使用多个基本估计量的预测来生成最终预测。 
在PyCaret中堆叠模型就像编写stack_models一样简单。 此函数使用estimator_list参数获取训练模型的列表。
 所有这些模型构成了堆栈的基础层，它们的预测用作元模型的输入，可以使用meta_model参数传递该元模型。 
 如果未传递任何元模型，则默认使用线性模型。 
 在分类的情况下，方法参数可用于定义“软”或“硬”，其中软使用预测的概率进行投票，而硬使用预测的标签。 
 此函数返回一个表，该表具有经过共同验证的指标的k倍交叉验证得分以及训练有素的模型对象'''
# 创建创建用于堆叠的单个模型
ridge = create_model('ridge')
lda = create_model('lda')
gbc = create_model('gbc')
xgboost = create_model('xgboost')
# 以xgboost为基础元模型进行堆叠
stacker = stack_models(estimator_list = [ridge,lda,gbc], meta_model = xgboost,verbose= True ,fold= 10,round=4,restack= False ,
        choose_better= True ,method= 'soft',finalize= False ,plot= True  ,optimize= 'Accuracy')
# 直接选择 compare_models 的模型也是可以的，下语句表示以top1为基础，堆叠top2~top5
top5 = compare_models(n_select = 5)
stacker = stack_models(estimator_list = top5[1:], meta_model = top5[0])


# 训练测试集函数  Train test split
'''此功能仅在pycaret.classification和pycaret.regression模块中可用
机器学习的目标是建立一个很好地概括新数据的模型。 因此，在监督机器学习实验期间，数据集被分为训练数据集和测试数据集。 
测试数据集可作为新数据的代理。 仅在Train数据集上使用k倍交叉验证，才能对经过训练的机器学习模型进行评估并优化PyCaret中的超参数。
 测试数据集（也称为保持集）未在模型训练中使用，因此可以在predict_model函数下用于评估指标并确定模型是否过度拟合了数据。 
 默认情况下，PyCaret使用70％的数据集进行训练，可以使用设置中的train_size参数进行更改
 在setup函数中使用train_size更改'''

# 缺失值计算 missing value imputation
'''numeric_imputation：字符串，默认='mean'如果在数字特征中发现缺失值，则会使用特征的平均值来估算它们。
 另一个可用的选项是中位数“median”，它使用训练数据集中的中位数来估算值。
   categorical_imputation：字符串，默认=“constant”如果在分类特征中发现缺失值，则将使用恒定的“ not_available”值来插补它们。 
   另一个可用的选项是模式“mode”，它使用训练数据集中的最频繁值来估算缺失值。'''


# 改变数据类型  Changing Data Types
import pandas as pd
hepatitis = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\hepatitis.csv')
from pycaret.classification import *
clf1 = setup(data = hepatitis, target = 'Class', categorical_features = ['AGE'])#AGE列为，30.0、50.0、78.0、31.0、34.0...
#结果是将AGE列分成四列，AGE_7,AGR_70,AGE_72,AGE_78，头3列均为0，AGE_78第3行的值=1.0
'''数据集中的每个要素都有一个关联的数据类型，例如数字要素，分类要素或日期时间要素。  PyCaret的推理算法会自动检测每个功能的数据类型。 但是，有时PyCaret推断的数据类型不正确。 
确保数据类型正确很重要，因为几个下游过程取决于要素的数据类型，例如：数字和分类要素的缺失值插补应单独执行。
 要覆盖推断的数据类型，可以在安装程序中传递numeric_features，categorical_features和date_features参数。'''
'''1、numeric_features: string, default = None
如果推断的数据类型不正确，则可以使用numeric_features覆盖推断的类型。 
如果在运行安装程序时将“ column1”的类型推断为类别而不是数字，则可以通过传递numeric_features = ['column1']来覆盖此参数
'''
'''2、categorical_features: string, default = None
如果推断的数据类型不正确，则可以使用categorical_features覆盖推断的类型。 
如果在运行安装程序时将“ column1”的类型推断为数字而不是分类，那么可以通过传递categorical_features = ['column1']来使用此参数来覆盖类型
'''
'''3、date_features: string, default = None
如果数据具有在运行安装程序时不会自动检测到的DateTime列，则可以通过传递date_features ='date_column_name'来使用此参数。
 它可以与多个日期列一起使用。 日期列未在建模中使用。 而是执行特征提取，并从数据集中删除日期列。 
 如果日期列中包含时间戳记，则还将提取与时间有关的功能
'''
'''4、ignore_features: string, default = None
忽略某些维度列，可以将其传递给参数ignore_features 
'''
pokemon = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\pokemon.csv')
from pycaret.classification import *
clf2 = setup(data = pokemon, target = 'Legendary', ignore_features = ['#', 'Name'])#忽略‘#’和'Name'两列


# 特征重要性 feature importance
'''特征重要性是用于在数据集中选择对预测目标变量贡献最大的特征的过程。 使用选定的特征而不是所有特征可以减少过度装配的风险，提高准确性，并减少训练时间。 
在PyCaret中，可以使用feature_selection参数来实现。 它结合了几种受监督的特征选择技术来选择对建模最重要的特征子集。 
子集的大小可以使用安装程序中的feature_selection_threshold参数来控制'''
'''1、feature_selection: bool, default = False
设置为True时，将使用各种置换重要性技术的组合来选择特征子集，包括随机森林，Adaboost和与目标变量的线性相关。 子集的大小取决于feature_selection_param。 
通常，这用于约束特征空间，以提高建模效率。 当使用polynomial_features和feature_interaction时，强烈建议使用较低的值定义feature_selection_threshold参数
'''
'''2、feature_selection_threshold: float, default = 0.8
用于特征选择的阈值（包括新创建的多项式特征）。 较高的值将导致较高的特征空间。 
建议在使用polynomial_features和feature_interaction的情况下，特别是使用feature_selection_threshold的不同值进行多次试验。
 设置一个非常低的值可能是有效的，但可能会导致拟合不足
'''



''' 以上程序不运行，只用于学习和调试用'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


#%%
# 现在的程序是用一个数据集boston创建好模型，再预测新的数据集，用逻辑回归的
# 123.csv数据集 ，507×10的数据集
import pandas as pd
boston = pd.read_csv('C:/Users/chenbei/Documents/python数据/pycaret-master/datasets/123.csv')
# 初始化
from pycaret.regression import *
reg1 = setup(data = boston, target = 'medv')
# 创建逻辑回归模型
lr = create_model('lr')
# 用于预测其他数据集的准备工作
lr_final = finalize_model(lr)
# 导入新数据集456.csv，768 × 10
energy = pd.read_csv('C:/Users/chenbei/Documents/python数据/pycaret-master/datasets/456.csv')
#预测新数据
'''列名字必须相同，列数也要相同，行数可以不同'''
prediction = predict_model(lr_final , data = boston) #预测boston比较准确，energy不准确，这是因为两个csv数据映射不同



