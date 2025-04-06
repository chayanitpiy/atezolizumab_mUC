import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb
import time, os, sys
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as sm
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import matthews_corrcoef
import lightgbm as ltb
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description="Run OMC Model on Discovery Dataset")
parser.add_argument('-d','--discovery', action = 'store' , dest = 'discovery_data', required = True)
parser.add_argument('-v','--validation', action = 'store' , dest = 'validation_data', required = True)
arguments = parser.parse_args()

inPut = arguments.discovery_data
if not os.path.exists(inPut):
	print("No such file : " + inPut + ".")
	sys.exit(1)

# Load dataset
discovery = pd.read_csv(inPut, index_col=0)
###built OMC model using the discovery dataset###

y = discovery.pop('recist')
X = discovery
# from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)
table = pd.DataFrame()
# for rd in np.random.randint(0, 99999999, size = 5):
for rd in np.random.randint(58554051, 58554052, size = 1):
    l_mae = []
    l_mse = []
    l_rmse = []
    l_mdae = []
    l_evs = []
    l_pc =[]
    l_pcpv = []
    l_sc =[]
    l_scpv = []
    l_r2 =[]
    l_mcc = []
    l_auc = []
    l_aucpr = []
    l_feat = []

    for i in range(2, int(X.shape[0]/2) + 1):
        dic = {'sel_feat':[k for k in list(range(2, int(X.shape[0]/2) + 1))]}
        selector = SelectKBest(f_regression, k=i)
        selector.fit(X, y)
        cols = selector.get_support(indices=True)
#         print(cols)
        features_df_new = X.iloc[:,cols]
#         print(features_df_new)
        feat = list(features_df_new.columns)
#         print(feat)
        X_new = selector.transform(X)
        cv_split = KFold(n_splits=10).split(X_new, y)
        df_pred = pd.DataFrame()
        df_test = pd.DataFrame()
        df_proba = pd.DataFrame()
        y_pred = np.array([])
        y_test = np.array([])
        y_proba = np.empty((0, 2))
        o =1
        for train_index, test_index in cv_split:
            Xtrain = X_new[train_index] ##for non-scaled data
            Xtest = X_new[test_index]
            ytrain = y[train_index]
            ytest = y[test_index]

            if df_test.empty:
                df_test = pd.DataFrame({'ind':test_index, 'test':ytest})
            else:
                current_test = pd.DataFrame({'ind':test_index, 'test':ytest})
                df_test = pd.concat([df_test, current_test])
            regressor = DecisionTreeRegressor(random_state=rd)
            model = regressor.fit(Xtrain,ytrain)
            if df_pred.empty:
                df_pred = pd.DataFrame({'ind':test_index, 'pred':model.predict(Xtest)})
            else:
                current_pred = pd.DataFrame({'ind':test_index, 'pred':model.predict(Xtest)})
                df_pred = pd.concat([df_pred, current_pred])
            o = o+1

        df_merged = pd.merge(df_test,df_pred, on ='ind').sort_values(by = 'ind')
        y_pred = df_merged['pred'].values
        y_test = df_merged['test'].values

        df_merged2 = df_merged.copy()
        df_merged2.loc[(df_merged2.test == 1.0), 'test'] = 0.0 #CR #68
        df_merged2.loc[(df_merged2.test == 2.0), 'test'] = 0.0 #PR
        df_merged2.loc[(df_merged2.test == 3.0), 'test'] = 1.0 #SD #230
        df_merged2.loc[(df_merged2.test == 4.0), 'test'] = 1.0 #PD

        df_merged2.loc[(df_merged2.pred <= 2.0), 'pred'] = 0.0
        df_merged2.loc[(df_merged2.pred > 2.0), 'pred'] = 1.0

        true = df_merged2['test'].values
        pred = df_merged2['pred'].values
        mcc = round(matthews_corrcoef(true, pred), 3)

        fpr, tpr, thresh = metrics.roc_curve(true, df_merged['pred'].values, drop_intermediate = False)
        auc = round(metrics.auc(fpr, tpr), 3)

        precision, recall, aucpr_thresholds = precision_recall_curve(true, df_merged['pred'].values) #AUCPR
        aucpr = round(metrics.auc(recall, precision), 3)

        MAE = round(sm.mean_absolute_error(y_test, y_pred), 2)
        MSE = round(sm.mean_squared_error(y_test, y_pred), 2)
        RMSE = round(sm.mean_squared_error(y_test, y_pred, squared = False), 2)
        MDAE = round(sm.median_absolute_error(y_test, y_pred), 2)
        EVS = round(sm.explained_variance_score(y_test, y_pred), 2)
        pearsoncor = stats.pearsonr(y_test, y_pred)
        spearmancor = stats.spearmanr(y_test, y_pred)
        R2_score =  round(sm.r2_score(y_test, y_pred), 2)

        l_mae.append(MAE)
        l_mse.append(MSE)
        l_rmse.append(RMSE)
        l_mdae.append(MDAE)
        l_evs.append(EVS)
        l_pc.append(pearsoncor[0])
        l_pcpv.append(pearsoncor[1])
        l_sc.append(spearmancor[0])
        l_scpv.append(spearmancor[1])
        l_r2.append(R2_score)
        l_mcc.append(mcc)
        l_auc.append(auc)
        l_aucpr.append(aucpr)
        l_feat.append(feat)

        dic['MAE'] = l_mae
        dic['MSE'] = l_mse
        dic['RMSE'] = l_rmse
        dic['MDAE'] = l_mdae
        dic['EVS'] = l_evs
        dic['MAE'] = l_mae
        dic['MSE'] = l_mse
        dic['RMSE'] = l_rmse
        dic['MDAE'] = l_mdae
        dic['EVS'] = l_evs
        dic['pearsoncor'] =l_pc
        dic['pearson_pvlue'] = l_pcpv
        dic['spearman_cor'] = l_sc
        dic['spearman_pvalue'] = l_scpv
        dic['r2'] = l_r2
        dic['MCC'] = l_mcc
        dic['AUC'] = l_auc
        dic['AUCPR'] = l_aucpr
        dic['features'] = l_feat
    print('seed', rd)
    df_dic = pd.DataFrame(dic)
    best_index = df_dic["MCC"].argmax()
    # print(best_index)
    best_params = df_dic['sel_feat'][best_index]
    # print(best_params)
    best_score = df_dic["MCC"].max()
    # print(best_score)
    AUC_score = df_dic['AUC'][best_index]
    # print(AUC_score)
    aucpr_score = df_dic['AUCPR'][best_index]
    # print('AUCPR: ', aucpr_score)

    new_row = df_dic.loc[best_index]
    # df_dic = df_dic.append(new_row, ignore_index=True)
    df_dic = pd.concat([df_dic, pd.DataFrame([new_row])], ignore_index=True)
    directory = ""
    try:
         if not os.path.exists(directory):
              os.makedirs(directory)
    except OSError:
         pass

    # directory = 'E:/CHAYANIT/Atezolizumab dataset/basal GEX_common entrezID_I021_298 patients/multiclass_regression/publication/GEX/Code_ocean/'
    filename = directory + 'seed' + str(rd) + '_' + "OMC_scores.csv"
    s = df_dic.to_csv(filename, index = False, sep = '\t')

    col = new_row['features']
    x_model_omc = X.loc[:,col]
    y_model_omc = y
    final_model = model.fit(x_model_omc,y_model_omc) ### fit the model on all data
    #save the model to disk
    model_filename = directory + "seed" + str(rd) + '_' + "saved_model.pkl"
    pickle.dump(model, open(model_filename, 'wb'))
    print('model saved')

    table_seed = pd.DataFrame([[str(rd), best_params, best_score, AUC_score, aucpr_score]], columns=["random seed", "sel_feat", "MCC", "AUC", "AUCPR"])
    table = pd.concat([table, table_seed], axis = 0, join = 'outer')
table


###tested with validation dataset###
inPut_v = arguments.validation_data
if not os.path.exists(inPut_v):
	print("No such file : " + inPut_v + ".")
	sys.exit(1)

# Load dataset
validation = pd.read_csv(inPut_v, index_col=0)

select_gene = pd.concat([validation['recist'], pd.DataFrame(validation.loc[:,col])], axis = 1)
select_gene

import pickle
filename = directory + 'seed58554051_saved_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)
print(model)

y_ind = select_gene.pop('recist')
X_ind = select_gene

test_index = select_gene.index
df_pred = pd.DataFrame()
y_pred = np.array([])

df_pred = pd.DataFrame({'ind':test_index, 'pred':model.predict(X_ind)})
df_pred = df_pred.sort_values('ind')
#print(df_pred)
y_pred = df_pred['pred'].values

df_y_ind = pd.DataFrame({'ind':test_index, 'true':y_ind})
merge = pd.merge(df_pred, df_y_ind, how = 'outer', on = 'ind')
merge.rename(columns={'ind' : 'Patient'}, inplace=True)
y_ind2 = merge['true'].values
y_pred2 = merge['pred'].values

df_merged2 = merge.copy()
df_merged2.loc[(df_merged2.true == 1.0), 'true'] = 0.0
df_merged2.loc[(df_merged2.true == 2.0), 'true'] = 0.0
df_merged2.loc[(df_merged2.true == 3.0), 'true'] = 1.0
df_merged2.loc[(df_merged2.true == 4.0), 'true'] = 1.0

df_merged2.loc[(df_merged2.pred <= 2.0), 'pred'] = 0.0
df_merged2.loc[(df_merged2.pred > 2.0), 'pred'] = 1.0
        
y_ind2 = df_merged2['true'].values
y_pred2 = df_merged2['pred'].values
mcc = round(matthews_corrcoef(y_ind2, y_pred2), 3)
print('MCC:', mcc)

fpr, tpr, thresh = metrics.roc_curve(df_merged2['true'], merge['pred'], drop_intermediate = False)
auc = round(metrics.auc(fpr, tpr), 3)
print('AUC:', auc)

precision, recall, aucpr_thresholds = precision_recall_curve(df_merged2['true'], merge['pred']) #AUCPR
aucpr = round(metrics.auc(recall, precision), 3)
print('AUCPR:', aucpr)

table_va = pd.DataFrame([[str(rd), mcc, auc, aucpr]], columns=["random seed", "MCC", "AUC", "AUCPR"])
filename_1 = directory + 'seed' + str(rd) + '_' + "validation_score.csv"
s = table_va.to_csv(filename_1, index = False, sep = '\t')

col_df = pd.DataFrame(col, columns=["predictive_features"])
filename_2 = directory + 'seed' + str(rd) + '_' + "predictive_features.csv"
col_df
s = col_df.to_csv(filename_2, index = False, sep = '\t')
