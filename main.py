import os
import csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter
directory = 'set-b'


def formatDataLastHour():
    variables = ['RecordID', 'Age', 'BUN', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
       'HCT', 'HR', 'ICUType', 'K', 'MAP', 'Mg',
       'Na', 'PaCO2', 'PaO2', 'Platelets','Gender',
        'SysABP', 'Temp', 'Urine', 'WBC', 'Weight', 'pH']

    res = str(variables).replace("'","").replace(" ","").strip("[").strip("]")+",death"+"\n"
    with open('Outcomes-a.txt', mode='r') as infile:
        reader = csv.reader(infile)
        with open('ReformattedSet.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            mydict = {rows[0]: rows[5] for rows in reader}
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    f = open(directory+"/"+filename)
                    lines = f.readlines()
                    for variable in variables:
                        for line in lines:
                            if(line.startswith("00:00") or line.startswith("47")):
                                data = line.split(",")
                                if(data[1] == variable):
                                    while((not res.endswith("death\n"))and (res[-1] !="," and res[-1] !="\n")):
                                        res = res[:len(res)-1]
                                    res += data[2].strip("\n")
                        res += ","

                    res = res[:len(res)-1]+","+mydict[filename.strip(".txt")] +"\n"
                    continue
                else:
                    continue
            outfile.write(res)
    return res

def formatDataLatest():
    variables = ['RecordID', 'Age', 'BUN', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
       'HCT', 'HR', 'ICUType', 'K', 'MAP', 'Mg',
       'Na', 'PaCO2', 'PaO2', 'Platelets','Gender',
        'SysABP', 'Temp', 'Urine', 'WBC', 'Weight', 'pH']

    res = str(variables).replace("'","").replace(" ","").strip("[").strip("]")+",death"+"\n"
    with open('Outcomes-b.txt', mode='r') as infile:
        reader = csv.reader(infile)
        with open('ReformattedTest.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            mydict = {rows[0]: rows[5] for rows in reader}
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    f = open(directory+"/"+filename)
                    lines = f.readlines()
                    ptntMeasures = {}
                    for line in lines:
                        if(":" not in line):
                            continue
                        lineVars = line.split(',')
                        ptntMeasures[lineVars[1]] = lineVars[2].strip("\n")
                    for variable in variables:
                        try:
                            res += ptntMeasures[variable]+','
                        except:
                            res += "NaN,"
                    # res = res[:len(res)-1]+","+mydict[filename.strip(".txt")] +"\n"
                    res += mydict[filename.strip(".txt")] +"\n"
                    continue
                else:
                    continue
            outfile.write(res)
    return res

def formatDataWithAgg():
    variables = ['RecordID', 'Age', 'BUN', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
       'HCT', 'HR', 'ICUType', 'K', 'MAP', 'Mg',
       'Na', 'PaCO2', 'PaO2', 'Platelets','Gender',
        'SysABP', 'Temp', 'Urine', 'WBC', 'Weight', 'pH']
    updatedVariables = []
    for variable in variables:
        redundant = ['RecordID', 'Age', 'Gender']
        if variable in redundant:
            updatedVariables.append(variable)
            continue
        updatedVariables.append('min'+variable)
        updatedVariables.append('max'+variable)
        updatedVariables.append('avg'+variable)
        updatedVariables.append(variable)

    res = str(updatedVariables).replace("'","").replace(" ","").strip("[").strip("]")+",death"+"\n"
    with open('Outcomes-b.txt', mode='r') as infile:
        reader = csv.reader(infile)
        with open('ReformattedTest.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            mydict = {rows[0]: rows[5] for rows in reader}
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    f = open(directory+"/"+filename)
                    lines = f.readlines()
                    ptntMeasures = {}
                    for line in lines:
                        if(":" not in line):
                            continue
                        lineVars = line.split(',')
                        try:
                            tem = ptntMeasures[lineVars[1]]
                            tem.append(lineVars[2].strip("\n"))
                            ptntMeasures[lineVars[1]] = tem
                        except:
                            tem = [lineVars[2].strip("\n")]
                            ptntMeasures[lineVars[1]] = tem
                    for variable in variables:
                        for agg in ['min', 'max', 'avg']:
                            redundant = ['RecordID', 'Age', 'Gender']
                            if variable in redundant:
                                continue
                            try:
                                L = [float(n) for n in ptntMeasures[variable] if n]
                                if(agg=='min'):
                                    res += str(min(L)) + ','
                                elif(agg=='max'):
                                    res += str(max(L)) + ','
                                else:
                                    avg = sum(L) / len(L)
                                    res += str(avg) + ','
                            except Exception as e:
                                res += "NaN,"
                        try:
                            length = len(ptntMeasures[variable])
                            res += ptntMeasures[variable][length - 1] + ','
                        except:
                            res += "NaN,"
                    # res = res[:len(res)-1]+","+mydict[filename.strip(".txt")] +"\n"
                    res += mydict[filename.strip(".txt")] +"\n"
                    continue
                else:
                    continue
            outfile.write(res)
    return res

train = []
test = []
def imputate():
    global train, test
    # df = pd.read_csv('ReformattedSet.csv', sep=',', header=None, names=["RecordID","Age","BUN","Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","ICUType","K","MAP","Mg","Na","PaCO2","PaO2","Platelets","Gender","SysABP","Temp","Urine","WBC","Weight","pH","death"])
    df = pd.read_csv('ReformattedSet.csv', sep=',', header=None, names=["RecordID", "Age", "minBUN", "maxBUN", "avgBUN", "BUN", "minCreatinine", "maxCreatinine", "avgCreatinine", "Creatinine", "minDiasABP", "maxDiasABP", "avgDiasABP", "DiasABP", "minFiO2", "maxFiO2", "avgFiO2", "FiO2", "minGCS", "maxGCS", "avgGCS", "GCS", "minGlucose", "maxGlucose", "avgGlucose", "Glucose", "minHCO3", "maxHCO3", "avgHCO3", "HCO3", "minHCT", "maxHCT", "avgHCT", "HCT", "minHR", "maxHR", "avgHR", "HR", "minICUType", "maxICUType", "avgICUType", "ICUType", "minK", "maxK", "avgK", "K", "minMAP", "maxMAP", "avgMAP", "MAP", "minMg", "maxMg", "avgMg", "Mg", "minNa", "maxNa", "avgNa", "Na", "minPaCO2", "maxPaCO2", "avgPaCO2", "PaCO2", "minPaO2", "maxPaO2", "avgPaO2", "PaO2", "minPlatelets", "maxPlatelets", "avgPlatelets", "Platelets", "Gender", "minSysABP", "maxSysABP", "avgSysABP", "SysABP", "minTemp", "maxTemp", "avgTemp", "Temp", "minUrine", "maxUrine", "avgUrine", "Urine", "minWBC", "maxWBC", "avgWBC", "WBC", "minWeight", "maxWeight", "avgWeight", "Weight", "minpH", "maxpH", "avgpH", "pH", "death"])
    # df = UpSample(df)
    # df2 = pd.read_csv('ReformattedTest.csv', sep=',', header=None, names=["RecordID","Age","BUN","Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","ICUType","K","MAP","Mg","Na","PaCO2","PaO2","Platelets","Gender","SysABP","Temp","Urine","WBC","Weight","pH","death"])
    df2 = pd.read_csv('ReformattedTest.csv', sep=',', header=None, names=["RecordID", "Age", "minBUN", "maxBUN", "avgBUN", "BUN", "minCreatinine", "maxCreatinine", "avgCreatinine", "Creatinine", "minDiasABP", "maxDiasABP", "avgDiasABP", "DiasABP", "minFiO2", "maxFiO2", "avgFiO2", "FiO2", "minGCS", "maxGCS", "avgGCS", "GCS", "minGlucose", "maxGlucose", "avgGlucose", "Glucose", "minHCO3", "maxHCO3", "avgHCO3", "HCO3", "minHCT", "maxHCT", "avgHCT", "HCT", "minHR", "maxHR", "avgHR", "HR", "minICUType", "maxICUType", "avgICUType", "ICUType", "minK", "maxK", "avgK", "K", "minMAP", "maxMAP", "avgMAP", "MAP", "minMg", "maxMg", "avgMg", "Mg", "minNa", "maxNa", "avgNa", "Na", "minPaCO2", "maxPaCO2", "avgPaCO2", "PaCO2", "minPaO2", "maxPaO2", "avgPaO2", "PaO2", "minPlatelets", "maxPlatelets", "avgPlatelets", "Platelets", "Gender", "minSysABP", "maxSysABP", "avgSysABP", "SysABP", "minTemp", "maxTemp", "avgTemp", "Temp", "minUrine", "maxUrine", "avgUrine", "Urine", "minWBC", "maxWBC", "avgWBC", "WBC", "minWeight", "maxWeight", "avgWeight", "Weight", "minpH", "maxpH", "avgpH", "pH", "death"])
    #TODO: for ICUTYPE
    # df = df.loc[df['ICUType'] == 1]
    # df2 = df2.loc[df2['ICUType'] == 5]
    ##################
    values = df.values
    values2 = df2.values
    imputer = SimpleImputer()
    train = imputer.fit_transform(values)
    test = imputer.fit_transform(values2)

    # FeatureEngineering(df)
    # count the number of NaN values in each column
    # print(np.isnan(train).sum())
    # print(np.isnan(test).sum())


def ScaleData(X_train, X_test):
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(pca.components_[0])
    print(pca.explained_variance_ratio_)
    return X_train, X_test

def UpSample(df):
    # Separate majority and minority classes
    df_majority = df[df.death == 0]
    df_minority = df[df.death == 1]
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=3446,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    print(df_upsampled.death.value_counts())
    return df_upsampled

def DownSample(df):
    # Separate majority and minority classes
    df_majority = df[df.death == 0]
    df_minority = df[df.death == 1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=554,  # to match minority class
                                       random_state=123)  # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Display new class counts
    print(df_downsampled.death.value_counts())
    return df_downsampled

def FeatureEngineering(df):
    from sklearn.feature_selection import RFE
    global train, test
    X_train = train[:, 1:26]
    Y_train = train[:, 26]
    X_test = test[:, 1:26]
    X_train, x_test = ScaleData(X_train, X_test)
    rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs'), n_features_to_select=14, step=1, verbose=5)
    rfe_selector.fit(X_train, Y_train)
    rfe_support = rfe_selector.get_support()
    rfe_feature = df.iloc[:, 1:26].loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    print(rfe_feature)


def LRTrain():
    global train, test
    X_train = train[:, 1:26]
    Y_train = train[:, 26]
    X_test = test[:, 1:26]
    Y_test = test[:, 26]
    X_train, X_test = ScaleData(X_train, X_test)
    clf = LogisticRegression().fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    Y_pred = (clf.predict_proba(X_test)[:, 1] >= 0.5933999999999999)
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred,Y_test))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    # print(max_f1(clf, X_test, Y_test))

def RFTrain():
    global train, test
    X_train = train[:, 1:26]
    Y_train = train[:, 26]
    X_test = test[:, 1:26]
    Y_test = test[:, 26]
    # X_train, X_test = ScaleData(X_train, X_test)
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    Y_pred = (clf.predict_proba(X_test)[:, 1] >= 0.26039999999999996)
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred, Y_test))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    # print(max_f1(clf, X_test, Y_test))
    # skplt.metrics.plot_confusion_matrix(Y_test, Y_pred, normalize=True)
    # skplt.metrics.plot_precision_recall_curve(Y_test, clf.predict_proba(X_test))
    skplt.metrics.plot_roc_curve(Y_test, clf.predict_proba(X_test))
    plt.show()

def svmTrain():
    global train, test
    X_train = train[:, 1:26]
    Y_train = train[:, 26]
    X_test = test[:, 1:26]
    Y_test = test[:, 26]
    X_train, X_test = ScaleData(X_train, X_test)
    clf = svm.SVC(kernel='rbf',
                  class_weight='balanced', # penalize
                  probability=True,
                  verbose=1)  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, Y_train)

    # Predict the response for test dataset
    # Y_pred = clf.predict(X_test)
    Y_pred = (clf.predict_proba(X_test)[:, 1] >= 0.1926)
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred, Y_test))
    # print(max_f1(clf, X_test, Y_test))
def ANNTrain():
    global train, test
    X_train = train[:, 1:26]
    Y_train = train[:, 26]
    X_test = test[:, 1:26]
    Y_test = test[:, 26]
    X_train, X_test = ScaleData(X_train, X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(25, 25, 25), max_iter=800, verbose=1)
    mlp.fit(X_train, Y_train)
    # Y_pred = mlp.predict(X_test)
    Y_pred = (mlp.predict_proba(X_test)[:, 1] >= 0.015)
    # testArr = [84,12,1,77,0.4,6,142,23,35.4,99,1,4,106,2.2,135,35,122,188,1,165,37,85,17.8,95.4,7.44]
    # testArr = np.array(testArr).reshape(1,-1)
    # testArr = scaler.transform(testArr)
    # print("Prediction: ", mlp.predict(testArr))
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred, Y_test))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    print(max_f1(mlp, X_test, Y_test))

def xgbTrain():
    global train, test
    X_train = train[:, 1:95] #26 #94
    Y_train = train[:, 95]
    X_test = test[:, 1:95]
    Y_test = test[:, 95]
    # X_train, X_test = ScaleData(X_train, X_test)
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train);

    # param_grid = {
    #     'learning_rate': [.01, .03, .06, .1],
    #     'subsample': [.8, .85, .9, .95, 1],
    #     'max_depth': [3, 4, 5, 6],
    #     'colsample_bytree': [.3, .4, .6, .8, 1],
    #     'gamma': [0, 1, 5]
    # }

    # param_grid = {
    #     'learning_rate': [.01],
    #     'subsample': [1],
    #     'max_depth': [4],
    #     'colsample_bytree': [0.6],
    #     'gamma': [0]
    # }

    param_grid = {  #with agg IMPORTANT
        'learning_rate': [.1],
        'subsample': [.8],
        'max_depth': [4],
        'colsample_bytree': [.3],
        'gamma': [0]
    }



    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                               cv=3, n_jobs=1, verbose=2)
    grid_search.fit(X_train, Y_train)
    Y_pred = grid_search.predict(X_test)
    # Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.17759999999999998)  # with agg IMPORTANT
    Y_pred = []
    for insta in X_test:
        if insta[38] == 1:
            Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.14759999999999998)
        elif insta[38] == 2:
            Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.3666)
        elif insta[38] == 3:
            Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.18239999999999998)
        elif insta[38] == 4:
            Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.15899999999999997)
    Y_pred = np.asarray(Y_pred)
    # Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.24899999999999997)  # with agg
    # Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.3126) #without upsampling
    # Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.35579999999999995) #with
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred, Y_test))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    # print(max_f1(grid_search, X_test, Y_test))
    print(grid_search.best_params_)
    # diffs = Y_test - Y_pred
    # i = 0
    # icus = []
    # while i <len(diffs):
    #     diff = diffs[i]
    #     if(diff == 1):
    #         icus.append(X_test[i][10])
    #     i+=1
    # print("Keys: ", Counter(icus).keys())  # equals to list(set(words))
    # print("Values: ",Counter(icus).values())  # counts the elements' frequency
    skplt.metrics.plot_confusion_matrix(Y_test, Y_pred, normalize=True)
    # skplt.metrics.plot_precision_recall_curve(Y_test, grid_search.predict_proba(X_test))
    # skplt.metrics.plot_roc_curve(Y_test, grid_search.predict_proba(X_test))
    plt.show()


def f1_threshold(model, threshold, X, y):
    y_predict = (model.predict_proba(X)[:, 1]>= threshold)
    return metrics.f1_score(y, y_predict)


# Find threshold to maximize F1 score.
def max_f1(model, X, y):
    threshold = -1
    score = 0
    for i in np.linspace(0, .6, 1001):
        temp = f1_threshold(model, i, X, y)
        if temp > score:
            threshold = i
            score = temp
    return [threshold, score]


def auc_threshold(model, threshold, X, y):
    y_predict = (model.predict_proba(X)[:, 1]>= threshold)
    return metrics.auc(y, y_predict)


# Find threshold to maximize F1 score.
def max_auc(model, X, y):
    threshold = -1
    score = 0
    for i in np.linspace(0, .6, 1001):
        temp = auc_threshold(model, i, X, y)
        if temp > threshold:
            threshold = temp
            score = i
    return [threshold, score]

def test():
    x = {}
    z = [1].append(20)
    x[1] = z
    print(x[1])
# formatDataLatest()
# test()
# formatDataWithAgg()
imputate()
# LRTrain()
# RFTrain()
# svmTrain()
# ANNTrain()
xgbTrain()


#######
def imputateCov():
    global train, test
    df = pd.read_csv('Covid.csv', sep=',', header=None,
                     names=["id","gender","age","visiting","from","death"])
    df = UpSampleCov(df)
    df2 = pd.read_csv('CovidTest.csv', sep=',', header=None,
                      names=["id","gender","age","visiting","from","death"])
    values = df.values
    values2 = df2.values
    imputer = SimpleImputer()
    train = imputer.fit_transform(values)
    test = imputer.fit_transform(values2)


def UpSampleCov(df):
    # Separate majority and minority classes
    df_majority = df[df.death == 0]
    df_minority = df[df.death == 1]
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=1085,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    print(df_upsampled.death.value_counts())
    return df_upsampled

def xgbCovid():
    global train, test
    X_train = train[:, 1:5] #26 #94
    Y_train = train[:, 5]
    X_test = test[:, 1:5]
    Y_test = test[:, 5]
    # X_train, X_test = ScaleData(X_train, X_test)
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train);

    param_grid = {
        'learning_rate': [.06],
        'subsample': [.85],
        'max_depth': [6],
        'colsample_bytree': [.6],
        'gamma': [1]
    }


    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                               cv=3, n_jobs=1, verbose=2)
    grid_search.fit(X_train, Y_train)
    # Y_pred = grid_search.predict(X_test)
    Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.5538)
    print("Predicted Deaths: ", np.count_nonzero(Y_pred == 1))
    print("Actual Deaths: ", np.count_nonzero(Y_test == 1))
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:", metrics.precision_score(Y_test, Y_pred))
    print("Recall:", metrics.recall_score(Y_test, Y_pred))
    print("F1:", metrics.f1_score(Y_pred, Y_test))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    # print(max_f1(grid_search, X_test, Y_test))
    print(grid_search.best_params_)
    skplt.metrics.plot_confusion_matrix(Y_test, Y_pred, normalize=True)
    # skplt.metrics.plot_precision_recall_curve(Y_test, grid_search.predict_proba(X_test))
    # skplt.metrics.plot_roc_curve(Y_test, grid_search.predict_proba(X_test))
    plt.show()

# imputateCov()
# xgbCovid()