# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.17759999999999998)  # with agg IMPORTANT
# Y_pred = []
# for insta in X_test:
#     if insta[38] == 1:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.14759999999999998)
#     elif insta[38] == 2:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.3666)
#     elif insta[38] == 3:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.18239999999999998)
#     elif insta[38] == 4:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.15899999999999997)
# Y_pred = np.asarray(Y_pred)
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.24899999999999997)  # with agg
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.3126) #without upsampling
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.35579999999999995) #with

#f1
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.17759999999999998)  # with agg IMPORTANT
# Y_pred = []
# for insta in X_test:
#     if insta[38] == 1:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.20879999999999999)
#     elif insta[38] == 2:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.2178)
#     elif insta[38] == 3:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.252)
#     elif insta[38] == 4:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.21)
# Y_pred = np.asarray(Y_pred)
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.24899999999999997)  # with agg
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.3126) #without upsampling
# Y_pred = (grid_search.predict_proba(X_test)[:, 1] >= 0.35579999999999995) #with

#comp
# Y_pred = []
# for insta in X_test:
#     if insta[38] == 1:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.27059999999999995)
#     elif insta[38] == 2:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.28619999999999995)
#     elif insta[38] == 3:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.3036)
#     elif insta[38] == 4:
#         Y_pred.append(grid_search.predict_proba(insta.reshape(1, -1))[:, 1] >= 0.2724)
# Y_pred = np.asarray(Y_pred)

import os

directory = 'set-b'
def removeLast():
    variables = ['RecordID', 'Age', 'BUN', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
       'HCT', 'HR', 'ICUType', 'K', 'MAP', 'Mg',
       'Na', 'PaCO2', 'PaO2', 'Platelets','Gender',
        'SysABP', 'Temp', 'Urine', 'WBC', 'Weight', 'pH']
    # variables = ['BUN']
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            for variable in variables:
                varInstances = []
                with open(directory + "/" + filename, mode='r') as f:
                    for num, line in enumerate(f, 1):
                        if(variable in line):
                            varInstances.append(num)

                if(len(varInstances) > 1):
                    finalStr = ""
                    with open(directory + "/" + filename, mode='r') as f:
                        for pos, line in enumerate(f, 1):
                            if pos != varInstances[-1]:
                                finalStr += line
                    with open(directory + "/" + filename, mode='w') as f:
                        f.write(finalStr)

removeLast()