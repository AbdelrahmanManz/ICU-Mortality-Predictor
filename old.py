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