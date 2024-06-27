
def get_test_results(model, x, y):

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(x)
    y_fault = y[160:]
    y_pred_fault = y_pred[160:]
    y_normal = y[:160]
    y_pred_normal = y_pred[:160]

    FAR = 1 - accuracy_score(y_pred_normal, y_normal)
    MDR = 1 - accuracy_score(y_pred_fault, y_fault )
    