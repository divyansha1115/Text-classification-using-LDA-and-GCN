def model_w2v(data_file):
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.svm import SVC
    import pandas as pd
    df = pd.read_csv(data_file)
    # Create a label (category) encoder object
    le = preprocessing.LabelEncoder()
    # Fit the encoder to the pandas column
    le.fit(df['1'])
    df['encoded_label'] = le.transform(df['1']) 
    y = df.iloc[:, 53].values
    x = df.iloc[:, 3:53].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)
    return svm_model_linear.score(x_test, y_test)

model_w2v('w2v_model_output.csv')