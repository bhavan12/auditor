from io import BytesIO
from flask import Flask, render_template, send_file, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import pandas as pd
import xgboost as xgb
con = pyodbc.connect(
"DRIVER={SQL Server};server=10.10.10.3;database=AizantIT_ML;uid=rnd;pwd=AizantIT123")
from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import pygal
plt.style.use('ggplot')
app = Flask(__name__)
Bootstrap(app)
@app.route('/donut_pie_cha/')
def donut_pie_cha():

    sql = """select * from [QMS].[auditorall]"""
    df = pd.io.sql.read_sql(sql, con)
    d=np.array(df)
    motality = d[:,1]
    year = d[:,0:1]
    fig, ax = plt.subplots()
    labels = np.array(year)
    y_pos = np.arange(len(labels))
    x_pos = np.array(motality)
    ax.bar(y_pos, motality, color=['r', 'g', 'y', 'b'], align='center', edgecolor='green')
    plt.xticks(y_pos, labels)
    plt.ylabel('critical percentage', fontsize=20)
    plt.xlabel('auditorid', fontsize=20)
    ax.set_title('overall critical percentage', fontsize=24)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')
@app.route('/bar_grap/')
def bar_grap():

    sql = """select * from [QMS].[siteall]"""
    df = pd.io.sql.read_sql(sql, con)
    d=np.array(df)
    motality = d[:,1]
    year = d[:,0:1]
    fig, ax = plt.subplots()
    labels = np.array(year)
    y_pos = np.arange(len(labels))
    x_pos = np.array(motality)
    ax.bar(y_pos, motality, color=['r', 'g', 'y', 'b'], align='center', edgecolor='green')
    plt.xticks(y_pos, labels)
    plt.ylabel('critical percentage', fontsize=20)
    plt.xlabel('siteId', fontsize=20)
    ax.set_title('overall critical percentage', fontsize=24)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        userDetails = request.form
        num1= userDetails['num1']
        num2 = userDetails['num2']
        num3 = userDetails['num3']
        num4 = userDetails['num4']
        num1=int(num1)
        num2 = int(num2)
        num3 = int(num3)
        num4 = int(num4)
        #print(type(num1))
        sql = """select * from [QMS].[aud2]"""
        df = pd.io.sql.read_sql(sql, con)
        DepartmentID = df['DeptID']
        AuditorID = df['AuditorID']
        SiteID = df['SiteID']
        noofobser = df['noofaber']
        cper = df['cper']
        X = np.array([AuditorID, SiteID, DepartmentID, noofobser]).T
        y = np.array(cper)
        # Split dataset into training set and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

        # Create adaboost classifer object
        abc = AdaBoostRegressor(n_estimators=50,
                                learning_rate=1)
        # Train Adaboost Classifer
        model = abc.fit(X_train, y_train)
        z = np.array([num1,num2,num3,num4]).reshape(1, -1)
        # Predict the response for test dataset
        y_pred = model.predict(z)
        y_pred=int(y_pred)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        #print(np.sqrt(mean_squared_error(y_test, y_pred)))
        '''cur = con.cursor()
        sqlcommand = ("INSERT INTO [QMS].[aud2](AuditorId,SiteID,DeptID,noofaber,cper) VALUES (?,?,?,?,?)")
        values = [num1,num2,num3,num4,y_pred]
        cur.execute(sqlcommand, values)
        con.commit()
        cur.close()'''
        # Visualizing the training Test Results
        '''plt.scatter(xtrain, ytrain, color='red')
        plt.plot(xtrain, regressor.predict(xtrain), color='blue')
        plt.title("Visuals for Training Dataset")
        plt.xlabel("Space")
        plt.ylabel("Price")
        #plt.show()
        # Visualizing the Test Results
        plt.scatter(xtest, ytest, color='red')
        plt.plot(xtrain, regressor.predict(xtrain), color='blue')
        plt.title("Visuals for Test DataSet")
        plt.xlabel("Space")
        plt.ylabel("Price")
        #plt.show()'''
        '''x = np.array(AuditorID)
        x=np.append(x,num1)
        x1 = np.array(SiteID)
        x1 = np.append(x, num2)
        x2= np.array(DepartmentID)
        x2 = np.append(x, num3)
        x3 = np.array(noofobser)
        x3= np.append(x, num4)

        y1 = np.array(cper)
        y2=np.append(y,y_pred)'''
        #z=zip(num1,num2,num3,num4,y_pred)
        return render_template('index.html',a=num1,b=num2,c=num3,d=num4,e=y_pred)
    else:
        sql = """select * from [QMS].[aud2]"""
        df = pd.io.sql.read_sql(sql, con)
        DepartmentID = df['DeptID']
        AuditorID = df['AuditorID']
        SiteID = df['SiteID']
        noofobser = df['noofaber']
        cper = df['cper']
        X = np.array([AuditorID, SiteID, DepartmentID, noofobser]).T
        y = np.array(cper)
        # Split dataset into training set and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

        # Create adaboost classifer object
        abc = AdaBoostRegressor(n_estimators=50,
                                learning_rate=1)
        # Train Adaboost Classifer
        model = abc.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = model.predict(X_test)
        x = np.array(AuditorID)
        #x = np.append(x, num1)
        x1 = np.array(SiteID)
        #x1 = np.append(x, num2)
        x2 = np.array(DepartmentID)
        #x2 = np.append(x, num3)
        x3 = np.array(noofobser)
        #x3 = np.append(x, num4)

        y1 = np.array(cper)
        y2 = np.append(y, y_pred)
        z = zip(x, x1, x2, x3, y)
        return render_template('index.html', x=x, y=y, z=z)

        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        #print(np.sqrt(mean_squared_error(y_test, y_pred)))
@app.route('/delete')
def user():
    print("")
if __name__ == '__main__':
   app.run(debug=True)

