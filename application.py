import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
import re
from flask import Flask,redirect,url_for,flash,render_template,request,session
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy

#Flask object instantiation
app=Flask(__name__)
db=SQLAlchemy(app)
app.secret_key='ztyp1x-1234'
app.permanent_session_lifetime=timedelta(minutes=100)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///usersDB.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False


class users(db.Model) :
    slno=db.Column("slno",db.Integer,primary_key=True)
    name=db.Column("name",db.String(100))
    email=db.Column("email",db.String(100))
    def __init__(self,name,email):
        self.name=name
        self.email=email

#decorator to map URL function
@app.route('/')
def home():
    #render_template() renders the template
    return render_template('home.html')

@app.route('/admin')
def admin():
    if "user" not in session :
        return redirect(url_for("login"))
    if session["user"]=="admin" :
        return render_template("admin.html",values=users.query.all())
    else :
        flash("Not Authorized!")
        return redirect(url_for("user"))

@app.route('/handle_user',methods=["POST"])     #only POST method is allowed
def handleUser():
    if request.method=="POST":
        name=request.form["user_name"]
        email=request.form["email"]
        found_users=users.query.filter_by(name=name).first()
        if found_users==None:
            user_object=users(name,email)
            db.session.add(user_object)
            db.session.commit()
            flash("Saved into db!")
        session["user"]=name
        session["email"]=email
        session.permanent=True
        return redirect(url_for("user"))
    else :
        return redirect(url_for("home"))

@app.route('/user')
def user():
    if "user" in session :
        return render_template("user.html",name=session["user"])    
    else :
        flash("Not Signed In!")
        return redirect(url_for("loginUser"))

@app.route('/login')
def loginUser():
    if "user" in session:
        flash("Already Signed In!")
        return redirect(url_for("user"))
    return render_template("login.html")

@app.route('/logout')
def logoutUser():
    if "user" in session :
        temp=session["user"]
        session.pop("user")
        flash("{} Signed Out".format(temp))
        return redirect(url_for("home"))
    else :
        flash("Not Signed In!")
        return redirect(url_for("loginUser"))

@app.route('/predict_input')
def inputForm():
    if "user" not in session :
        flash("Not Signed In!")
        return redirect(url_for("user"))
    else :
        return render_template("predict.html")

@app.route('/predict',methods=['POST'])
def predict():
    if "user" not in session:
        flash("Not Signed In!")
        return redirect(url_for("login"))    
    '''
    df1=pd.read_csv('data.csv')
    #print(df1.head(),'\n',df1.columns)


    #preprocessing

    df1.loc[df1['label'] == 'Fake', ['label']] = 'FAKE'
    df1.loc[df1['label'] == 'fake', ['label']] = 'FAKE'
    df1.loc[df1['source'] == 'facebook', ['source']] = 'Facebook'

    df1.loc[5]['label'] = 'FAKE'
    df1.loc[15]['label'] = 'TRUE'
    df1.loc[43]['label'] = 'FAKE'
    df1.loc[131]['label'] = 'TRUE'
    df1.loc[242]['label'] = 'FAKE'

    df1.title.fillna('/', inplace=True)
    df1.text.fillna('/', inplace=True)
    df1.source.fillna('/', inplace=True)
    df1.label.fillna('/', inplace=True)

    df1 = df1[df1['label']!='/']
    df1 = df1[df1['title']!='/']
    df1 = df1[df1['source']!='/']
    df1 = df1[df1['text']!='/']

    df1['title_text'] = df1['title'] + ' ' + df1['text']
    df1.title_text.fillna('/', inplace=True)
    df1 = df1[df1['title_text']!='/']
    df1=df1[['title_text','label']]
    #print(df1['label'].value_counts())



    #preprocessing data
    for i in range(len(df1.index)):
        temp=df1.iloc[i].title_text
        temp=re.sub('[^a-zA-Z]',' ',temp)
        temp=(temp.lower()).split()
        temp=[word for word in temp if (word not in set(stopwords.words('english')))]
        temp=' '.join(temp)
        #print(text)
        df1.iloc[i].title_text=temp


    encoder=LabelEncoder()
    df1['label']=encoder.fit_transform(df1['label'])



    vectorizer=CountVectorizer(max_features=1500,lowercase=True)
    x=df1.iloc[:,0]
    y=df1.iloc[:,1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
    x_train_transform=vectorizer.fit_transform(x_train).toarray()
    x_test=vectorizer.transform(x_test).toarray()
    pickle.dump(vectorizer,open('vectorizer_saved.pickle','wb'))
    
    model1=MultinomialNB()
    model1.fit(x_train_transform,y_train)



    #saving model
    saved_model = open('model2.sav', 'wb')
    pickle.dump(model1, saved_model)
    saved_model.close()


    #visualizing results
    cm = confusion_matrix(y_test,model1.predict(x_test))
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues,xticklabels=['FAKE', 'TRUE'], yticklabels=['FAKE', 'TRUE'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    '''
    #loading saved model
    filename = 'model2.sav'
    saved_clf = pickle.load(open(filename, 'rb'))
    saved_vectorizer=pickle.load(open('vectorizer_saved.pickle','rb'))
    
    #print(saved_clf.score(x_test, y_test))

    #POST method transports the form data to the server in the message body
    if request.method=='POST':
        text=request.form['message']
        #preprocessing input
        temp=re.sub('[^a-zA-Z]',' ',text)
        temp=(temp.lower()).split()
        temp=[word for word in temp if (word not in set(stopwords.words('english')))]
        temp=' '.join(temp)
        ls=[temp]
        val=saved_vectorizer.transform(ls)
        res=saved_clf.predict(val)
        print("RES : ",res)
    return render_template('result.html',prediction=res)

#run() makes sure to run only app.py on the server when this script is executed by the Python interpreter

if __name__=='__main__':
    db.create_all()
    #debug==True activates the Flask debugger and provides detailed error messages
    app.run(debug=True)

