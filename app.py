from flask import Flask, request, render_template,Response
from flask_cors import cross_origin,CORS
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
CORS(app)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    if request.form is not None:
        avg_rss12 = float(request.form['avg_rss12'])
        var_rss12 = float(request.form['var_rss12'])
        avg_rss13 = float(request.form['avg_rss13'])
        var_rss13 = float(request.form['var_rss13'])
        avg_rss23 = float(request.form['avg_rss23'])
        var_rss23 = float(request.form['var_rss23'])
        std_scalar=StandardScaler()
        to_predict_list=std_scalar.fit_transform([[avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23]])
        #to_predict_list=[avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23]
        model = pickle.load(open('ActivityRecognition.sav', "rb"))
        #to_predict = np.array(to_predict_list).reshape(1,)
        #to=pd.DataFrame(to_predict,columns=list(var))
        result=model.predict(to_predict_list)
        if result==0 or result==1:
            result="bending"
        else:
            if result== 2:
                result="cycling"
            else:
                if result==3:
                    result="lying"
                else:
                    if result==4:
                        result="sitting"
                    else:
                        if result==5:
                            result="standing"
                        else:
                            result="walking"

        return render_template('index.html', prediction_text=result)
if __name__=='__main__':
    app.run(debug=True,threaded=True)

