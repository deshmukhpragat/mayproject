from flask import Flask, jsonify, request
import pickle

app=Flask(__name__)

model=pickle.load(open('iris_reg.pkl','rb'))

@app.route('/')
def status():
    return jsonify({'massage':'status active'})

@app.route('/predict_sepal_length',methods=['POST'])
def sepal_length():
    data=request.get_json()
    print(data)
    SepalWidthCm=data['SepalWidthCm']
    print('SepalWidthCm : ',SepalWidthCm)
    PetalLengthCm=data['PetalLengthCm']
    print('PetalLengthCm:',PetalLengthCm)
    PetalWidthCm=data['PetalWidthCm']
    print('PetalWidthCm : ',PetalWidthCm)
    Species=data['Species']
    print('Species : ',Species)

    if Species=='Iris-setosa':
        Species=0
    elif Species=='Iris-versicolor':
        Species=1
    elif Species=='Iris-verginica':
        Species=2

    print('encoded species : ',Species)

    test_array=[SepalWidthCm,PetalLengthCm,PetalWidthCm,Species]
    prediction = model.predict([test_array])

    return jsonify ({'predicted sepal length is':prediction[0]})

if __name__=='__main__':
    app.run(debug=True)
