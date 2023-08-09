from flask import Flask, request, render_template, redirect


app = Flask( __name__  )

@app.route('/')
def index():
    return 'hello flask'

@app.route('/a')
def atest():
    return '<h1>korea</h1>'

@app.route('/b')
def btest():
    return '<h1>test</h1>'

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/myform')
def myform():
    return render_template('myform.html')

@app.route('/formresult')
def formresult():
    myname = request.args['myname']
    myage = request.args['myage']
    return render_template('formresult.html',myname=myname, myage=myage)
    # return f'<h1>이름:{myname} 나이:{myage}</h1>'

@app.route('/arr')
def arr():
    return render_template('arr.html', mylist=['사과','딸기','포도','수박'] )

@app.route('/hap')
def hap():
    return render_template('hap.html')

@app.route('/hapresult')
def hapresult():
    one=request.args['one']
    two=request.args['two']
    result = int(one) + int(two)
    return  render_template('hapresult.html', result=result )

@app.route('/imgselect')
def imageselect():
    return render_template('imgselect.html')

if __name__ == '__main__':
    app.run( host='0.0.0.0', port =4500, debug=True)