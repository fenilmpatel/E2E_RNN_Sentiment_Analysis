from app import app as application
import os

port = os.environ.get('PORT',5000)

if __name__ == '__main__':
    
    app.run(debug=1,host='0.0.0.0',port=port)