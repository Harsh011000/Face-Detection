import os
from flask import Flask, request, render_template, send_from_directory
import username

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    uploaded_file = request.files['image']

    if uploaded_file.filename != '':
        # Get the user's name
        user_name = request.form['name']

        # Create a unique filename for the image
        #filename = user_name + '_image.jpg'
        filename="sample.jpg"

        # Save the uploaded image to the server
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        username.usernamefromsite=user_name
        username.printuser()
        return 'Image uploaded and saved as ' + filename
    else:
        return 'No file selected'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/exit', methods=['POST'])
# def exit_app():
#     shutdown_server()
#     return 'Application is closing...', 200
#
# def shutdown_server():
#     app.run(debug=False)
#     # func = request.environ.get('werkzeug.server.shutdown')
#     # if func is None:
#     #     raise RuntimeError('Not running with the Werkzeug Server')
#     # func()

if __name__ == '__main__':
    app.run(debug=True)
