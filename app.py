from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import time
import base64
import json
import subprocess
import requests
import threading
import webview  # PyWebview for desktop window
import logging


app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Necessary for session management


# Global variables to store the latest image and score
device_ips = {
    '0': "10.0.0.32",
    '1': "192.168.1.1",
    '2': "192.168.1.2",
    '3': "192.168.1.3",
    '4': "192.168.1.4",
    '5': "192.168.1.5",
    '6': "192.168.1.6",
    '7': "192.168.1.7",
    '8': "192.168.1.8",
    '9': "192.168.1.9",
    '16': "192.168.1.19"
    # Add more as needed
}

selected_ip = None
latest_image = None
latest_score = 0
latest_angles_dict = None
last_update_time = None
script_process = None

# Set up the logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='app.log',  # Log to a file named app.log
                    filemode='w')  # Write mode, change to 'a' for append mode

logger = logging.getLogger(__name__)


# def calculateTotal(angles):
#     total = 0
#     for points in angles.keys():
#         total += int(points) * len(angles[points])
#     return total


# Route for handling login
@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            device_id = request.form['device_id']

            # Check if the device ID exists in the list
            if device_id in device_ips:
                global selected_ip
                selected_ip = device_ips[device_id]
                session['logged_in'] = True  # Store login state in the session
                flash('Device selected successfully!', 'success')
                print("redirecting to index")
                return redirect(url_for('index'))  # Redirect to the index page upon successful login
            else:
                flash('Invalid device ID. Please try again.', 'error')
                logger.warning(f"Login attempt failed for device ID: {device_id}")

        return render_template('login.html')  # Render the login form
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return "An error occurred during login.", 500


# Route for handling logout
@app.route('/logout')
def logout():
    try:
        session.pop('logged_in', None)  # Log out the user
        flash('Logged out successfully.', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        return "An error occurred during logout.", 500


# Main index route
@app.route('/')
def index():
    try:
        if not session.get('logged_in'):
            print("redirecting to login")
            logger.info("redirecting to login")
            return redirect(url_for('login'))  # Redirect to login page if not logged in
        
        print("Starting starter_script")
        subprocess.run(['python', './starter_script.py'])
        print("rendering Index")
        
        return render_template('index.html', ip_address=selected_ip)  # Pass selected_ip to the template
    except Exception as e:
        print(e)
        logger.error(f"Error loading index page: {e}")
        return "An error occurred loading the index page.", 500


# # Route for receiving score data
# @app.route('/api/score', methods=['POST'])
# def receive_score():
#     try:
#         global latest_image, latest_angles_dict, last_update_time, latest_score
#
#         image_file = request.files['image']
#         latest_image = image_file.read()
#         last_update_time = time.time()
#         latest_angles_dict = json.loads(request.form['angles'])
#         print(latest_angles_dict)
#
#         if latest_angles_dict is not None:
#             latest_score = calculateTotal(latest_angles_dict)
#
#         return 'Data received successfully', 200
#     except Exception as e:
#         logger.error(f"Error in /api/score: {e}")
#         return "An error occurred while receiving score data.", 500
#

# Route to get the currently selected IP address
@app.route('/api/selected_ip', methods=['GET'])
def get_selected_ip():
    try:
        global selected_ip
        if selected_ip:
            return jsonify({'selected_ip': selected_ip}), 200
        else:
            return jsonify({'error': 'No IP selected'}), 404
    except Exception as e:
        logger.error(f"Error in /api/selected_ip: {e}")
        return "An error occurred while retrieving the selected IP address.", 500






@app.route('/api/starter', methods=['POST'])
def receive_starter():
    global latest_image

    image_file = request.files['image']

    latest_image = image_file.read()

    return 'Data received successfully', 200


@app.route('/api/data')
def get_data():
    try:
        global latest_image, latest_angles_dict, last_update_time, latest_score

        return jsonify({
            'image': base64.b64encode(latest_image).decode('utf-8', 'ignore') if latest_image else None,
            # 'angles': latest_angles_dict if latest_angles_dict else None,
            # 'total_score': latest_score if latest_score else None,
            'last_update': last_update_time
        })
    except Exception as e:
        logger.error(f"Error in /api/data: {e}")
        return "An error occurred while retrieving data.", 500


@app.route('/api/start')
def launchScorer():
    global script_process
    try:
        print("launchScorer")
        if script_process is None or script_process.poll() is not None:
            subprocess.run(['python', './score_once.py'])
            script_process = subprocess.Popen(['python', './scoring_script.py'])
            return jsonify({"status": "Script started"}), 200
        else:
            return jsonify({"status": "Script is already running", "pid": script_process.pid}), 400
    except Exception as e:
        logger.error(f"Error starting scorer script: {e}")
        return jsonify({"status": "Error starting script", "error": str(e)}), 500


@app.route('/api/stop')
def terminateScorer():
    global script_process
    try:
        if script_process is not None and script_process.poll() is None:
            script_process.terminate()
            script_process.wait()
            return jsonify({"status": "Script stopped"}), 200
        else:
            return jsonify({"status": "No running script to stop"}), 400
    except Exception as e:
        logger.error(f"Error stopping scorer script: {e}")
        return jsonify({"status": "Error stopping script", "error": str(e)}), 500


@app.route('/api/reset')
def scoreResetter():
    global latest_angles_dict, latest_image, latest_score, script_process
    if script_process is not None and script_process.poll() is None:
        script_process.terminate()
        script_process.wait()
    latest_angles_dict = latest_image = latest_score = None
    return "Reset Complete"


# Various target control routes
@app.route('/api/rifle')
def rifle():
    try:
        response = requests.get(f'http://{selected_ip}:8000/rifle')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/rifle: {e}")
        return "An error occurred while requesting rifle data.", 500


@app.route('/api/pistol')
def pistol():
    try:
        response = requests.get(f'http://{selected_ip}:8000/pistol')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/pistol: {e}")
        return "An error occurred while requesting pistol data.", 500


@app.route('/api/nexttarget')
def nexttarget():
    try:
        response = requests.get(f'http://{selected_ip}:8000/nexttarget')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/nexttarget: {e}")
        return "An error occurred while moving to the next target.", 500


@app.route('/api/focus_increase')
def focus_increase():
    try:
        response = requests.get(f'http://{selected_ip}:8000/focusin')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/focus_increase: {e}")
        return "An error occurred while increasing focus.", 500


@app.route('/api/focus_decrease')
def focus_decrease():
    try:
        response = requests.get(f'http://{selected_ip}:8000/focusout')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/focus_decrease: {e}")
        return "An error occurred while decreasing focus.", 500


@app.route('/api/zoom_increase')
def zoom_increase():
    try:
        response = requests.get(f'http://{selected_ip}:8000/zoomin')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/zoom_increase: {e}")
        return "An error occurred while increasing zoom.", 500


@app.route('/api/zoom_decrease')
def zoom_decrease():
    try:
        response = requests.get(f'http://{selected_ip}:8000/zoomout')
        return response.text
    except Exception as e:
        logger.error(f"Error in /api/zoom_decrease: {e}")
        return "An error occurred while decreasing zoom.", 500


# Function to start Flask server in a separate thread
def start_flask():
    try:
        app.run(debug=False, port=5000, host="0.0.0.0")
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")


# Start Flask app in a separate thread
flask_thread = threading.Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

# Create a PyWebview window to display the Flask app
try:
    print("starting webview")
    webview.create_window('Target', 'http://127.0.0.1:5000/login', width=1000, height=800)
    webview.start()
except Exception as e:
    logger.error(f"Error creating PyWebview window: {e}")
