from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import time
import base64
import json
import subprocess
import threading
import webview  # PyWebview for desktop window
import logging
import cv2  # For webcam access

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Necessary for session management

# Global variables
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

# Main index route - no login needed anymore
@app.route('/')
def index():
    try:
        # Initialize the webcam directly
        print("Starting starter_script")
        subprocess.run(['python', './starter_script.py'])
        print("rendering Index")
        
        return render_template('index.html')
    except Exception as e:
        print(e)
        logger.error(f"Error loading index page: {e}")
        return "An error occurred loading the index page.", 500

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


# Camera control APIs - simplified for local webcam
@app.route('/api/nexttarget')
def nexttarget():
    try:
        # This would need to be implemented differently for physical target systems
        # For a webcam demo, we might just display a new target image
        flash('Next target requested - please place a new target in view', 'info')
        return "Next target requested", 200
    except Exception as e:
        logger.error(f"Error in /api/nexttarget: {e}")
        return "An error occurred while moving to the next target.", 500


@app.route('/api/rifle')
def rifle():
    try:
        # Simplified mode selection for local webcam
        return "Rifle mode selected", 200
    except Exception as e:
        logger.error(f"Error in /api/rifle: {e}")
        return "An error occurred while requesting rifle data.", 500


@app.route('/api/pistol')
def pistol():
    try:
        # Simplified mode selection for local webcam
        return "Pistol mode selected", 200
    except Exception as e:
        logger.error(f"Error in /api/pistol: {e}")
        return "An error occurred while requesting pistol data.", 500


# Function to start Flask server in a separate thread
def start_flask():
    try:
        app.run(debug=False, port=5000, host="127.0.0.1")  # Only listen on localhost
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")


# Start Flask app in a separate thread
flask_thread = threading.Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

# Create a PyWebview window to display the Flask app
try:
    print("starting webview")
    webview.create_window('Target', 'http://127.0.0.1:5000/', width=1000, height=800)
    webview.start()
except Exception as e:
    logger.error(f"Error creating PyWebview window: {e}")