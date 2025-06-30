#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import websocket
import threading
import json
import base64
import time
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# ASR Server URLs
VI_ASR_URL = "ws://localhost:9432/voice/api/asr/v1/ws/decode_online?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int)16000"
EN_ASR_URL = "ws://localhost:9433/voice/api/asr/v1/ws/decode_online?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int)16000"

active_connections = {}
ws_to_session = {}  # Maps websocket objects to session IDs

def on_vi_message(ws, message):
    try:
        data = json.loads(message)
        
        if not 'result' in data:
            return
        
        # Get the session_id for this websocket
        session_id = ws_to_session.get(ws)
        if not session_id:
            print("Warning: Received VI ASR result for unknown session")
            return
            
        if data['result']['final']:
            transcript = data["result"]["hypotheses"][0]["transcript_normalized"]
            response = {
                'type': 'vi',
                'text': transcript,
                'isFinal': True
            }
            # Emit only to the specific client
            socketio.emit('asr_result', response, room=session_id)
        else:
            transcript = data["result"]["hypotheses"][0]["transcript"]
            response = {
                'type': 'vi',
                'text': transcript,
                'isFinal': False
            }
            # Emit only to the specific client
            socketio.emit('asr_result', response, room=session_id)
    except Exception as e:
        print(f"Error processing Vietnamese ASR message: {e}")
        
def on_en_message(ws, message):
    try:
        data = json.loads(message)
        
        if not 'result' in data:
            return
            
        # Get the session_id for this websocket
        session_id = ws_to_session.get(ws)
        if not session_id:
            print("Warning: Received EN ASR result for unknown session")
            return
            
        if data['result']['final']:
            transcript = data["result"]["hypotheses"][0]["transcript"]
            response = {
                'type': 'en',
                'text': transcript,
                'isFinal': True
            }
            # Emit only to the specific client
            socketio.emit('asr_result', response, room=session_id)
        else:
            transcript = data["result"]["hypotheses"][0]["transcript"]
            response = {
                'type': 'en',
                'text': transcript,
                'isFinal': False
            }
            # Emit only to the specific client
            socketio.emit('asr_result', response, room=session_id)
    except Exception as e:
        print(f"Error processing English ASR message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    # Remove from the mapping when closed
    if ws in ws_to_session:
        del ws_to_session[ws]

def on_vi_open(ws):
    print("Vietnamese ASR connection opened")

def on_en_open(ws):
    print("English ASR connection opened")

def create_asr_connection(session_id):
    if session_id in active_connections:
        return
        
    # Create Vietnamese ASR connection
    vi_ws = websocket.WebSocketApp(
        VI_ASR_URL,
        on_message=on_vi_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_vi_open
    )
    
    # Create English ASR connection
    en_ws = websocket.WebSocketApp(
        EN_ASR_URL,
        on_message=on_en_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_en_open
    )
    
    # Map websockets to this session
    ws_to_session[vi_ws] = session_id
    ws_to_session[en_ws] = session_id
    
    # Start WebSocket connections in background threads
    vi_thread = threading.Thread(target=vi_ws.run_forever)
    vi_thread.daemon = True
    vi_thread.start()
    
    en_thread = threading.Thread(target=en_ws.run_forever)
    en_thread.daemon = True
    en_thread.start()
    
    # Store connections
    active_connections[session_id] = {
        'vi_ws': vi_ws,
        'en_ws': en_ws,
        'vi_thread': vi_thread,
        'en_thread': en_thread
    }
    
    # Wait for connections to establish
    time.sleep(1)

def close_asr_connection(session_id):
    if session_id not in active_connections:
        return
        
    connections = active_connections[session_id]
    
    # Remove from the mapping
    if 'vi_ws' in connections and connections['vi_ws'] in ws_to_session:
        del ws_to_session[connections['vi_ws']]
    
    if 'en_ws' in connections and connections['en_ws'] in ws_to_session:
        del ws_to_session[connections['en_ws']]
    
    # Close connections
    if 'vi_ws' in connections:
        connections['vi_ws'].close()
    
    if 'en_ws' in connections:
        connections['en_ws'].close()
    
    # Remove from active connections
    del active_connections[session_id]
    print(f"Closed connections for session {session_id}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    print(f"Client connected: {session_id}")
    create_asr_connection(session_id)

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    print(f"Client disconnected: {session_id}")
    close_asr_connection(session_id)

@socketio.on('audio_data')
def handle_audio_data(data):
    session_id = request.sid
    
    if session_id not in active_connections:
        return
    
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(data['audio'])
        
        # Send to both ASR services
        connections = active_connections[session_id]
        if 'vi_ws' in connections:
            connections['vi_ws'].send(audio_data, websocket.ABNF.OPCODE_BINARY)
        
        if 'en_ws' in connections:
            connections['en_ws'].send(audio_data, websocket.ABNF.OPCODE_BINARY)
    except Exception as e:
        print(f"Error processing audio data: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)