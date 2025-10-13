#Performance Dashboard using MLflow logs
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
import mlflow
from mlflow_config import setup_mlflow, get_experiment_runs
import psutil
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for real-time tracking
current_metrics = {
    'total_predictions': 0,
    'avg_processing_time': 0.0,
    'avg_character_accuracy': 0.0,
    'avg_cer': 0.0,
    'predictions_per_minute': 0,
    'system_cpu': 0.0,
    'system_memory': 0.0,
    'active_users': 0
}

# Track connected sessions with timestamps for cleanup
connected_sessions = {}  # {session_id: {'ip': ip, 'timestamp': time}}

# Performance history for trends
performance_history = {
    'timestamps': [],
    'processing_times': [],
    'character_accuracies': [],
    'cer_values': [],
    'cpu_usage': [],
    'memory_usage': []
}

def get_system_metrics():
    """Get current system performance metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            'cpu': cpu_percent,
            'memory': memory.percent,
            'memory_available': memory.available // (1024**3)  # GB
        }
    except:
        return {'cpu': 0, 'memory': 0, 'memory_available': 0}

def get_mlflow_metrics():
    """Get latest metrics from MLflow - focused on CER-based evaluation"""
    try:
        setup_mlflow()
        runs_df = get_experiment_runs()
        
        if runs_df is not None and not runs_df.empty:
            # Get recent runs (last 50 runs)
            recent_runs = runs_df.head(50)
            
            # Processing time metrics
            if 'metrics.processing_time' in recent_runs.columns:
                print("Processing time metrics found")
                processing_times = recent_runs['metrics.processing_time'].dropna()
                print(processing_times)
                avg_time = processing_times.mean() if len(processing_times) > 0 else 0
                print(avg_time)
            elif 'metrics.processing_time_seconds' in recent_runs.columns:
                processing_times = recent_runs['metrics.processing_time_seconds'].dropna()
                avg_time = processing_times.mean() if len(processing_times) > 0 else 0
            else:
                print("Processing time metrics not found")
                avg_time = 0
            
            # CER-based accuracy metrics (from accuracy_evaluator.py)
            character_accuracy = 0.0
            cer = 0.0
            edit_distance = 0.0
            total_characters = 0
            correct_characters = 0
            
            if 'metrics.character_accuracy' in recent_runs.columns:
                char_accuracies = recent_runs['metrics.character_accuracy'].dropna()
                character_accuracy = char_accuracies.mean() if len(char_accuracies) > 0 else 0
            
            if 'metrics.cer' in recent_runs.columns:
                cer_values = recent_runs['metrics.cer'].dropna()
                cer = cer_values.mean() if len(cer_values) > 0 else 0
            
            if 'metrics.edit_distance' in recent_runs.columns:
                edit_distances = recent_runs['metrics.edit_distance'].dropna()
                edit_distance = edit_distances.mean() if len(edit_distances) > 0 else 0
            
            if 'metrics.total_characters' in recent_runs.columns:
                total_chars = recent_runs['metrics.total_characters'].dropna()
                total_characters = total_chars.mean() if len(total_chars) > 0 else 0
            
            if 'metrics.correct_characters' in recent_runs.columns:
                correct_chars = recent_runs['metrics.correct_characters'].dropna()
                correct_characters = correct_chars.mean() if len(correct_chars) > 0 else 0
                
            return {
                'total_runs': len(runs_df),
                'recent_runs': len(recent_runs),
                'avg_processing_time': avg_time,
                'avg_character_accuracy': character_accuracy,
                'avg_cer': cer,
                'avg_edit_distance': edit_distance,
                'avg_total_characters': total_characters,
                'avg_correct_characters': correct_characters,
                'evaluation_type': 'cer_based',
                'reference_source': 'gpt4o_transcription'
            }
    except Exception as e:
        print(f"Error getting MLflow metrics: {e}")
        return {
            'total_runs': 0,
            'recent_runs': 0,
            'avg_processing_time': 0,
            'avg_character_accuracy': 0,
            'avg_cer': 0
        }

def cleanup_stale_sessions():
    """Remove sessions that haven't been active for more than 30 seconds"""
    current_time = time.time()
    stale_sessions = []
    
    for session_id, session_data in connected_sessions.items():
        if current_time - session_data['timestamp'] > 30:  # 30 seconds timeout
            stale_sessions.append(session_id)
    
    for session_id in stale_sessions:
        del connected_sessions[session_id]
        print(f"Cleaned up stale session: {session_id}")
    
    return len(stale_sessions)

def update_metrics():
    """Update metrics every second and emit to clients"""
    while True:
        try:
            # Cleanup stale sessions
            cleaned = cleanup_stale_sessions()
            if cleaned > 0:
                current_metrics['active_users'] = len(connected_sessions)
                print(f"Cleaned {cleaned} stale sessions. Active users: {current_metrics['active_users']}")
            
            # Get system metrics
            system_metrics = get_system_metrics()
            
            # Get MLflow metrics
            mlflow_metrics = get_mlflow_metrics()
            
            # Update global metrics
            current_metrics.update({
                'avg_processing_time': mlflow_metrics['avg_processing_time'],
                'system_cpu': system_metrics['cpu'],
                'system_memory': system_metrics['memory'],
                'total_predictions': mlflow_metrics['total_runs'],
                'avg_character_accuracy': mlflow_metrics['avg_character_accuracy'],
                'avg_cer': mlflow_metrics['avg_cer'],            
                })
            # Add to history (keep last 100 data points)
            now = datetime.now()
            performance_history['timestamps'].append(now.strftime('%H:%M:%S'))
            performance_history['processing_times'].append(mlflow_metrics['avg_processing_time'])
            performance_history['character_accuracies'].append(mlflow_metrics['avg_character_accuracy'])
            performance_history['cer_values'].append(mlflow_metrics['avg_cer'])
            performance_history['cpu_usage'].append(system_metrics['cpu'])
            performance_history['memory_usage'].append(system_metrics['memory'])
            
            # Keep only last 100 points
            max_points = 100
            if len(performance_history['timestamps']) > max_points:
                performance_history['timestamps'] = performance_history['timestamps'][-max_points:]
                performance_history['processing_times'] = performance_history['processing_times'][-max_points:]
                performance_history['character_accuracies'] = performance_history['character_accuracies'][-max_points:]
                performance_history['cer_values'] = performance_history['cer_values'][-max_points:]
                performance_history['cpu_usage'] = performance_history['cpu_usage'][-max_points:]
                performance_history['memory_usage'] = performance_history['memory_usage'][-max_points:]
            
            # Emit to all connected clients
            socketio.emit('metrics_update', {
                'current_metrics': current_metrics,
                'performance_history': performance_history
            })
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            print(f"Error in metrics update: {e}")
            time.sleep(5)  # Wait longer on error

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for current metrics"""
    return jsonify({
        'current_metrics': current_metrics,
        'performance_history': performance_history
    })

@app.route('/api/recent-predictions')
def get_recent_predictions():
    """Get recent prediction results with CER-based metrics"""
    try:
        setup_mlflow()
        runs_df = get_experiment_runs()
        
        if runs_df is not None and not runs_df.empty:
            recent_runs = runs_df.head(50) 
            
            predictions = []
            for idx, run in recent_runs.iterrows():
                predictions.append({
                    'run_id': run.get('run_id', 'N/A'),
                    'timestamp': run.get('tags.timestamp', 'N/A'),
                    'processing_time': run.get('metrics.processing_time', run.get('metrics.processing_time_seconds', 'N/A')),
                    'character_accuracy': run.get('metrics.character_accuracy', 'N/A'),
                    'cer': run.get('metrics.cer', 'N/A'),
                    'edit_distance': run.get('metrics.edit_distance', 'N/A'),
                    'total_characters': run.get('metrics.total_characters', 'N/A'),
                    'correct_characters': run.get('metrics.correct_characters', 'N/A'),
                    'draft_length': run.get('metrics.draft_text_length', 'N/A'),
                    'corrected_length': run.get('metrics.corrected_text_length', 'N/A'),
                    'reference_source': run.get('tags.reference_source', 'N/A')
                })
            
            return jsonify(predictions)
        else:
            return jsonify([])
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/evaluation-summary')
def get_evaluation_summary():
    """Get comprehensive evaluation summary based on CER metrics"""
    try:
        setup_mlflow()
        runs_df = get_experiment_runs()
        
        if runs_df is not None and not runs_df.empty:
            recent_runs = runs_df.head(100)  # Last 100 runs for better statistics
            
            # Extract CER-based metrics
            character_accuracies = recent_runs['metrics.character_accuracy'].dropna() if 'metrics.character_accuracy' in recent_runs.columns else []
            cer_values = recent_runs['metrics.cer'].dropna() if 'metrics.cer' in recent_runs.columns else []
            edit_distances = recent_runs['metrics.edit_distance'].dropna() if 'metrics.edit_distance' in recent_runs.columns else []
            
            # Calculate summary statistics
            summary = {
                'total_evaluations': len(recent_runs),
                'avg_character_accuracy': character_accuracies.mean() if len(character_accuracies) > 0 else 0,
                'avg_cer': cer_values.mean() if len(cer_values) > 0 else 0,
                'avg_edit_distance': edit_distances.mean() if len(edit_distances) > 0 else 0,
                'min_character_accuracy': character_accuracies.min() if len(character_accuracies) > 0 else 0,
                'max_character_accuracy': character_accuracies.max() if len(character_accuracies) > 0 else 0,
                'min_cer': cer_values.min() if len(cer_values) > 0 else 0,
                'max_cer': cer_values.max() if len(cer_values) > 0 else 0
            }
            
            # CER distribution (quality categories)
            if len(cer_values) > 0:
                cer_distribution = {
                    'excellent_0_5': len([c for c in cer_values if c <= 0.05]),  # 0-5% CER
                    'very_good_5_10': len([c for c in cer_values if 0.05 < c <= 0.10]),  # 5-10% CER
                    'good_10_20': len([c for c in cer_values if 0.10 < c <= 0.20]),  # 10-20% CER
                    'fair_20_30': len([c for c in cer_values if 0.20 < c <= 0.30]),  # 20-30% CER
                    'poor_above_30': len([c for c in cer_values if c > 0.30])  # >30% CER
                }
                summary['cer_distribution'] = cer_distribution
            
            return jsonify(summary)
        else:
            return jsonify({'error': 'No evaluation data available'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection with IP-based deduplication"""
    session_id = request.sid
    client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
    current_time = time.time()
    
    # Check if this IP already has an active session
    existing_sessions = [sid for sid, data in connected_sessions.items() 
                        if data['ip'] == client_ip and current_time - data['timestamp'] < 30]
    
    if existing_sessions:
        # Remove old sessions from same IP
        for old_sid in existing_sessions:
            if old_sid in connected_sessions:
                del connected_sessions[old_sid]
                print(f"Removed old session {old_sid} for IP {client_ip}")
    
    # Add new session
    connected_sessions[session_id] = {
        'ip': client_ip,
        'timestamp': current_time
    }
    
    current_metrics['active_users'] = len(connected_sessions)
    print(f"Client connected: {session_id} from {client_ip}, Total users: {current_metrics['active_users']}")
    
    emit('metrics_update', {
        'current_metrics': current_metrics,
        'performance_history': performance_history
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    
    if session_id in connected_sessions:
        del connected_sessions[session_id]
        current_metrics['active_users'] = len(connected_sessions)
        print(f"Client disconnected: {session_id}, Total users: {current_metrics['active_users']}")
    else:
        print(f"Disconnect from unknown session: {session_id}")
    
    # Emit updated metrics to remaining clients
    emit('metrics_update', {
        'current_metrics': current_metrics,
        'performance_history': performance_history
    }, broadcast=True)

def start_metrics_thread():
    """Start the metrics update thread"""
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()

if __name__ == '__main__':
    # print("🚀 Starting Real-time Performance Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5002")
    # print("🔌 WebSocket updates enabled for real-time metrics")
    # print("📈 Displaying CER-based character accuracy metrics")
    
    # Start metrics update thread
    start_metrics_thread()
    
    # Start the dashboard
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)
