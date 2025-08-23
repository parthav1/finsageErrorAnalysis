import os
import json
import logging
import yaml
import uuid
import datetime
import threading
from threading import Timer
from flask import Flask, request, Response, current_app, render_template, stream_with_context, g
from flask_cors import CORS
from functools import wraps
import sqlite3
import requests
from pathlib import Path

from utils.ragManager import RAGManager
from utils.vllmChatService import ChatService

db_lock = threading.Lock()

def get_db():
    if 'db' not in g:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log', 'feedback.db')
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s when database is locked
    return g.db

# Global response handler remains the same
class GlobalResponseHandler:
    @staticmethod
    def success(data=None, message="Success", status_code=200):
        return GlobalResponseHandler._create_response("success", message, data, status_code)

    @staticmethod
    def error(message="An error occurred", data=None, status_code=400):
        return GlobalResponseHandler._create_response("error", message, data, status_code)

    @staticmethod
    def _create_response(status, message, data, status_code):
        response = {
            "status": status,
            "message": message,
            "data": data
        }
        response_json = json.dumps(response)
        return Response(response=response_json, status=status_code, mimetype='application/json')
        
    @staticmethod
    def stream_response(generate_func):
        return Response(stream_with_context(generate_func()), content_type='text/event-stream')

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Updated decorator that uses current_app instead of the global app instance.
def require_bearer_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return GlobalResponseHandler.error(
                message="Missing Authorization header",
                status_code=401
            )
        if not auth_header.startswith('Bearer '):
            return GlobalResponseHandler.error(
                message="Invalid authorization format. Use 'Bearer <token>'",
                status_code=401
            )
        token = auth_header.split(' ')[1]
        if not token or token != current_app.config['BEARER_TOKEN']:
            return GlobalResponseHandler.error(
                message="Invalid bearer token",
                status_code=401
            )
        return f(*args, **kwargs)
    return decorated

def create_app():
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create the Flask app instance and configure it
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    CORS(app)

    # Load configuration (from an environment variable or a default file)
    config_path = os.getenv('CONFIG_PATH', '../config/production.yaml')
    config = load_config(config_path)
    
    # Set up required configuration values
    app.config['BEARER_TOKEN'] = config.get('bearer_token') or os.getenv('BEARER_TOKEN')
    if not app.config['BEARER_TOKEN']:
        raise ValueError("Bearer token not configured")

    # Setup logging based on configuration
    log_level = config.get('log_level', 'WARNING')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        filename='server.log', 
        filemode='w', 
        level=numeric_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    app.logger = logger  # Attach the logger to the app

    # 添加信号钩子 + atexit 备份 ----------
    import atexit, shutil, signal, sys, datetime as dt
    PROJECT_ROOT = Path(__file__).resolve().parent          # 当前文件所在目录
    LOG_PATH = PROJECT_ROOT / 'server.log'
    BACKUP_DIR = Path('/root/autodl-tmp/server_logs')

    def backup_log():
        for h in logger.handlers:
            h.flush()
        ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            shutil.copy2(LOG_PATH, BACKUP_DIR / f'server{ts}.log')
            print(f"Log backed up to backup/server{ts}.log")
        except Exception as e:
            print(f"Backup failed: {e}")

    atexit.register(backup_log)  # Python 解释器正常退出时兜底

    def graceful_exit(signum, frame):
        logger.warning(f"Received signal {signum}, backing up log before exit")
        backup_log()
        logging.shutdown()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        signal.signal(sig, graceful_exit)
    # ----------代码块结束 ----------



    # Initialize your managers/services
    collections = {'lotus': 10}
    rag_manager = RAGManager(config=config, collections=collections)
    chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk=config['rerank_topk'])
    
    # Set up periodic cleanup of old sessions every 5 minutes
    def schedule_cleanup():
        chat_service.cleanup_old_sessions()
        cleanup_timer = Timer(300, schedule_cleanup)
        cleanup_timer.daemon = True
        cleanup_timer.start()
        logger.info("Scheduled next session cleanup in 5 minutes")
    
    # Start the initial cleanup timer
    cleanup_timer = Timer(300, schedule_cleanup)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info("Initial session cleanup scheduled in 5 minutes")
    # logger.warning("Load ChatService: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    
    # Save instances in the app config for access in your routes
    app.config['rag_manager'] = rag_manager
    app.config['chat_service'] = chat_service
    app.config['cleanup_timer'] = cleanup_timer  # Store the timer reference for potential cleanup

    # Define your routes (using the updated decorator)
    @app.route('/health', methods=['GET'])
    def health_check():
        return GlobalResponseHandler.success(message="Server is running")

    @app.route('/api/check_token', methods=['GET'])
    def check_token():
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return GlobalResponseHandler.error(
                message="Missing Authorization header",
                status_code=401
            )
        if not auth_header.startswith('Bearer '):
            return GlobalResponseHandler.error(
                message="Invalid authorization format. Use 'Bearer <token>'",
                status_code=401
            )
        token = auth_header.split(' ')[1]
        if not token or token != current_app.config['BEARER_TOKEN']:
            return GlobalResponseHandler.error(
                message="Invalid bearer token",
                status_code=401
            )
        return GlobalResponseHandler.success(message="Token is valid")


    @app.route('/api_chat', methods=['POST'])
    @require_bearer_token
    def api_chat():
        try:
            data = request.json
            question = data.get('question')
            session_id = data.get('session_id', str(uuid.uuid4()))
            internal_input = data.get('internal_input', None)
            interrupt_index = data.get('interrupt_index', None)

            if not question:
                return GlobalResponseHandler.error(message="Question not provided")

            # Retrieve the chat service instance from the app config
            chat_service_instance = current_app.config['chat_service']
            response_text, _, _, _, _, _, history = chat_service_instance.generate_response_async(
                question,
                session_id,
                internal_input,
                interrupt_index,
            )

            return GlobalResponseHandler.success(data={
                "response": response_text,
                "session_id": session_id,
                "history": history
            })
        
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api_chat endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))


    @app.route('/api_chat_stream', methods=['POST'])
    @require_bearer_token
    def api_chat_stream():
        try:
            data = request.json
            question = data.get('question')
            session_id = data.get('session_id', str(uuid.uuid4()))
            internal_input = data.get('internal_input', None)
            interrupt_index = data.get('interrupt_index', None)

            if not question:
                return GlobalResponseHandler.error(message="Question not provided")

            # Retrieve the chat service instance from the app config
            chat_service_instance = current_app.config['chat_service']
            
            # Create a wrapper generator to intercept and save the response
            def response_interceptor():
                full_response = ""
                # Generate a unique response_id at the beginning
                response_id = str(uuid.uuid4())
                
                # Get the streaming response generator
                stream_generator = chat_service_instance.generate_response_async_stream(
                    question,
                    session_id,
                    internal_input,
                    interrupt_index,
                )
                
                # Process each chunk
                for chunk in stream_generator:
                    # First chunk handling - modify to include response_id before yielding
                    if full_response == "":
                        try:
                            # Parse the chunk data
                            chunk_data = chunk.replace("data: ", "").strip()
                            chunk_json = json.loads(chunk_data)
                            if "response" in chunk_json:
                                # Add response_id to the first chunk
                                chunk_json["question"] = question
                                chunk_json["response_id"] = response_id
                                # Update the chunk with the response_id
                                chunk = f"data: {json.dumps(chunk_json)}\n\n"
                                # Add to full response
                                full_response += chunk_json["response"]
                        except Exception as e:
                            current_app.logger.error(f"Error modifying first chunk: {str(e)}")
                    else:
                        # For subsequent chunks, just extract content
                        try:
                            # Parse the chunk data (format: "data: {"response": "chunk_text"}\n\n")
                            chunk_data = chunk.replace("data: ", "").strip()
                            chunk_json = json.loads(chunk_data)
                            if "response" in chunk_json:
                                full_response += chunk_json["response"]
                        except Exception as e:
                            current_app.logger.error(f"Error parsing chunk: {str(e)}")
                    
                    # Pass the chunk to the client after any modifications
                    yield chunk
                
                # Save the complete Q&A pair to the database
                try:
                    chat_manager = chat_service_instance.get_or_create_chat_manager(session_id)
                    log = chat_manager.get_runtime_log()
                        
                    with db_lock:
                        db = get_db()
                        cursor = db.cursor()
                        cursor.execute(
                            "INSERT INTO feedback (session_id, response_id, rating, question, response, is_rag, log) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (session_id, response_id, 0, question, full_response, 1, json.dumps(log))
                        )
                        db.commit()
                    current_app.logger.info(f"Saved Q&A pair to database with response_id: {response_id}")
                except Exception as e:
                    current_app.logger.error(f"Error saving Q&A to database: {str(e)}")
            
            # Use the interceptor in the response
            return Response(
                stream_with_context(response_interceptor()),
                content_type='text/event-stream'
            )
        
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api_chat_stream endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))

    @app.route('/test_api_chat')
    def test_api_chat():
        session_id = str(uuid.uuid4())
        # Initialize a chat manager for this session
        chat_service_instance = current_app.config['chat_service']
        _ = chat_service_instance.get_or_create_chat_manager(session_id)
        return render_template('test_api.html', session_id=session_id)

    @app.route('/feedback')
    def feedback():
        try:
            with db_lock:
                db = get_db()
                cursor = db.execute('SELECT session_id, response_id, rating, feedback, question, response, log, user, created_at FROM feedback ORDER BY id DESC')
                feedbacks = cursor.fetchall()
                
            # Convert to list of dictionaries for easier handling in template
            feedback_list = []
            for row in feedbacks:
                feedback_list.append({
                    'session_id': row[0],
                    'time': row[8],
                    'rating': row[2],
                    'feedback': row[3] or '',
                    'question': row[4],
                    'response': row[5],
                    'log': row[6],
                    'user': row[7] or ''
                })
                
            return render_template('feedback.html', feedbacks=feedback_list)
        except Exception as e:
            current_app.logger.error(f"An error occurred in /feedback endpoint: {str(e)}")
            return f"Error: {str(e)}", 500

    @app.route('/api/internal_assistant', methods=['POST'])
    @require_bearer_token
    def internal_assistant():
        try:
            data = request.json
            session_id = data.get('session_id')
            internal_message = data.get('message')
            
            if not session_id:
                return GlobalResponseHandler.error(message="Session ID not provided")
            
            if not internal_message:
                return GlobalResponseHandler.error(message="Internal assistant message not provided")
                
            # Get the chat service instance from the app config
            chat_service_instance = current_app.config['chat_service']
            
            # Get or create chat manager for this session
            chat_manager = chat_service_instance.get_or_create_chat_manager(session_id)
            
            # Add the internal assistant message to the QA history
            chat_manager.add_internal_assitant_message(internal_message)
            
            return GlobalResponseHandler.success(data={
                "session_id": session_id,
                "status": "Internal assistant message added successfully"
            })
            
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api/internal_assistant endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))

    @app.route('/api/log', methods=['GET'])
    @require_bearer_token
    def get_log():
        try:
            session_id = request.args.get('session_id')
            if not session_id:
                return GlobalResponseHandler.error(message="Session ID not provided")
            chat_service_instance = current_app.config['chat_service']
            chat_manager = chat_service_instance.get_or_create_chat_manager(session_id)
            logs = chat_manager.get_runtime_log()
            current_app.logger.info(logs)
            return GlobalResponseHandler.success(data=logs)
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api/log endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))

    @app.route('/api/report_error', methods=['POST'])
    @require_bearer_token
    def report_error():
        try:
            data = request.json
            session_id = data.get('session_id')
            error_message = data.get('error_message')
            
            if not session_id:
                return GlobalResponseHandler.error(message="Session ID not provided")
            
            if not error_message:
                return GlobalResponseHandler.error(message="Error message not provided")
            
            # Get session log
            chat_service_instance = current_app.config['chat_service']
            chat_manager = chat_service_instance.get_or_create_chat_manager(session_id)
            logs = chat_manager.get_runtime_log()
            
            # Create error log directory if it doesn't exist
            error_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log', 'error')
            os.makedirs(error_log_dir, exist_ok=True)
            
            # Create error report with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_report = {
                "timestamp": timestamp,
                "session_id": session_id,
                "error_message": error_message,
                "session_log": logs
            }
            
            # Save error report to file
            error_file_path = os.path.join(error_log_dir, f"{timestamp}.json")
            with open(error_file_path, 'w') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            
            current_app.logger.warning(f"Error report saved to {error_file_path}")
            return GlobalResponseHandler.success(message="Error report submitted successfully")
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api/report_error endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))

    @app.route('/api/submit_rating', methods=['POST'])
    @require_bearer_token
    def submit_rating():
        try:
            data = request.json
            session_id = data.get('session_id')
            response_id = data.get('response_id')
            # rating = data.get('rating')
            feedback = data.get('feedback')
            question = data.get('question')
            response = data.get('response_content')
            user = data.get('user', '')

            rating_raw = data.get('rating')
            try:
                rating = int(rating_raw)
            except (TypeError, ValueError):
                return GlobalResponseHandler.error(message="Invalid rating value")
            
            if not session_id or not response_id or not question or not response:
                return GlobalResponseHandler.error(message="Missing required fields")
            
            # Save the rating and feedback and chat manager runtime log into sqlite database
            chat_service_instance = current_app.config['chat_service']
            chat_manager = chat_service_instance.get_or_create_chat_manager(session_id)
            log = chat_manager.get_runtime_log()

            with db_lock:
                db = get_db()
                # Try to update existing record first, if not found then insert a new one
                cursor = db.execute(
                    'UPDATE feedback SET rating = ?, feedback = ?, question = ?, response = ?, log = ?, user = ? WHERE session_id = ? AND response_id = ?',
                    (rating, feedback, question, response, json.dumps(log), user, session_id, response_id)
                )
                # If no rows were affected by the update, insert a new record
                if cursor.rowcount == 0:
                    db.execute(
                        'INSERT INTO feedback (session_id, response_id, rating, feedback, question, response, log, user) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (session_id, response_id, rating, feedback, question, response, json.dumps(log), user)
                    )
                db.commit()

            if rating <= 2:
                online_answer = handle_low_rating(session_id=session_id,feedback=feedback,question=question,response=response)  
            else:
                online_answer = None     

            return GlobalResponseHandler.success(
                data={
                    "online_answer": online_answer  # ② 放进 data
                },
                message="Rating submitted successfully"
            )
        except Exception as e:
            current_app.logger.error(f"An error occurred in /api/submit_rating endpoint: {str(e)}")
            return GlobalResponseHandler.error(message=str(e))

    def handle_low_rating(session_id ,feedback: str,question: str, response:str):
        appkey = config.get('r1_online_appkey')
        url = config.get('r1_online_url')

        if not feedback: # 无
            content = f""""
            用户问题：{question}
            目前答案： {response}
            用户对于目前的答案不满意 请联网搜索并进行评判
            """  
        else:
            content = f""""
            用户问题：{question}
            目前答案： {response}
            用户对于目前的答案不满意 这是用户对于答案的反馈： {feedback}
            请联网搜索并进行评判
            """    

        payload = {
            "session_id":      session_id,
            "bot_app_key":     appkey,
            "visitor_biz_id":  session_id,
            "content":         content,
            "incremental":     True,
            "streaming_throttle": 10,
            "visitor_labels":  [],
            "custom_variables": {},
            "search_network":"enable"
        }

        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=False, timeout=None
        )
        resp.encoding = "utf-8"          # 告诉 requests 按 UTF‑8 解码
        sse_text = resp.text             # 现在就是正常中文
        # print(sse_text)
        return extract_reply_contents(sse_text)
  
    def extract_reply_contents(sse_text: str) -> str:
        """
        从完整的 SSE 文本中提取所有 event:reply 的 payload.content。
        返回一个列表，按出现顺序排列。
        """
        contents = []
        current_event = None
        data_buffer = []

        for line in sse_text.splitlines():
            if line.startswith("event:"):
                # 遇到新事件——先把上一段 data 处理掉
                if current_event == "reply" and data_buffer:
                    data_json = json.loads("".join(data_buffer))
                    contents.append(data_json["payload"]["content"])
                # 重置并记录新事件名
                current_event = line[6:].strip()
                data_buffer = []
            elif line.startswith("data:"):
                # 去掉开头 "data:" 累积 JSON 字符串
                data_buffer.append(line[5:].strip())

        # 处理文本末尾最后一段（若也是 reply）
        if current_event == "reply" and data_buffer:
            data_json = json.loads("".join(data_buffer))
            contents.append(data_json["payload"]["content"])

        return contents[-1]        




    @app.errorhandler(Exception)
    def handle_exception(e):
        current_app.logger.error(f"An unexpected error occurred: {str(e)}")
        return GlobalResponseHandler.error(message=f"Internal Server Error: {str(e)}")

    @app.teardown_appcontext
    def close_db(e=None):
        db = g.pop('db', None)
        if db is not None:
            db.close()

    return app

# Create the app instance for production (this is what Gunicorn will import)
app = create_app()

if __name__ == "__main__":
    # For local development only. Gunicorn will use the module-level "app".
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 6005)))
