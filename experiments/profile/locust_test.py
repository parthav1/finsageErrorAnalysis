from locust import HttpUser, task, between, events
import random
import uuid
import json
import time
import gevent

class ChatUser(HttpUser):
    wait_time = between(1,3)
    
    def on_start(self):
        # Load questions from file
        question_file = "/root/autodl-tmp/cjj/RAG_Agent/src/test/subquestions/subquestion_1.md"
        
        with open(question_file, 'r', encoding='utf-8') as f:
            self.questions = [q.strip() for q in f.readlines() if q.strip()]
            # self.questions = [q['question'] for q in questions
        
        # Generate a session ID for this user
        self.session_id = str(uuid.uuid4())
        
        # Add your bearer token here
        self.bearer_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJsb3R1cyBhZ2VudCB0ZXN0IiwibmFtZSI6IlNpbXBsZSBXYXkiLCJpYXQiOjE1MTYyMzkwMjJ9.L1FjNCS7G5L5KSwoMBIOWX5i7VOF2yGoq31ZdoNEpSI'
        
        # Determine number of chat rounds for this user (10-15)
        self.num_rounds = random.randint(10,15)
        self.current_round = 0
        
        # Start the first chat round
        self.perform_chat_round()
    
    @task
    def perform_chat_round(self):
        if self.current_round >= self.num_rounds:
            # User has completed all chat rounds, stop the user
            return
        
        # Increment the round counter
        self.current_round += 1
        
        # Select a question for this round
        question = random.choice(self.questions)
        
        # Perform chat request (streaming endpoint)
        self.api_chat_stream(question)
        
        # Schedule the next chat round after waiting (if more rounds are needed)
        if self.current_round < self.num_rounds:
            # Use gevent.spawn_later to schedule the next round
            wait_time = random.uniform(1, 3)
            gevent.spawn_later(wait_time, self.perform_chat_round)
    
    def api_chat_stream(self, question):
        payload = {
            "question": question,
            "session_id": self.session_id,
        }
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        # Record the start time
        start_time = time.time()
        
        with self.client.post(
            "/api_chat_stream",
            json=payload, 
            headers=headers,
            catch_response=True,
            stream=True,
            name="api_chat_stream"
        ) as response:
            if response.status_code == 200:
                try:
                    first_token_time = time.time()
                    ttft = (first_token_time - start_time) * 1000  # in ms
                    self.environment.events.request.fire(
                        request_type="TTFT",
                        name="time_to_first_token",
                        response_time=ttft,
                        response_length=0,
                        context={},
                        exception=None
                    )
                    # # Parse the first chunk to check for errors
                    # first_chunk_received = False
                    # for line in response.iter_lines():
                    #     if line:
                    #         # Decode and process the first SSE data chunk
                    #         first_chunk_received = True
                    #         json_part = line.decode('utf-8').split('data: ')[1]
                    #         data = json.loads(json_part)
                            
                    #         # Record successful time to first token
                            
                    #         # Check if the first chunk contains an error
                    #         if 'error' in data:
                    #             response.failure(f"Error in first chunk")
                    #         else:
                    #             print("good chunk")
                    #             # Log the time to first token as a custom stat
                                
                    #         break
                    
                    # If we didn't get any chunk at all, record that as failure
                    # if not first_chunk_received:
                    #     response.failure("No data chunks received in the response")
                    
                    # Consume the rest of the stream to complete the request
                    for _ in response.iter_content(chunk_size=4096):
                        pass
                        
                except Exception as e:
                    response.failure(f"Streaming error: {str(e)}")
            else:
                response.failure(f"Status code {response.status_code}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    with open("locust_stats.csv", "w") as f:
        f.write(environment.stats.to_csv())

# To run:
# locust -f locust_test.py --headless -u 5 -r 1 --host=http://localhost:6006 --run-time 30m
# This will simulate 10 users with 1 users spawned per second, running for 30 minutes
# Each user will have 10-15 chat rounds with the same session ID
