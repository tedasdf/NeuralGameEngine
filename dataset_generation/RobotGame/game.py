import random
import time
import json
import websocket
import threading
import numpy as np
class Game:
    def __init__(self, url, sequence_length):

        self.num_sequences = sequence_length
        self.sequence = []
        self.player_sequence = []

        self.active = False
        
        self.url = url
        self.connected = False

        self.reward = None
        
        
        self.ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: self.on_message(ws, msg),
            on_open=lambda ws: self.on_open(ws),
            on_error=lambda ws, err: self.on_error(ws, err),
            on_close=lambda ws: self.on_close(ws)
        )   

        self.start_seq()
        

    def start_seq(self):
        # Start WebSocket listener in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        # Wait until connected (max wait time: 5 seconds)
        timeout = 5
        waited = 0
        while not self.connected and waited < timeout:
            print("Waiting for WebSocket connection...")
            time.sleep(1)
            waited += 1

    def reset(self, new_sequence):
        self.sequence = []
        self.player_sequence = []
        self.active = False

        print("Game reset.")

        self.beginning_seq(new_sequence)
        

    def beginning_seq(self, new_sequence):
        self.sequence = new_sequence
        self.active = True
        print(f"Generated sequence: {new_sequence}")
        self.ws.send(json.dumps({"command": "begin" , "seq" : new_sequence}))
        

    def check_sequence(self, robot_pos):
        reward = 0 
        if not self.active:
            return 0 , False
        else:    
            if self.player_sequence == self.sequence[:len(self.player_sequence)]:
                if len(self.player_sequence) == self.num_sequences:
                        return self.num_sequences , True
            
                # Reward based on how close robot is to the target button
                dist = np.linalg.norm(np.array(robot_pos) - np.array(self.target_position[self.sequence[len(self.player_sequence)] -  1]))
                max_dist = 0.5  # max meaningful distance, tune this
                proximity_reward = max(0, (max_dist - dist) / max_dist)  # normalized between 0 and 1
                proximity_reward = min(proximity_reward, 0.7)  # cap at 0.7
                reward += proximity_reward
                
                return len(self.player_sequence) , False
            else:

                return len(self.player_sequence) - 1, True

    

    def on_message(self, ws, message):
        print(f"Received from ESP32: {message}")
        data = json.loads(message)

        if 'button' in data:
            print("Here the button")
            print(data['button'])
            button = data['button'] - 1  # ESP32 sends 1-indexed buttons
            self.player_sequence.append(button)
            print(f"Check player_sequence: {self.player_sequence}")

            # if result == "Win":
            #     ws.send(json.dumps({"command": "END", "result": "win"}))
            # elif result == "Incorrect":
            #     ws.send(json.dumps({"command": "END", "result": "incorrect"}))
            # else:
            #     pass  # Correct so far

    def on_open(self,ws):
        print("Connected to ESP32 WebSocket server.")
        self.connected = True
        # seq = [ 2, 1, 0 ,3]
        # seq = self.beginning_seq(seq)
        # self.ws.send(json.dumps({"command": "begin" , "seq" : seq}))

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws):
        self.connected = False
        print("WebSocket connection closed.")
    
    def send_command(self, command_dict):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(command_dict))
            print(f"Sent: {command_dict}")
        else:
            print("WebSocket not connected")

    


if __name__ == "__main__":
    ESP32_IP = "192.168.0.100"  # Replace with your ESP32's IP
    PORT = 81                   # Your ESP32 WebSocket port
    url = f"ws://{ESP32_IP}:{PORT}"

    game = Game(url, sequence_length=4)

    # Start WebSocket listening thread
    sequence = [ 3, 2, 0 ,1]

    if  game.connected & game.active == False:
        game.reset(sequence)
    time.sleep(1)
    # if game.connected:
    #     print("WebSocket connected successfully!")
    #     # You can now interact here (you could add input() to send custom commands)
    #     while True:
    #         if game.active == False:
    #             game.reset(sequence)
    #         print(game.player_sequence)
    #         print(game.sequence)
    #         print(game.check_sequence())
    #         time.sleep(1)
    #         # if game.player_sequence !== prev_sequence :
    #         #     print(game.player_sequence)


    #         # prev_sequence = game.player_sequence        
            

    # else:
    #     print("WebSocket not connected.")





