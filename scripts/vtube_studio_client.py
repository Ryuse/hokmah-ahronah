import json
import os
import time
from configparser import ConfigParser
from functools import partial
from pprint import pprint

import websocket

class VTubeStudioClient:
    def __init__(self, config_file="config.ini", ws_link="ws://localhost:8008", timeout=36000):
        self.cur_path = os.path.dirname(__file__)
        self.config_file_path = os.path.join(self.cur_path, config_file)

        self.config = ConfigParser()
        self.config.read(self.config_file_path)

        self.ws_link = ws_link
        self.timeout = timeout
        self.ws = None

        self.plugin_name = "Hokmah Ahronah"
        self.plugin_developer = "Ryuse"
        self.token = self.config["vtube_studio"].get("token", "")

        self.connect()

        # hotkey IDs
        self.hotkeys = {
            "reset": "bcfaf883e7f34bb39ab252c792ac05cc",
            "look_at_screen": "a61fc682bf2843de834f8101e5ce8b31",
            "happy_eyes_closed": "e25db6bf951b451ea1d5615dba7b5313",
            "sad": "d8e5f9027eb14944939997d642d44bc6",
            "damage": "18b21170c2554a5686fac80eea5e51c9",
            "strawberry_milk": "fb042609932e40b6b1e64b1ceb977cca",
            "nod_down": "9d3603e9336f40a7943a1f4fca44564e",
            "happy_eyes_closed_mouth_open": "8655bc33f4024860af62343a936de1fe",
            "happy_mouth_open": "76fc965e8e4e49a4bb49d73ed52b8022",
            "dum_ai": "bc3c4b14d331490f81cd3f46818114b6",
            "angry": "ace7be69c8d4432f96947bb6acc27753",
            "look_right": "7c26cef1b6994b809610585f705224f6",
            "smug": "e40d4fd9281a45908d38da27530ccf2a",
            "pain": "b2ba44ed00364dc78171ebd53c590c59",
            "confused": "30599166ebac40a1883d7c7398027b1f",
            "angery": "f8e1ac81b2424f69ba85ee01ccd44a85",
            "concerned": "371701a0f1c14426ad05c3dd45616a17",
            "pain_eyes_closed": "0c13a5f0127c49b88fc7a9c79bc2c730",
            "yandere": "11597af6c1df49a9bd213f024d572ab1",
            "indifferent": "26ca0eeb03d047d19c124e45c74c1c3f",
            "surprised": "28f3fe101ef64db6afd7f9c2c0dbeb9a",
            "scared": "a94ba4514510445399a3be2bfb8200ba",
            "disgust": "d337897358164376a62c13c92734869b",
            "nod_up": "3579eea405624665b2db1b05cfd9fb7b",
            "headpat_ahronah": "1a2cc03170f442ceb1040681a493ee74",
        }

        # map hotkey name → callable
        self.functions = {
            name: partial(self.toggle_expression, hotkey_id)
            for name, hotkey_id in self.hotkeys.items()
        }

    def connect(self):
        try:
            self.ws = websocket.create_connection(self.ws_link, timeout=self.timeout)
        except Exception as e:
            print(f"[VTubeStudioClient - Exception] {e}")
            self.ws = None

    def send_request(self, request_header, fail_count=0, debug=False):
        try:
            if debug:
                print("Sending request..")
            self.ws.send(json.dumps(request_header))
            result = json.loads(self.ws.recv())
            if debug:
                pprint(result, width=1)
                print("Request sent")
            return result
        except Exception as e:
            print(f"[VTubeStudioClient send_request - EXCEPTION] {e}")
            self.connect()
            self.authenticate()
            if fail_count < 5:
                return self.send_request(request_header, fail_count=fail_count + 1, debug=debug)

    def check_session(self):
        request_header = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "SessionCheck",
            "messageType": "APIStateRequest"
        }

        result = self.send_request(request_header, debug=False)
        if result and not result["data"]["currentSessionAuthenticated"]:
            self.authenticate()

    def get_token(self):
        request_header = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "GETTOKENID",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer
            }
        }

        result = self.send_request(request_header)
        pprint(f'TOKEN RESULTS {result}')

        if result:
            self.token = result["data"]["authenticationToken"]
            self.config["vtube_studio"]["TOKEN"] = self.token
            with open(self.config_file_path, "w") as configfile:
                self.config.write(configfile)

    def authenticate(self):
        if not self.token:
            print("No Token. Getting token..")
            self.get_token()

        request_header = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "AUTHENTICATE",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": self.token
            }
        }

        return self.send_request(request_header)

    def request_hotkeys(self):
        request_header = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "HotkeyListRequest",
            "messageType": "HotkeysInCurrentModelRequest",
            "data": {}
        }
        return self.send_request(request_header, debug=True)

    def toggle_expression(self, hotkey_id):
        st = time.time()
        request_header = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"HotkeyTrigger_{hotkey_id}",
            "messageType": "HotkeyTriggerRequest",
            "data": {
                "hotkeyID": hotkey_id
            }
        }
        self.send_request(request_header)
        elapsed_time = time.time() - st
        print('[toggle_expression Execution time]', elapsed_time, 'seconds')


if __name__ == "__main__":
    from voice.tts import VTuberTTS

    client = VTubeStudioClient(config_file="../config.ini")
    client.authenticate()
    client.request_hotkeys()

    tts = VTuberTTS(client)
    text = "I'm Hokmah Ahronah and I'm slowly gonna take over Noah. No one is safe here. No one. :("
    text_dict = {"text": text, "expression": None}
    st = time.time()
    tts.play_sound(text_dict)
    print("Execution time:", time.time() - st, "seconds")

    while True:
        inp = input(">")
        tts.play_sound({"text": inp, "expression": None})
    # Example: client.functions["angry"]()
