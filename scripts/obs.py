import sys
import logging
import time
from configparser import ConfigParser
from obswebsocket import obsws, requests, events

logging.basicConfig(level=logging.INFO)
sys.path.append('../')


class OBSController:
    def __init__(self, config_file="config.ini", host="localhost", port=4444):
        # Read config
        self.config = ConfigParser()
        self.config.read(config_file)

        self.host = host
        self.port = port
        self.password = self.config['obs']['PASSWORD']

        # WebSocket connection
        self.ws = obsws(self.host, self.port, self.password)

        # Register default event handlers
        self.ws.register(self._on_event)
        self.ws.register(self._on_switch, events.SwitchScenes)

    def connect(self):
        """Establish connection to OBS."""
        try:
            self.ws.connect()
            logging.info("Connected to OBS WebSocket")
        except Exception as e:
            logging.error(f"Failed to connect to OBS: {e}")
            raise

    def disconnect(self):
        """Close connection to OBS."""
        try:
            self.ws.disconnect()
            logging.info("Disconnected from OBS WebSocket")
        except Exception as e:
            logging.error(f"Error disconnecting from OBS: {e}")

    def _on_event(self, message):
        """Generic event handler (currently unused)."""
        # logging.debug(f"Got event: {message}")
        pass

    def _on_switch(self, message):
        """Triggered when scenes are switched."""
        logging.info(f"Switched scene to: {message.getSceneName()}")

    def switch_scene(self, name: str, delay: int = 2):
        """Switch to a given scene by name."""
        self.connect()
        logging.info(f"Switching to scene: {name}")
        self.ws.call(requests.SetCurrentScene(name))
        time.sleep(delay)
        self.disconnect()

    def list_scenes(self):
        """Fetch and print all available scenes."""
        self.connect()
        scenes = self.ws.call(requests.GetSceneList())
        for s in scenes.getScenes():
            logging.info(f"[SCENE NAME] {s['name']}")
        self.disconnect()


if __name__ == "__main__":
    obs = OBSController(config_file="../config.ini")

    # Example usage
    obs.list_scenes()
    obs.switch_scene("Scene 2")  # Replace with your actual scene name
