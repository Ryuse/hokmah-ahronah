from threading import Thread
from scripts.vtube_studio_client import VTubeStudioClient
from scripts.text_generation.text_generator import TextGenerator
from scripts.voice.tts import VTuberTTS
from scripts.twitch import TwitchBot
from scripts.database.db import DatabaseManager
from scripts.obs import OBSController
# ============= ## ============= ## ============= #
def run_twitch():
    vtube_studio_client = VTubeStudioClient()
    vtube_studio_client.authenticate()

    db = DatabaseManager("sqlite:///scripts/database/twitch.db")
    text_generator = TextGenerator()
    vtube_studio_client = VTubeStudioClient("../config.ini")
    vtube_studio_client.authenticate()
    vtuber_tts = VTuberTTS(vtube_studio_client)

    bot_instance = TwitchBot(db=db,
                             textgen=text_generator,
                             tts=vtuber_tts,
                             obs_controller=OBSController(),
                             vtube_studio_client=vtube_studio_client)
    t1 = Thread(target=bot_instance.run)
    # t2 = Thread(target=between_voice_callback)
    # t3 = Thread(target=vtube.between_wsapp_callback)
    print('Starting..')

    t1.start()
    # time.sleep(4)
    # t2.start()
    # t3.start()

def try_tts():
    text_generator = TextGenerator()
    vtube_studio_client = VTubeStudioClient("../config.ini")
    vtube_studio_client.authenticate()
    vtuber_tts = VTuberTTS(vtube_studio_client)

    print("\n--- Starting Chatbot Loop ---")
    print("Type '.q' to quit or '.updatecontext' to reload files.")

    while True:
        user_input = input("> ")
        if user_input.lower() == ".q":
            break
        elif user_input.lower() == ".updatecontext":
            text_generator.reopen_files()
            continue

        response_dict = text_generator.generate_text(
            sentence=user_input,
            is_question=True,
            name="TheTester"
        )

        texts = response_dict.get("response", [])
        print(f"[FINAL TEXT] {texts}\n")

        for text in texts:
            vtuber_tts.play_sound(text)

if __name__ == '__main__':
    try_tts()
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
