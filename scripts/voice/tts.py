import os
import time
from pprint import pprint
import threading

import librosa
import numpy as np

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

from pydub import AudioSegment
from pydub.playback import _play_with_pyaudio
from transformers import pipeline


class VTuberTTS:
    def __init__(self, vtube_studio_client, model_name='tts_models/en/ljspeech/tacotron2-DDC_ph'):
        self.vtube_studio_client = vtube_studio_client
        # paths
        self.cur_path = os.path.dirname(__file__)
        self.speech_text_path = os.path.join(self.cur_path, 'export/speech.txt')
        self.audio_path = os.path.join(self.cur_path, 'export/audio.wav')

        # model manager
        tts_model_path = os.path.join(self.cur_path, 'voice_model/TTS/TTS/.models.json')
        self.model_manager = ModelManager(tts_model_path, verbose=False)

        # load TTS + vocoder
        model_path, config_path, model_item = self.model_manager.download_model(model_name)
        voc_path, voc_config_path, _ = self.model_manager.download_model(model_item["default_vocoder"])

        self.syn = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            vocoder_checkpoint=voc_path,
            vocoder_config=voc_config_path,
            use_cuda=True
        )

        # emotion classifier
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

        self.current_expression = ""

        # custom punctuation mapping
        self.punctuations = {
            ":(": " sad face ",
            "):": " sad face ",
            ":)": " happy face ",
            "(:": " happy face ",
            ":D": " big smiley face ",
            " ok?": " O.K?",
            " / ": " slash ",
            " ' ": " quote  ",
            ' " ': " quote  ",
            " = ": " equals ",
            " ( ": "",
            " ) ": "",
            ",": ".",
            "vtuber": "V Tuber",
            "<3": "heart",
            "II": "I.I",
            "III": "I.I.I",
        }

    # --- new threading methods ---
    def _write_to_file_thread(self, text_dict_list, duration_per_character):
        self.vtube_studio_client.authenticate()
        self.vtube_studio_client.functions["reset"]()

        self.current_expression = ""
        send_expression = True
        for text_dict in text_dict_list:
            text = text_dict["text"]
            expression = text_dict["expression"]

            if expression and expression in self.vtube_studio_client.functions and expression != self.current_expression and send_expression:
                self.vtube_studio_client.functions[expression]()
                self.current_expression = expression
                send_expression = not send_expression

            with open(self.speech_text_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

            time.sleep(duration_per_character * len(text))

        self.vtube_studio_client.functions["look_at_screen"]()
        self.current_expression = ""

    def _play_with_pyaudio_thread(self, hipitch_sound):
        time.sleep(1)
        _play_with_pyaudio(hipitch_sound)

    # --- core helpers ---
    def _get_expression(self, label, score):
        if label == "neutral":
            return None
        if label == "sadness":
            return "pain_eyes_closed" if score > 0.8 else "sad" if score > 0.5 else "pain"
        if label == "anger":
            return "indifferent" if score > 0.8 else "angry"
        if label == "disgust":
            return "confused"
        if label == "fear":
            return "scared" if score > 0.4 else "confused"
        if label == "joy":
            if score > 0.9:
                return "happy_eyes_closed_mouth_open"
            elif score > 0.5:
                return "happy_eyes_closed"
            else:
                return "angery"
        if label == "surprise":
            return "surprised"
        return None

    def _sentence_splitter(self, response_dict, length):
        words = iter(response_dict["text"].split())
        main_expression = response_dict["expression"]
        current = next(words)
        lines = []

        for word in words:
            if len(current) + 1 + len(word) > length:
                lines.append(current)
                current = word
            else:
                current += " " + word
        lines.append(current)

        formatted = []
        for line in lines:
            expression = main_expression
            if expression is None:
                emotion = self.classifier(line)
                label = emotion[0][0]["label"]
                score = emotion[0][0]["score"]
                expression = self._get_expression(label, score)
            formatted.append({"text": line, "expression": expression})

        pprint(formatted)
        return formatted

    # --- main interface ---
    def play_sound(self, response_dict):
        clean_text = response_dict["text"]
        for punc, word in self.punctuations.items():
            clean_text = clean_text.replace(punc, word)

        outputs = self.syn.tts(clean_text)
        self.syn.save_wav(outputs, self.audio_path)

        sound = AudioSegment.from_wav(self.audio_path)
        new_sample_rate = int(sound.frame_rate * (2.0 ** 0.3))
        hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

        total_duration = hipitch_sound.duration_seconds
        duration_per_character = total_duration / len(clean_text)

        text_list = self._sentence_splitter(response_dict, 60)

        # Create and start the threads
        write_thread = threading.Thread(target=self._write_to_file_thread, args=(text_list, duration_per_character,))
        play_thread = threading.Thread(target=self._play_with_pyaudio_thread, args=(hipitch_sound,))

        write_thread.start()
        play_thread.start()

        # Wait for both threads to finish
        write_thread.join()
        play_thread.join()

        with open(self.speech_text_path, 'w', encoding="utf-8") as f:
            f.write("")

    @staticmethod
    def pitch_shift(sound, n_steps):
        y = np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32)
        y = librosa.effects.pitch_shift(y, sound.frame_rate, n_steps=n_steps)
        return AudioSegment(np.array(y, dtype=np.int16).tobytes(),
                            frame_rate=sound.frame_rate,
                            sample_width=2,
                            channels=1)