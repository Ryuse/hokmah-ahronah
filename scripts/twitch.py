import asyncio
import datetime
import glob
import json
import os
import random
import re
import time
from collections import deque
from configparser import ConfigParser

import speech_recognition as sr
import speech_recognition as srf
import twitchio
from nltk.stem import SnowballStemmer
from pyngrok import ngrok, conf
from twitchio.ext import commands, routines, eventsub, pubsub

# Set up NLTK stemmer
stemmer = SnowballStemmer(language='english')


class TwitchBot:
    def __init__(self,
                 db,
                 tts,
                 textgen,
                 obs_controller,
                 vtube_studio_client,
                 config="config.ini", ):
        # --- Config & File Paths ---
        self.tts = tts
        self.textgen = textgen
        self.config_file = config
        self.config = ConfigParser()
        self.config.read(self.config_file)
        self.cur_path = os.getcwd()
        self.responses_dict = {}

        # --- Twitch & Bot Variables ---
        self.BOT_NICKNAME = 'hokmahahronah'
        self.CHANNELS = ['noaharklightch']
        self.BROADCASTER_NUM = self.config['twitch']['main_broadcaster_num']
        self.MODERATOR_NUM = self.config['twitch']['moderator_broadcaster_num']
        self.CLIENT_ACCESS_TOKEN = self.config['twitch']['BROADCASTER_ACCESS_TOKEN']
        self.last_message_sent_at = datetime.datetime.now()
        self.is_talking = False
        self.recent_message_dict = {}
        self.current_expression = ""

        # --- Counters and States ---
        self.stream_strawberry_milk_count = 0
        self.stream_headpat_count = 0
        self.garlic_bread_stolen_counter = 0
        self.last_follow = ""
        self.best_headpatter_dict = db.check_best_headpatter()
        self.best_strawberry_milk_dict = db.check_best_strawberry_milk_giver()
        self.IDLE_MESSAGE_INTERVAL = 60

        # --- NGROK Setup ---
        ngrok.kill()
        conf.get_default().ngrok_version = "v3"
        conf.get_default().auth_token = self.config['ngrok']['AUTH_TOKEN']
        self.ssh_tunnel = ngrok.connect(4000, "http", bind_tls=True)
        print(f"NGROK URL: {self.ssh_tunnel.public_url}")
        self.url = self.ssh_tunnel.public_url
        self.port = 4000

        # --- Bot and Client Instances ---
        self.bot = commands.Bot.from_client_credentials(
            client_id=self.config['twitch']['BROADCASTER_API_ID'],
            client_secret=self.config['twitch']['BROADCASTER_API_SECRET']
        )
        self.client = twitchio.Client(token=self.CLIENT_ACCESS_TOKEN)
        self.client.pubsub = pubsub.PubSubPool(self.client)
        self.esclient = eventsub.EventSubClient(
            self.bot,
            webhook_secret=self.config['ngrok']['AUTH_TOKEN'],
            callback_route=f"{self.url}/callback"
        )

        # --- Custom Punctuation Mapping ---
        with open("responses/response_helpers/jokes.json", 'r', encoding="utf-8") as f:
            self.joke_list = json.load(f)["jokes"]

        # --- External Class Instances ---
        self.obs_controller = obs_controller
        self.vtube_studio_client = vtube_studio_client
        self.response_functions = {
            "obs_toggle_scene": self.obs_controller.switch_scene,
            "hotkeys": self.vtube_studio_client.request_hotkeys,
            "fetch_best_headpatter": db.check_best_headpatter,
            "joke_reply": self.joke_reply,
            "damage": self.vtube_studio_client.functions["damage"],
            "headpat_ahronah": self.headpat_ahronah,
            "strawberry_milk": self.strawberry_milk
        }

        # --- Initialize Bot and Listeners ---
        self.bot_instance = commands.Bot(
            token=self.CLIENT_ACCESS_TOKEN,
            nick=self.BOT_NICKNAME,
            prefix='?!?',
            initial_channels=self.CHANNELS
        )
        self._register_listeners()
        self._load_responses()

    def _register_listeners(self):
        """Register event handlers and routines with the bot instance."""
        # Main bot events
        self.bot_instance.event(self.event_message)
        self.bot_instance.event(self.event_ready)

        # Custom commands
        self.bot_instance.command(name='test')(self.test_command)
        self.bot_instance.command(name='updateResponses')(self.update_response_from_files)

        # Routines
        self.reply_message_routine = routines.routine(seconds=3)(self.reply_message)
        self.idle_message_routine = routines.routine(seconds=self.IDLE_MESSAGE_INTERVAL)(self.idle_message)
        self.reply_message_routine.error(self.reply_message_on_error)
        self.idle_message_routine.error(self.idle_message_on_error)

        # EventSub & PubSub listeners
        self.bot.event(self.event_eventsub_notification_followV2)
        self.bot.event(self.event_eventsub_notification_raid)
        self.bot.event(self.event_eventsub_notification_subscription)
        self.bot.event(self.event_eventsub_notification_cheer)
        self.client.event(self.event_pubsub_channel_points)
        self.client.event(self.event_pubsub_bits)

    def _load_responses(self):
        """Loads responses from JSON files and initializes deques."""
        path = f'{self.cur_path}/responses'
        for file_path in glob.glob(os.path.join(path, '*.json')):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_name = os.path.basename(file_path)
                file_name_no_ext = os.path.splitext(file_name)[0]
                self.responses_dict[file_name_no_ext] = json.load(f)["responses"]

        self.idle_responses = self.responses_dict["idle"]
        self.lore_responses = self.responses_dict["lore"]
        self.noah_responses = self.responses_dict["noah"]
        self.follow_responses = self.responses_dict["follow"]
        self.subscribe_responses = self.responses_dict["subscribe"]
        self.cheer_responses = self.responses_dict["cheer"]
        self.raid_responses = self.responses_dict["raid"]
        self.rewards_responses = self.responses_dict["rewards"]

        self.last_idle_messages = deque([], len(self.idle_responses) - 1)
        self.last_follow_messages = deque([], len(self.follow_responses) - 1)
        self.last_subscribe_messages = deque([], len(self.subscribe_responses) - 1)
        self.last_cheer_messages = deque([], len(self.cheer_responses) - 1)
        self.last_raid_messages = deque([], len(self.raid_responses) - 1)

    async def _start_eventsub(self):
        """Initializes and subscribes to Twitch EventSub."""
        self.bot.loop.create_task(self.esclient.listen(port=self.port))
        await asyncio.sleep(5)
        await self.esclient.delete_all_active_subscriptions()
        await self.esclient.subscribe_channel_follows_v2(broadcaster=self.BROADCASTER_NUM,
                                                         moderator=self.BROADCASTER_NUM)
        await self.esclient.subscribe_channel_raid(to_broadcaster=self.BROADCASTER_NUM)
        await self.esclient.subscribe_channel_subscriptions(broadcaster=self.BROADCASTER_NUM)
        print("TwitchIO Subs All subscribed")

    async def run(self):
        """Starts the bot, EventSub client, and PubSub client."""
        await self._start_eventsub()

        # Start PubSub client
        topics = [
            pubsub.channel_points(self.CLIENT_ACCESS_TOKEN)[self.BROADCASTER_NUM],
            pubsub.bits(self.CLIENT_ACCESS_TOKEN)[self.BROADCASTER_NUM]
        ]
        await self.client.pubsub.subscribe_topics(topics)
        await self.client.start()

        # Start bot routines
        self.idle_message_routine.start(stop_on_error=False)
        self.reply_message_routine.start(stop_on_error=False)

        await self.bot_instance.start()

    # ===================== #
    # Helper Methods
    # ===================== #
    def remove_prefix(self, string, prefix):
        return string[len(prefix):] if string.startswith(prefix) else string

    def remove_suffix(self, string, suffix):
        return string[:-len(suffix)] if string.endswith(suffix) else string

    def joke_reply(self):
        joke = random.choice(self.joke_list)
        split_sentence = joke.split("?")
        if "?" in joke:
            split_sentence[0] = split_sentence[0] + "?"
        formatted_responses = [{"text": line.strip(), "expression": "happy_eyes_closed"} for line in split_sentence]
        asyncio.create_task(self.send_multiline_msg(formatted_responses))

    def headpat_ahronah(self):
        self.stream_headpat_count += 1
        self.vtube_studio_client.functions["headpat_ahronah"]()
        self.best_headpatter_dict = self.db.check_best_headpatter()

    def strawberry_milk(self):
        self.stream_strawberry_milk_count += 1
        self.vtube_studio_client.functions["strawberry_milk"]()
        self.best_strawberry_milk_dict = self.db.check_best_strawberry_milk_giver()

    def message_probability(self, user_message, recognised_words, required_words=()):
        message_certainty = sum(1 for word in user_message if word in recognised_words)
        percentage = float(message_certainty) / float(len(recognised_words))
        has_required_words = all(word in user_message for word in required_words)
        return percentage * 100 if has_required_words else 0

    def check_all_messages(self, message, responses):
        highest_prob_list = {}

        for i, response_data in enumerate(responses):
            keywords = {stemmer.stem(word) for sentence in response_data["keywords"] for word in
                        re.split(r'\s+|[,;?!.\'-]\s*', sentence) if word.strip()}
            required_keywords = {stemmer.stem(word) for word in response_data["required_keywords"] if word.strip()}
            highest_prob_list[i] = self.message_probability(message, keywords, required_keywords)

        best_match = max(highest_prob_list, key=highest_prob_list.get)
        print(f"[check_all_messages USER WORDS] {message}")
        print(f'[check_all_messages BEST MATCH] {responses[best_match]["tag"]}: {highest_prob_list[best_match]:.2f}%')

        return best_match if highest_prob_list[best_match] >= 45 else -1

    def get_response(self, user_id, user_name, user_input, responses, prefix=None, suffix=None, is_question=True):
        if prefix: user_input = self.remove_prefix(user_input.lower(), prefix)
        if suffix: user_input = self.remove_suffix(user_input.lower(), suffix)

        text_command = user_input.lower().strip()
        stemmed_message = [stemmer.stem(word) for word in re.split(r'\s+|[,;?!.-]\s*', text_command) if word.strip()]

        response_index = self.check_all_messages(stemmed_message, responses)

        if response_index == -1:
            self.db.record_message(user_id, user_input, unknown=True)
            dict_response = self.textgen.generate_text(text_command, name=user_name, is_question=is_question)
            response_context_set = "text_generator"
        else:
            response_data = responses[response_index]
            response_context_set = response_data["context_set"]
            response_tag = "responses"
            response = response_data[response_tag]

            recent_context = self.db.check_message_context(user_id)
            if response_context_set in recent_context:
                context_count = recent_context.count(response_context_set)
                if f"responses_sequence_{context_count + 1}" in response_data:
                    response_tag = f"responses_sequence_{context_count + 1}"
                    response = response_data[response_tag]

            response = random.choice(response)

            dict_response = {
                "response": response,
                "function": response_data["function"],
                "args": response_data["args"],
            }
            if f"{response_tag}_to_text_gen" in response_data and response_data[f"{response_tag}_to_text_gen"]:
                gen_dict_response = self.textgen.generate_text(response[0]["text"], name=user_name, is_question=False)
                dict_response["response"] = gen_dict_response["response"]

        self.db.record_message_context(user_id, response_context_set)
        return dict_response

    async def handle_lore_command(self, lore, **kwargs):
        response = lore["response"]
        function = lore.get("function")
        args = lore.get("args")
        before_text_function = lore.get("before_text_function")

        print(f"[handle_lore_command response] {response}")

        if before_text_function and before_text_function in self.response_functions:
            print(f"[LORE COMMANDS] Before Text Function '{before_text_function}' found.")
            if args:
                self.response_functions[before_text_function](args)
            else:
                self.response_functions[before_text_function]()

        await self.send_multiline_msg(response, **kwargs)

        if function and function in self.response_functions:
            print(f"[LORE COMMANDS] Function '{function}' found.")
            if args:
                self.response_functions[function](args)
            else:
                self.response_functions[function]()

    async def send_random_response(self, response_list, last_message_list, **kwargs):
        random_response_dict = random.choice(response_list)
        random_response = random.choice(random_response_dict["responses"])
        while random_response[0]["text"] in last_message_list:
            random_response_dict = random.choice(response_list)
            random_response = random.choice(random_response_dict["responses"])

        response_dict = {
            "response": random_response,
            "function": random_response_dict["function"],
            "args": random_response_dict["args"],
        }
        await self.handle_lore_command(response_dict, **kwargs)
        last_message_list.appendleft(random_response[0]["text"])

    async def send_multiline_msg(self, texts, **kwargs):
        if self.is_talking:
            return

        self.is_talking = True
        for text_dict in texts:
            print(f"[send_multiline_msg is_talking:{self.is_talking}]")
            text_dict["text"] = text_dict["text"].format(
                strawberry_milk_count=self.stream_strawberry_milk_count,
                headpat_count=self.stream_headpat_count,
                **self.best_headpatter_dict,
                **self.best_strawberry_milk_dict,
                **kwargs
            )
            self.tts.play_sound(text_dict)
            time.sleep(1)  # Delay between lines

        self.is_talking = False
        self.last_message_sent_at = datetime.datetime.now()
        print(f'[is_talking end] {self.is_talking}')

    # ===================== #
    # Event Handlers
    # ===================== #
    async def event_message(self, ctx):
        if ctx.echo:
            self.last_message_sent_at = datetime.datetime.now()
            return

        sender_name = ctx.author.name
        sender_id = ctx.author.id
        text = ctx.content

        self.db.record_user(sender_name, sender_id)
        self.db.record_message(sender_id, text)

        if text.startswith(("?", "!", ">", "-")):
            return

        self.recent_message_dict = {"name": sender_name, "id": sender_id, "text": text}
        print(f'> {sender_name} ({sender_id}): {text}')

        await self.bot_instance.handle_commands(ctx)

    async def event_ready(self):
        print('====================')
        print('Bot is ready')
        print(f'Logged in as: {self.bot_instance.nick}')
        print(f'User id is: {self.bot_instance.user_id}')
        print('====================\n')

    async def event_eventsub_notification_followV2(self, payload: eventsub.ChannelFollowData):
        print('Received follow event!')
        user = payload.data.user.name
        if user != self.last_follow:
            self.last_follow = user
            await self.send_random_response(self.follow_responses, self.last_follow_messages, user=user)

    async def event_eventsub_notification_raid(self, payload: eventsub.ChannelRaidData):
        raider = payload.data.raider.name
        print(f'[RAID] {raider} raided')
        await self.send_random_response(self.raid_responses, self.last_raid_messages, user=raider)

    async def event_eventsub_notification_subscription(self, payload: eventsub.ChannelSubscribeData):
        user = payload.data.user.name
        print(f"{user} has subbed to you!")
        await self.send_random_response(self.subscribe_responses, self.last_subscribe_messages, user=user)

    async def event_eventsub_notification_cheer(self, payload: eventsub.ChannelCheerData):
        user = payload.data.user.name
        print(f"[CHEER] {user} cheered")
        await self.send_random_response(self.cheer_responses, self.last_cheer_messages, user=user)

    async def event_pubsub_channel_points(self, event: pubsub.PubSubChannelPointsMessage):
        print("Channel Points Redeemed")
        event_id = event.reward.id
        user = event.user.name
        self.db.record_redeem(event.user.id, event.reward.id)

        for response_data in self.rewards_responses:
            if event_id == response_data["reward_id"]:
                response_list = response_data["responses"]
                response = random.choice(response_list)

                dict_response = {
                    "response": response,
                    "function": response_data.get("function"),
                    "args": response_data.get("args"),
                }

                if response_data.get("responses_to_text_gen"):
                    gen_dict_response = self.textgen.generate_text(response[0]["text"], name=user, is_question=False)
                    dict_response["response"] = gen_dict_response["response"]

                dict_response["before_text_function"] = response_data.get("before_text_function")
                await self.handle_lore_command(dict_response, user=user)
                return

    async def event_pubsub_bits(self, event: pubsub.PubSubBitsMessage):
        pass

    # ===================== #
    # Routines
    # ===================== #
    async def reply_message(self):
        if self.is_talking or not self.recent_message_dict:
            return

        sender_id = self.recent_message_dict["id"]
        name = self.recent_message_dict["name"]
        text = self.recent_message_dict["text"]

        if name.lower() == "noaharklightch":
            name = "Noah"

        if re.search('[a-zA-Z]', text) and len(text) < 100:
            response = None
            used_prefix = next((p for p in srf.prefixes if text.lower().startswith(p)), None)
            used_suffix = next((s for s in srf.suffixes if text.lower().endswith(s)), None)

            if used_prefix:
                response = self.get_response(sender_id, name, text, self.lore_responses, prefix=used_prefix)
            elif used_suffix:
                response = self.get_response(sender_id, name, text, self.lore_responses, suffix=used_suffix)
            else:  # If no special prefix/suffix, assume it's a general question
                response = self.get_response(sender_id, name, text, self.lore_responses)

            if response:
                await self.handle_lore_command(response, user=name)

        self.recent_message_dict = {}

    async def reply_message_on_error(self, error: Exception):
        print(f"[reply_message EXCEPTION] {error}")

    async def idle_message(self):
        time_diff = (datetime.datetime.now() - self.last_message_sent_at).total_seconds()
        if time_diff < self.IDLE_MESSAGE_INTERVAL or self.is_talking:
            return

        sender_id = self.MODERATOR_NUM
        num = random.randint(0, 100)

        if num <= 90:
            print('[idle message routine] Sending preset query')
            story_starters = ["The other day I", "Once upon a time", "Did you know that I",
                              "Sometimes I really like to", "I'm sorry guys I", "I'm not sure", "I just want to",
                              "I wonder", "I think", "Sometimes", "I really like", "Maybe I", "Someone told me",
                              "Somebody once told me", "There was this one time", "What I want to do sometimes is",
                              "As someone who", "Um.."]
            starter = random.choice(story_starters)
            long_response = random.randint(0, 100) > 70
            response = self.textgen.generate_text(starter, is_question=False, long_response=long_response)
            await self.handle_lore_command(response)
        else:
            print('[idle message routine] Sending random response')
            await self.send_random_response(self.idle_responses, self.last_idle_messages)

    async def idle_message_on_error(self, error: Exception):
        print(f"[idle_message EXCEPTION] {error}")

    # ===================== #
    # Voice Command Threading
    # ===================== #
    async def voice(self):
        print('Starting voice')
        r = sr.Recognizer()
        mic_list = sr.Microphone.list_working_microphones()
        mic_device = next((sr.Microphone(device_index=mic_idx) for mic_idx, mic_name in mic_list.items() if
                           "Microphone (USB Condenser Micro" in mic_name), None)

        if not mic_device:
            print("No working microphones found!")
            return

        with mic_device as source:
            r.adjust_for_ambient_noise(source)
            while True:
                print("Say something")
                try:
                    audio = r.listen(source, phrase_time_limit=5)
                    print("Recognizing..")
                    text = r.recognize_google(audio).strip().lower()

                    used_prefix = next((p for p in srf.prefixes if text.startswith(p)), None)
                    used_suffix = next((s for s in srf.suffixes if text.endswith(s)), None)

                    if used_prefix or used_suffix:
                        response = self.get_response(
                            user_id=self.BROADCASTER_NUM,
                            user_name="Noah",
                            user_input=text,
                            responses=self.lore_responses,
                            prefix=used_prefix,
                            suffix=used_suffix
                        )
                        print(f"[VOICE RESPONSE] {response}")
                        await self.handle_lore_command(response, user="Noah")
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(2)

    def between_voice_callback(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.voice())
        loop.close()

