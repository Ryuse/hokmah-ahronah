import random
import time
import re
import torch
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer
from nltk.tokenize import sent_tokenize
from better_profanity import profanity


class TextGenerator:
    """A class for generating text-based responses using a pre-trained language model."""

    def __init__(self, model_id="EleutherAI/gpt-neo-1.3B",
                 context_path="context.txt",
                 responses_path="responses/lore.json"):
        """Initializes the Chatbot with a model and context files."""
        # --- Constants & Paths ---
        cur_path = os.path.dirname(__file__)
        self.context_text_path = os.path.join(cur_path, context_path)
        self.lore_responses_path = os.path.join(cur_path, responses_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        # --- Profanity & Whitelisting ---
        self.bad_words = ["fuck", "fucking", "shit", "shitty", "bitch", "cancer", "gay", "gays", "faggot", "bisexual",
                          "homosexual", "incel", "simp", "native", "weed", "holocaust", "nazis", "nazi", "jew",
                          "hitler", "black", "white", "cunt", "damn", "ex", "kill", "penis", "vagina", "virgin", "dick",
                          "sex", "dicks", "asshole", "butt", "kink", "boyfriend", "girlfriend", "wife", "husband",
                          "kids"]
        self.other_bad_words = ["skin tone", "gender fluidity"]
        self.whitelisted_bad_words = ["god", "hell", "suck", "stupid", "ass"]

        # --- Model and Tokenizer ---
        self._load_model_and_tokenizer()

        # --- Load Context and Data ---
        self.context = ""
        self.data = {}
        self.reopen_files()

    def _load_model_and_tokenizer(self):
        """Loads and optimizes the language model and tokenizer."""
        print("Loading model and tokenizer...")
        model_hf = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model = BetterTransformer.transform(model_hf, keep_original_model=False)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.bad_words_ids = self.tokenizer(self.bad_words, add_special_tokens=False).input_ids
        profanity.add_censor_words(self.bad_words)
        profanity.load_censor_words(whitelist_words=self.whitelisted_bad_words)
        print("Model and tokenizer loaded.")

    def reopen_files(self):
        """Reloads the context and lore data from their respective files."""
        print("Reloading context and lore files...")
        with open(self.context_text_path, "r", encoding="utf-8") as f:
            context_lines = f.readlines()
            context_list = [line for line in context_lines if line.strip() and line[0] not in ["-", "#"]]
            self.context = " ".join(context_list).replace("\n", " ").strip()
            print("Context loaded.")

        with open(self.lore_responses_path, 'r', encoding="utf-8") as f:
            self.data = json.load(f)
            print("Lore responses loaded.")

    def _clean_text(self, text):
        """Cleans and formats generated text."""
        replace_words = {
            "RAW Paste Data": "", "TitleBody": "", "TitleSave": "", "CancelSave": "",
            "Chapter Text": "", "Body Cancel": "", "http:/": "", "http://": "",
            "\xa0": "", "\n": "", "\\": "", "?/": "", "<|endoftext|>": ".",
            "é": "e", "[IMG]": "", "Advertisements.": ""
        }

        # URL handling
        url_regex = '(:?(?:https?:\/\/)?(?:www\.)?)?[-a-z0-9]+\.(?:com|gov|org|net|edu|biz|tv)'
        for word in text.split():
            match = re.search(url_regex, word)
            if match:
                url = match.group()
                suffix = "noaharklight"
                if "twitch.tv" in url:
                    suffix = "noaharklightch"
                elif "youtube.com" in url:
                    suffix = "@noaharklight"
                text = text.replace(word, f"{url}/{suffix}")

        for key, value in replace_words.items():
            text = text.replace(key, value)

        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r'([~_])\1+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        prefixes = ["Ahronah", "ahronah"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text.replace(prefix, "").strip()

        if text.startswith(","):
            text = text[1:].strip()

        return text.strip()

    def _check_profanity(self, text):
        """Checks if the text contains any profane words."""
        if profanity.contains_profanity(text):
            return True
        for bad_word in self.other_bad_words:
            if bad_word in text:
                return True
        return False

    def _you_to_i(self, statement):
        """Converts 'you' references to 'I' references in a statement for conversational flow."""
        statement = statement.strip().lower()
        if statement.endswith(("?", ".")):
            statement = statement[:-1]

        replace_dict = {
            "you've": "I've", "yourself": "myself", " your ": " my ",
            "you're": "I'm", "i'm": "you're", "she ": "i ", "we're": "you're",
            "we've": "you've", "i've": "you've"
        }

        word_list = statement.split()
        first_you_replaced = False
        for i, word in enumerate(word_list):
            if word == "you":
                words_before = ["am", "do", "can", "will", "did", "should", "would", "have", "had", "think", "who",
                                "when", "where", "what", "why", "how"]
                if i > 0 and word_list[i - 1] in words_before:
                    word_list[i] = "I"
                    first_you_replaced = True
                elif i == 0 or (not first_you_replaced and word_list.count("you") > 1):
                    word_list[i] = "I"
                    first_you_replaced = True
                else:
                    word_list[i] = "me"
            elif word == "am":
                word_list[i] = "are"
            elif word == "are":
                word_list[i] = "am"
            elif word in ("i", "me"):
                word_list[i] = "you"

        statement = " ".join(word_list)
        statement = statement.replace(' i ', " I ")
        for key, value in replace_dict.items():
            statement = statement.replace(key, value)

        if not statement.endswith("?"):
            statement += "?"

        return statement

    def _generate_response(self, query, penalty=5, max_length=50, response_context=None, name=None, is_question=False):
        """Generates a text response from the language model."""
        if response_context is None:
            response_context = self.context

        if is_question:
            query = self._you_to_i(query)
            prompt = f"{response_context} {name}, you asked about {query}"
        else:
            prompt = f"{response_context} {query}"

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        token_length = len(input_ids[0])
        max_message_length = token_length + max_length

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            max_length=max_message_length,
            length_penalty=penalty,
            bad_words_ids=self.bad_words_ids,
        )

        inference_result = self.tokenizer.batch_decode(gen_tokens)[0].replace(response_context, "").strip()
        if is_question:
            inference_result = inference_result.replace(f"{name}, you asked about ", "").strip()

        print(f"[generate_response RESULTS] {inference_result}")
        return inference_result

    def generate_text(self, sentence, name=None, is_question=False, long_response=False, initial_max_length=70,
                      continuation_length=70):
        """
        Generates a complete, cleaned, and formatted text response.

        Handles profanity checks, text continuation, and formatting.
        """
        try:
            st = time.time()

            generated_text = self._generate_response(
                query=sentence,
                response_context=self.context,
                max_length=initial_max_length,
                is_question=is_question,
                name=name
            )

            test_generated_text = generated_text.replace(sentence, "")
            roll_count = 0
            while self._check_profanity(test_generated_text):
                if roll_count >= 2:
                    raise Exception("No generated text without profanities found after 3 rerolls")
                print("[PROFANITY FOUND] rerolling...")
                generated_text = self._generate_response(
                    query=sentence,
                    response_context=self.context,
                    max_length=initial_max_length,
                    is_question=is_question,
                    name=name
                )
                test_generated_text = generated_text.replace(sentence, "")
                roll_count += 1

            if long_response and not generated_text.endswith((".", "?", "!", "<|endoftext|>")):
                print("Text not finished. Generating longer text.")
                clean_initial_text = self._clean_text(generated_text)
                text_continuation = self._generate_response(
                    query=f"{clean_initial_text} ",
                    penalty=20,
                    max_length=continuation_length,
                    response_context=""
                )

                roll_count = 0
                while self._check_profanity(text_continuation):
                    if roll_count >= 2:
                        raise Exception("No generated text without profanities found after 3 rerolls")
                    print("[PROFANITY FOUND] rerolling...")
                    text_continuation = self._generate_response(
                        query=f"{clean_initial_text} ",
                        penalty=20,
                        max_length=continuation_length,
                        response_context=""
                    )
                    roll_count += 1

                generated_text = text_continuation

            generated_text = self._clean_text(generated_text)
            generated_text = profanity.censor(generated_text)

            # Formatting sentences and lines
            split_sentences = sent_tokenize(generated_text)
            if len(split_sentences) > 1 and not split_sentences[-1].endswith((".", "?", "!")):
                split_sentences.pop()

            lines = [" ".join(sent_tokenize(" ".join(split_sentences)))]

            formatted_responses = [{"text": line, "expression": None} for line in lines]

            response_dict = {
                "response": formatted_responses,
                "function": None,
                "args": None
            }

            print(f'[generate_text EXECUTION TIME] {time.time() - st:.2f} seconds')
            return response_dict

        except Exception as e:
            print(f"[generate_text EXCEPTION] {e}")
            for responses_list in self.data["responses"]:
                if responses_list["tag"] == "confused":
                    responses = responses_list['responses']
                    response_dict = {
                        "response": random.choice(responses),
                        "function": responses_list["function"],
                        "args": responses_list["args"]
                    }
                    return response_dict


if __name__ == "__main__":

    # You to I test cases
    you_to_i_test_cases = [
        "who are you", "who made you", "what do you think of you",
        "are you ok", "Do you like ice cream?", "can you tell me who you are",
        "can you tell me who I am"
    ]

    text_generator = TextGenerator()

    print("--- Testing you_to_i function ---")
    for case in you_to_i_test_cases:
        print(f"Before: {case}\nAfter: {text_generator._you_to_i(case)}\n")

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