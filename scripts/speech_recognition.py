import pyaudio

prefixes = ["hey ahronah",
            "hay ahronah",
            "hey ahoronah",
            "hey arona",
            "hey ahornah",
            "hey arhonah",
            "hey aronah",
            "hey irona",
            "hey rona",
            "hey iroda",
            "ahronah",
            "aro de", "giro de", "piero de",
    "hey irina", "hey iruna", "hey irina", "arizona", "8 arona", "hey arena", "hey arona",
    "hey irona", "hey aruna", "aronia", "heroina", "verona", "aerona", "the erona",
    "hey tyrone", "aro de", "aruna", "hey orona", "crear una", "a tairona", "play irina", "corona"]



prefixes = sorted(prefixes, key=len)

suffixes = ["ahronah", "ahronah?"]

def get_input():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    devices_num = info.get("deviceCount")

    print(f"DEVICE NUM {devices_num}")

    for i in range(0, devices_num):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i))

        if (p.get_device_info_by_host_api_device_index(0, i).get("maxOutputChannels")) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get("name"))

