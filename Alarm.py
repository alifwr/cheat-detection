import soundfile as sf
import sounddevice as sd

class Alarm:
    def __init__(self):
        self.aud, self.fs = sf.read('alarm.wav', dtype='float32')

    def start(self):
        sd.play(self.aud, self.fs)
    
    def stop(self):
        sd.stop()