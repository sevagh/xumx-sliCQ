# disable kivy's native argument parser
import os
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.slider import MDSlider
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout


class DemixApp(MDApp):
    def __init__(self, nb_chunks, **kwargs):
        self.nb_chunks = nb_chunks
        self.updater = None

        super().__init__(**kwargs)

    def build(self):
        self.drums_label = MDLabel(
            text="DRUMS",
            pos_hint={'center_x': 0.66, 'center_y': 0.8},
        )
        self.drums_slider = MDSlider(
            min=0, max=100, value=100, orientation='vertical',
            pos_hint={'center_x': 0.2, 'center_y': 0.5},
            size_hint=(0.1, 0.5),
        )

        self.bass_label = MDLabel(
            text="BASS",
            pos_hint={'center_x': 0.87, 'center_y': 0.8},
        )
        self.bass_slider = MDSlider(
            min=0, max=100, value=100, orientation='vertical',
            pos_hint={'center_x': 0.4, 'center_y': 0.5},
            size_hint=(0.1, 0.5),
        )

        self.vocals_label = MDLabel(
            text="VOCALS",
            pos_hint={'center_x': 1.06, 'center_y': 0.8},
        )
        self.vocals_slider = MDSlider(
            min=0, max=100, value=100, orientation='vertical',
            pos_hint={'center_x': 0.6, 'center_y': 0.5},
            size_hint=(0.1, 0.5),
        )

        self.other_label = MDLabel(
            text="OTHER",
            pos_hint={'center_x': 1.26, 'center_y': 0.8},
        )
        self.other_slider = MDSlider(
            min=0, max=100, value=100, orientation='vertical',
            pos_hint={'center_x': 0.8, 'center_y': 0.5},
            size_hint=(0.1, 0.5),
        )

        # create slider and pass the sound to it
        self.progress_slider = MDSlider(
            min=0, max=self.nb_chunks, value=0,
            pos_hint={'center_x': 0.50, 'center_y': 0.1},
            size_hint=(0.6, 0.1),
            orientation='horizontal',
        )

        # spectrogram box
        self.spectrogram_box = MDBoxLayout(
            pos_hint={'center_x': 0.50, 'center_y': 0.0},
            size_hint=(0.6, 0.1)
        )

        screen = Screen()

        screen.add_widget(self.drums_label)
        screen.add_widget(self.bass_label)
        screen.add_widget(self.vocals_label)
        screen.add_widget(self.other_label)

        screen.add_widget(self.drums_slider)
        screen.add_widget(self.bass_slider)
        screen.add_widget(self.vocals_slider)
        screen.add_widget(self.other_slider)

        screen.add_widget(self.progress_slider)
        screen.add_widget(self.spectrogram_box)

        return screen

    def update_slider(self, frame):
        # update slider
        try:
            self.progress_slider.value = frame
        except AttributeError:
            pass

    def update_spectrogram(self, spec):
        try:
            self.spectrogram_box.clear_widgets()
            self.spectrogram_box.add_widget(spec)
        except AttributeError:
            pass
