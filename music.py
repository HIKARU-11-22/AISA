import vlc
class MusicPlayer:
    def __init__(self):
        self.player = None
        self.is_playing = False
        self.is_paused = False
        self.track_list = []
        self.current_track_index = -1
        self.playlists = {
            # to make a playlist, put the file path of music and make a playlist.
            'playlist_1': [
                "/home/hikaru/Music/Mrs. GREEN APPLE - インフェルノ（Inferno）.mp3",
                # "/home/hikaru/Music/Track2.mp3",
                # "/home/hikaru/Music/Track3.mp3"
            ],
            'playlist_2': [
                # "/home/hikaru/Music/Track4.mp3",
                # "/home/hikaru/Music/Track5.mp3",
                # "/home/hikaru/Music/Track6.mp3"
            ]
        }

    def load_playlist(self, playlist_name):
        if playlist_name in self.playlists:
            self.track_list = self.playlists[playlist_name]
            self.current_track_index = 0
            print(f"Loaded playlist: {playlist_name}")
        else:
            print(f"Playlist '{playlist_name}' not found!")
            return False  # Return False if the playlist doesn't exist
        return True 
    
    def load_track(self, track):
        if self.player:
            self.player.stop()  # Stop previous track
        self.player = vlc.MediaPlayer(track)
        self.is_playing = False
        self.is_paused = False

    def play(self):
        if not self.is_playing and self.current_track_index < len(self.track_list):
            track = self.track_list[self.current_track_index]
            self.load_track(track)
            self.player.play()
            self.is_playing = True
            self.is_paused = False
            print(f"Playing music: {track}")

    def pause(self):
        if self.is_playing and not self.is_paused:
            self.player.pause()
            self.is_paused = True
            print("Music paused.")

    def resume(self):
        if self.is_playing and self.is_paused:
            self.player.play()
            self.is_paused = False
            print("Music resumed.")

    def next_track(self):
        if self.current_track_index + 1 < len(self.track_list):
            self.current_track_index += 1
            self.play()
            print(f"Playing next track: {self.track_list[self.current_track_index]}")
        else:
            print("No more tracks in the playlist.")
            return False # to return false if the next track is not available
        return True

    def play_song(self, song_name):
        for idx, track in enumerate(self.track_list):
            if song_name.lower() in track.lower():  # Simple matching by name
                self.current_track_index = idx
                self.play()
                return
        print("Song not found in the current playlist.")

# Create an instance of MusicPlayer
music_player = MusicPlayer()

def play_music(track="default.mp3"):
    music_player.load_track(track)
    music_player.play()
