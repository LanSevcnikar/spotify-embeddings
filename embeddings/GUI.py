import numpy as np
import tkinter as tk
import json

# Create the main window
window = tk.Tk()
window.title("Music Search")

# set width and height for the window
window.geometry("1000x600")

all_songs = []

track_uri_to_track_name = json.load(
    open('weights/track_uri_to_track_name.json', 'r'))
track_uri_to_artist_name = json.load(
    open('weights/track_uri_to_artist_name.json', 'r'))
track_uri_to_album_name = json.load(
    open('weights/track_uri_to_album_name.json', 'r'))

def get_all_songs():

    for track_uri, track_name in track_uri_to_track_name.items():
        artist_name = track_uri_to_artist_name[track_uri]
        album_name = track_uri_to_album_name[track_uri]

        all_songs.append({
            'track_name': track_name,
            'artist_name': artist_name,
            'album_name': album_name,
            'track_uri': track_uri
        })


get_all_songs()


def set_table(songs_to_display=[]):
    for widget in table_frame.winfo_children():
        widget.destroy()

    song_title_label = tk.Label(table_frame, text="Song Title")
    song_title_label.grid(row=0, column=0)
    song_title_label.config(width=30)

    song_artist_label = tk.Label(table_frame, text="Artist")
    song_artist_label.grid(row=0, column=1)
    song_artist_label.config(width=30)

    song_album_label = tk.Label(table_frame, text="Album")
    song_artist_label.grid(row=0, column=2)
    song_artist_label.config(width=30)

    song_id_label = tk.Label(table_frame, text="Song Id")
    song_id_label.grid(row=0, column=3)
    song_id_label.config(width=30)

    for i, song in enumerate(songs_to_display):
        song_title = tk.Label(table_frame, text=song['track_name'])
        song_title.grid(row=i+1, column=0)
        song_title.config(width=30)

        song_artist = tk.Label(table_frame, text=song['artist_name'])
        song_artist.grid(row=i+1, column=1)
        song_artist.config(width=30)

        song_album = tk.Label(table_frame, text=song['album_name'])
        song_album.grid(row=i+1, column=2)
        song_album.config(width=30)

        song_id = tk.Label(table_frame, text=song['track_uri'])
        song_id.grid(row=i+1, column=3)
        song_id.config(width=30)
        song_id.bind("<Button-1>", copy_song_id)

# Define functions for later use (placeholders for now)


def search_music():
    search_song_title = title_entry.get()
    search_artist = artist_entry.get()
    search_album = album_entry.get()

    songs_to_display = []

    for song in all_songs:
        if search_song_title.lower() in song['track_name'].lower():
            if search_artist.lower() in song['artist_name'].lower():
                if search_album.lower() in song['album_name'].lower():
                    songs_to_display.append(song)

    set_table(songs_to_display[:8])


def copy_song_id(event):
    # copy song id to clipboard
    window.clipboard_clear()
    window.clipboard_append(event.widget.cget("text"))
    window.update()

    # set song_uri_entry to the song id
    song_uri_entry.delete(0, tk.END)
    song_uri_entry.insert(0, event.widget.cget("text"))


# Create search labels and entry fields
# Do not use grid() for these widgets
# Instead, use pack() to place them in the window


title_label = tk.Label(window, text="Song Title")
title_label.pack()

title_entry = tk.Entry(window)
title_entry.pack()

artist_label = tk.Label(window, text="Artist")
artist_label.pack()

artist_entry = tk.Entry(window)
artist_entry.pack()

album_label = tk.Label(window, text="Album")
album_label.pack()

album_entry = tk.Entry(window)
album_entry.pack()

search_button = tk.Button(window, text="Search", command=search_music)
search_button.pack()

# Create a frame to hold the table
table_frame = tk.Frame(window)
table_frame.pack()

# Create a table with 4 columns
# The first row will be the column headers
# The other rows will be populated with search results
# The table will be placed in the table_frame

# Column headers (to be added first)
#
# Song Title | Artist | Album | Song Id
#

table_header = tk.Label(table_frame, text="Song Title")
table_header.grid(row=0, column=0)
table_header.config(width=40)

table_header = tk.Label(table_frame, text="Artist")
table_header.grid(row=0, column=1)
table_header.config(width=40)

table_header = tk.Label(table_frame, text="Album")
table_header.grid(row=0, column=2)
table_header.config(width=40)

table_header = tk.Label(table_frame, text="Song Id")
table_header.grid(row=0, column=3)
table_header.config(width=40)

separator = tk.Label(
    table_frame, text="-------------------------------------------------")

# load metadata.tsv and vectors.tsv
metadata = np.load('weights/vocab.npy')
vectors = np.load('weights/weights.npy')

song_to_index = {}
index_to_song = {}

for index, song in enumerate(metadata):
    song_uri = song
    song_to_index[song_uri] = index
    index_to_song[index] = song_uri


def find_most_similar_songs(song_uri, top_n=30):
    song_index = song_to_index[song_uri]
    song_vector = vectors[song_index]

    # calculate cosine similarity
    cosine_similarities = np.dot(
        vectors, song_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(song_vector))

    # get top n songs
    top_n_indices = np.argsort(cosine_similarities)[::-1][:top_n]

    top_n_songs = [index_to_song[index] for index in top_n_indices]

    return top_n_songs


def set_table_2(song_uri):
    song_name = track_uri_to_track_name[song_uri]
    song_artist = track_uri_to_artist_name[song_uri]
    song_album = track_uri_to_album_name[song_uri]

    # set song_simm_search_title
    song_simm_search_title.config(
        text=f"Similar Songs to {song_name} by {song_artist} from the album {song_album}")
    

    similar_songs = find_most_similar_songs(song_uri)
    songs_to_display = [{'track_name': track_uri_to_track_name[song], 'artist_name': track_uri_to_artist_name[song],
                         'album_name': track_uri_to_album_name[song]} for song in similar_songs]

    for widget in table_frame2.winfo_children():
        widget.destroy()

    song_title_label2 = tk.Label(table_frame2, text="Song Title")
    song_title_label2.grid(row=0, column=0)
    song_title_label2.config(width=30)

    song_artist_label2 = tk.Label(table_frame2, text="Artist")
    song_artist_label2.grid(row=0, column=1)
    song_artist_label2.config(width=30)

    song_album_label2 = tk.Label(table_frame2, text="Album")
    song_artist_label2.grid(row=0, column=2)
    song_artist_label2.config(width=30)

    for i, song in enumerate(songs_to_display):
        song_title = tk.Label(table_frame2, text=song['track_name'])
        song_title.grid(row=i+1, column=0)
        song_title.config(width=40)

        song_artist = tk.Label(table_frame2, text=song['artist_name'])
        song_artist.grid(row=i+1, column=1)
        song_artist.config(width=40)

        song_album = tk.Label(table_frame2, text=song['album_name'])
        song_album.grid(row=i+1, column=2)
        song_album.config(width=40)


song_uri_label = tk.Label(window, text="Song URI")
song_uri_label.pack()

song_uri_entry = tk.Entry(window)
song_uri_entry.pack()

song_simm_search_title = tk.Label(window, text="")
song_simm_search_title.pack()

song_sim_search = tk.Button(window, text="Search Similar Songs",
                            command=lambda: set_table_2(song_uri_entry.get()))
song_sim_search.pack()

table_frame2 = tk.Frame(window)
table_frame2.pack()
table_header2 = tk.Label(table_frame2, text="Song Title")
table_header2.grid(row=0, column=0)
table_header2.config(width=40)
table_header2 = tk.Label(table_frame2, text="Artist")
table_header2.grid(row=0, column=1)
table_header2.config(width=40)
table_header2 = tk.Label(table_frame2, text="Album")
table_header2.grid(row=0, column=2)
table_header2.config(width=40)

window.mainloop()
