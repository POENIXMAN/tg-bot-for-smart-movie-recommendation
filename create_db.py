import sqlite3
import pandas as pd

# Create a connection to the database
conn = sqlite3.connect("movies_bot.db")

# Create Movies table
conn.execute("""
CREATE TABLE IF NOT EXISTS Movies (
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    ReleaseDate DATE,
    Title TEXT NOT NULL,
    Overview TEXT,
    Popularity REAL,
    VoteCount INTEGER,
    VoteAverage REAL,
    OriginalLanguage TEXT,
    PosterUrl TEXT,
    UpdatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

# Genres Table
conn.execute("""
CREATE TABLE IF NOT EXISTS Genres (
    GenreID INTEGER PRIMARY KEY AUTOINCREMENT,
    GenreName TEXT UNIQUE NOT NULL
);
""")

# MovieGenres Table
conn.execute("""
CREATE TABLE IF NOT EXISTS MovieGenres (
    MovieID INTEGER NOT NULL,
    GenreID INTEGER NOT NULL,
    FOREIGN KEY (MovieID) REFERENCES Movies(MovieID),
    FOREIGN KEY (GenreID) REFERENCES Genres(GenreID),
    PRIMARY KEY (MovieID, GenreID)
);
""")

# Users Table
conn.execute("""
CREATE TABLE IF NOT EXISTS Users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    TelegramID INTEGER UNIQUE NOT NULL,
    Name TEXT
);
""")

# Ratings Table
conn.execute("""
CREATE TABLE IF NOT EXISTS Ratings (
    RatingID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER NOT NULL,
    MovieID INTEGER NOT NULL,
    Rating INTEGER,
    Status TEXT CHECK(Status IN ('rated', 'skipped', 'watch_later')) NOT NULL DEFAULT 'rated',
    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID),
    FOREIGN KEY (MovieID) REFERENCES Movies(MovieID),
    UNIQUE (UserID, MovieID)
);
""")

# WatchLater Table
conn.execute("""
CREATE TABLE IF NOT EXISTS WatchLater (
    UserID INTEGER NOT NULL,
    MovieID INTEGER NOT NULL,
    AddedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID),
    FOREIGN KEY (MovieID) REFERENCES Movies(MovieID),
    PRIMARY KEY (UserID, MovieID)
);
""")



conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_popularity ON Movies (Popularity);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user_movie ON Ratings (UserID, MovieID);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_users_telegramid ON Users (TelegramID);")

# Load the CSV file
movies_csv = pd.read_csv("mymoviedb.csv")

# Default poster URL
default_poster_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/No_poster.svg/411px-No_poster.svg.png"

cursor = conn.cursor()

# Function to insert genres and return their IDs
def get_or_insert_genre(genre_name):
    # Check if the genre already exists
    cursor.execute("SELECT GenreID FROM Genres WHERE GenreName = ?", (genre_name,))
    result = cursor.fetchone()
    
    if result:
        return result[0]  # Return existing GenreID
    else:
        # Insert the new genre
        cursor.execute("INSERT INTO Genres (GenreName) VALUES (?)", (genre_name,))
        return cursor.lastrowid  # Return the new GenreID

for _, row in movies_csv.iterrows():
    # Handle the poster URL
    poster_url = row['Poster_URL']
    if poster_url == "Failed to retrieve data.":
        poster_url = default_poster_url
    
    # Insert the movie into the Movies table
    cursor.execute("""
    INSERT INTO Movies (ReleaseDate, Title, Overview, Popularity, VoteCount, VoteAverage, OriginalLanguage, PosterUrl)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        row['Release_Date'], 
        row['Title'], 
        row['Overview'], 
        row['Popularity'], 
        row['Vote_Count'], 
        row['Vote_Average'], 
        row['Original_Language'], 
        poster_url
    ))
    
    # Get the MovieID of the inserted movie
    movie_id = cursor.lastrowid
    
    # Split the genres and insert them into the Genres and MovieGenres tables
    genres = row['Genre'].split(", ")  # Assuming genres are comma-separated
    print(f"Movie: {row['Title']}, Genres: {genres}")  # Debugging: Print genres
    
    for genre_name in genres:
        genre_id = get_or_insert_genre(genre_name)
        
        # Insert the relationship into MovieGenres
        cursor.execute("""
        INSERT INTO MovieGenres (MovieID, GenreID)
        VALUES (?, ?);
        """, (movie_id, genre_id))
        print(f"Inserted: MovieID={movie_id}, GenreID={genre_id}") 

# Commit the changes and close the connection
conn.commit()
conn.close()