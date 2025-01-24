from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler,CallbackContext, CallbackQueryHandler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sqlite3,joblib



data = joblib.load('data.pkl')
features = joblib.load('features.pkl')

def get_db_connection():
    conn = sqlite3.connect("movies_bot.db")
    conn.row_factory = sqlite3.Row
    return conn

def register_user(telegram_id, name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE TelegramID = ?", (telegram_id,))
    user = cursor.fetchone()
    if user is None:
        cursor.execute("INSERT INTO Users (TelegramID, Name) VALUES (?, ?)", (telegram_id, name))
        conn.commit()
    conn.close()



def recommend_movies_reranked(features, user_preferences, data, top_n=10, alpha=0.5):
    """
    Recommend movies using k-NN and re-rank based on diversity.

    :param features: Feature matrix (e.g., numerical and one-hot encoded data).
    :param user_preferences: Array representing user preferences for features.
    :param data: Original metadata for movies (e.g., title, popularity).
    :param top_n: Number of recommendations to return.
    :param alpha: Weight for similarity vs diversity.
    :return: DataFrame of top_n recommended movies.
    """

    knn = NearestNeighbors(n_neighbors=top_n * 2, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(user_preferences)

    recommended_movies = data.iloc[indices[0]].copy()
    recommended_movies['Similarity'] = 1 - distances[0]

    # Calculate Diversity Score (inverse of Popularity, normalized)
    recommended_movies['Diversity'] = 1 - (recommended_movies['Popularity'] / recommended_movies['Popularity'].max())
    
    genre_columns = [col for col in features.columns if col in data.columns and data[col].max() == 1]
    
    # add Genre Diversity
    genre_features = features[genre_columns]
    genre_similarity = cosine_similarity(
        genre_features.iloc[indices[0]], genre_features.iloc[indices[0]]
    ).mean(axis=1)

    recommended_movies['Genre_Diversity'] = 1 - genre_similarity
    
    recommended_movies['Diversity'] = (
        0.5 * (1 - (recommended_movies['Popularity'] / recommended_movies['Popularity'].max())) +
        0.5 * recommended_movies['Genre_Diversity']
    )
    
    recommended_movies['Final_Score'] = (
        alpha * recommended_movies['Similarity'] +
        (1 - alpha) * recommended_movies['Diversity']
    )

    recommended_movies = recommended_movies.sort_values(by='Final_Score', ascending=False)

    return recommended_movies.head(top_n)

def recommend_movies_knn(features, user_preferences, data, top_n=10):
    """
    Recommend movies based on user preferences using k-Nearest Neighbors.

    :param features: Feature matrix (e.g., numerical and one-hot encoded data).
    :param user_preferences: Array representing user preferences for features.
    :param data: Original metadata for movies (e.g., title, genre).
    :param top_n: Number of recommendations to return.
    :return: DataFrame of top_n recommended movies.
    """
    knn = NearestNeighbors(n_neighbors=top_n, metric='cosine')
    knn.fit(features)

    distances, indices = knn.kneighbors(user_preferences)

    recommended_movies = data.iloc[indices[0]]

    recommended_movies = recommended_movies.copy()
    recommended_movies['Similarity'] = 1 - distances[0]

    return recommended_movies.sort_values(by='Similarity', ascending=False)

def get_user_recommendations(telegram_id, conn, features, data, top_n=10):
    """
    Get movie recommendations for a user based on their ratings in the database.

    :param telegram_id: The Telegram ID of the user.
    :param conn: SQLite database connection.
    :param features: Feature matrix (e.g., numerical and one-hot encoded data).
    :param data: Original metadata for movies (e.g., title, genre).
    :param top_n: Number of recommendations to return.
    :return: DataFrame of top_n recommended movies.
    """
    # Fetch user ratings from the database
    query = """
    SELECT r.MovieID, r.Rating
    FROM Ratings r
    WHERE r.UserID = ?
    AND r.Status = 'rated'
    """   
    user_ratings = pd.read_sql_query(query, conn, params=(telegram_id,))
    
    if user_ratings.empty:
        print("No ratings found for the user.")
        return pd.DataFrame()
    # Исключаем фильмы с недопустимыми статусами
    excluded_query = """
    SELECT MovieID FROM Ratings
    WHERE UserID = ? AND Status IN ('rated', 'skipped', 'watch_later')
    """
    excluded_movies = pd.read_sql_query(excluded_query, conn, params=(telegram_id,))['MovieID'].values
    features = features.drop(index=excluded_movies - 1, errors='ignore')  # Убираем уже оцененные фильмы

    # Get the indices of the rated movies
    rated_indices = user_ratings['MovieID'].values - 1  

    # Get the ratings
    ratings = user_ratings['Rating'].values

    # Compute user preferences
    rated_features = features.iloc[rated_indices]
    user_preferences = np.dot(rated_features.T, ratings)
    user_preferences = user_preferences / np.linalg.norm(user_preferences)
    user_preferences = pd.DataFrame([user_preferences], columns=features.columns)

    # Get recommendations
    recommendations = recommend_movies_reranked(features, user_preferences, data, top_n)

    return recommendations


async def get_random_movie(user_id, is_initial):
    conn = get_db_connection()
    cursor = conn.cursor()

    number_of_movies = 10 
    attempts_to_recommend_new_movies = 0
    
    if is_initial:
        cursor.execute("""
            SELECT * FROM Movies 
            WHERE MovieID <= 50 AND MovieID NOT IN (
                SELECT MovieID FROM Ratings WHERE UserID = ? AND Status IN ('rated', 'skipped', 'watch_later')
            )
            ORDER BY RANDOM() LIMIT 1
""", (user_id,))
        movie = cursor.fetchone()

    else:
        while True:
            recommendations = get_user_recommendations(user_id, conn, features, data, top_n=number_of_movies)
            print(recommendations)
            if recommendations.empty:
                movie = None
                break

            for i in range(len(recommendations)):
                movie = recommendations.iloc[i]
                movie_id = int(movie.name) + 1  # pandas starts at 0, db at 1
                cursor.execute("""
                    SELECT * FROM Ratings WHERE UserID = ? AND MovieID = ? AND Status IN ('rated', 'skipped', 'watch_later')
                """, (user_id, movie_id))
                if cursor.fetchone() is None:
                    # Movie not rated, skipped, or in watch later, use this one
                    cursor.execute("""
                        SELECT * FROM Movies WHERE MovieID = ?
                    """, (movie_id,))
                    movie = cursor.fetchone() 
                    break
            else:
                attempts_to_recommend_new_movies+=1
                if attempts_to_recommend_new_movies > 3:
                    number_of_movies += 10
                    attempts_to_recommend_new_movies = 0 
                continue
            break

    conn.close()    
    return movie



async def show_movie(user_id, message, context: CallbackContext, is_initial=True):
    movie = await get_random_movie(user_id, is_initial)
    context.user_data["current_movie_id"] = movie["MovieID"]

    if not is_initial and not context.user_data.get("congrats_sent", False):
        await message.reply_text(
            "Отличная работа! 🎉 Вы оценили 10 фильмов. Теперь мы начнем рекомендовать фильмы на основе ваших предпочтений!"
        )
        context.user_data["congrats_sent"] = True 
        
    movie_title = movie["Title"]
    movie_genre = movie["Genre"]
    movie_overview = movie["Overview"]
    movie_release = movie["ReleaseDate"]
    movie_poster = movie["PosterUrl"]
    vote_average = movie["VoteAverage"]

    movie_message = f"🎬 <b>Название:</b> {movie_title}\n"
    movie_message += f"⭐ <b>Рейтинг:</b> {vote_average}\n"
    movie_message += f"🎭 <b>Жанр:</b> {movie_genre}\n"
    movie_message += f"📅 <b>Дата выхода:</b> {movie_release}\n"
    movie_message += f"📖 <b>Описание:</b> {movie_overview}"

    keyboard = [
        [InlineKeyboardButton("⭐1", callback_data="1"),
        InlineKeyboardButton("⭐2", callback_data="2"),
        InlineKeyboardButton("⭐3", callback_data="3"),
        InlineKeyboardButton("⭐4", callback_data="4"),
        InlineKeyboardButton("⭐5", callback_data="5")],
        [InlineKeyboardButton("Пропустить ⏭️", callback_data="skip"),
        InlineKeyboardButton("Посмотреть позже ⏳", callback_data="watch_later")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_photo(
            photo=movie_poster,
            caption=movie_message,
            reply_markup=reply_markup,
            parse_mode="HTML"
        )

async def handle_button(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    print(f"handle_button - user id: {user_id}")
    movie_id = context.user_data.get("current_movie_id") 
    status = query.data

    conn = get_db_connection()
    cursor = conn.cursor()

    if status.isdigit():
        rating = int(status)
        cursor.execute("""
            INSERT INTO Ratings (UserID, MovieID, Rating, Status)
            VALUES (?, ?, ?, ?)
        """, (user_id, movie_id, rating, "rated"))
        conn.commit()
    elif status == "skip":
        cursor.execute("""
            INSERT INTO Ratings (UserID, MovieID, Rating, Status)
            VALUES (?, ?, ?, ?)
        """, (user_id, movie_id, None, "skipped"))
        conn.commit()
    elif status == "watch_later":
        cursor.execute("""
            INSERT INTO Ratings (UserID, MovieID, Rating, Status)
            VALUES (?, ?, ?, ?)
        """, (user_id, movie_id, None, "watch_later"))
        conn.commit()

    cursor.execute("SELECT COUNT(*) FROM Ratings WHERE UserID = ? AND Status = 'rated'", (user_id,))
    ratings_count = cursor.fetchone()[0]
    conn.close()

    is_initial = ratings_count <= 10
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    await show_movie(user_id, query.message, context, is_initial)


async def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_name = update.message.from_user.first_name

    register_user(user_id, user_name)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM Ratings WHERE UserID = (SELECT UserID FROM Users WHERE TelegramID = ?)", (user_id,))
    ratings_count = cursor.fetchone()[0]
    conn.close()

    if ratings_count < 10:
        await update.message.reply_text(
            "Добро пожаловать! 🎉 Чтобы получить рекомендации фильмов, сначала нам нужно, чтобы вы оценили 10 фильмов. "
            "Мы покажем вам несколько фильмов для оценки, и как только вы закончите, мы порекомендуем фильм для просмотра. "
            "Давайте начнем!"
        )
        is_initial = True  
    else:
        is_initial = False 

    await show_movie(user_id, update.message, context, is_initial)

async def restart(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM Ratings WHERE UserID = ?", (user_id,))
    conn.commit()
    conn.close()

    await update.message.reply_text(
        "Ваши оценки были удалены. Вы можете начать заново! 🎬 Чтобы получить рекомендации фильмов, "
        "нам нужно, чтобы вы сначала оценили 10 фильмов. Мы покажем вам несколько фильмов для оценки, "
        "а затем порекомендуем фильм на основе ваших оценок. Давайте начнем!"
    )
    await show_movie(user_id, update.message, context)


def main():
    application = Application.builder().token("7935183232:AAGOt6lFz_sfvsL8aMoTlQtYgcimxvQwA24").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("restart", restart))
    application.add_handler(CallbackQueryHandler(handle_button))
    print("Бот запущен")
    application.run_polling()

if __name__ == '__main__':  
    main()