# Importing Libraries
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import gradio as gr


# Books Files
popular_books_df = pickle.load(open('books-recommendation/popular.pkl','rb'))
pt = pickle.load(open('books-recommendation/pt.pkl','rb'))
books = pickle.load(open('books-recommendation/books.pkl','rb'))
similarity_scores = pickle.load(open('books-recommendation/similarity_score.pkl','rb'))

# Movies Files
movies_dict = pickle.load(open('movie-recommendation/movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('movie-recommendation/similarity.pkl','rb'))

# Music Files

# Initialize embedding model for music
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB for music
chroma_client = chromadb.PersistentClient(path="music-recommendation/chroma_db/")
collection = chroma_client.get_or_create_collection(name="music_recommendations")

# Recommendations Functions

## Popular Books Function
popular_books = [list(popular_books_df['Book-Title'].values),
                  list(popular_books_df['Book-Author'].values),
                  list(popular_books_df['Image-URL-M'].values),
                  list(popular_books_df['num_ratings'].values),
                  list(popular_books_df['avg_rating'].values)]

def popular_books_display():
    html = "<h1>Top 100 Books in database </h1> <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; max-width: 1400px; margin: auto;'>"
    for idx in range(len(popular_books[0])):
        html += f"""
        <div>
            <img src="{popular_books[2][idx]}" style="max-width: 150px; height: auto;">
            <p><b>{popular_books[0][idx]}</b></p>
            <p><i>By {popular_books[1][idx]}</i></p>
            <p>Num of ratinngs: {popular_books[3][idx]}</p>
            <p>Avg ratinng: {popular_books[4][idx]}</p>
            <br>
        </div>
        """
        # print(books[0][idx], ", ",books[1][idx], ", ",books[2][idx], ", ",books[3][idx], ", ",books[4][idx])
    html += "</div>"
    return html

## Books Recommendations Functions
def books_recommend(user_input):
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]

        item = {}
        item = {
            "Title": temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
            "Author": temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
            "Thumbnail": temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values[0],
        }
        data.append(item)

    return data

def books_recommend_display(query):
    books = books_recommend(query)

    html = "<h2>Recommended Books</h2> <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; max-width: 1400px; margin: auto;'>"
    for book in books:
        html += f"""
        <div>
            <img src="{book['Thumbnail']}" style="max-width: 150px; height: auto;">
            <p><b>{book['Title']}</b></p>
            <p><i>By {book['Author']}</i></p>
            <br>
        </div>
        """
    html += "</div>"

    return html

# Movies Recommendations Function

def recommend_movies(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]
    recommend_movies = []

    for i in movie_list:
        item = {"Thumbnail":movies.iloc[i[0]].Thumbnail ,"Title":movies.iloc[i[0]].title}
        recommend_movies.append(item)

    return recommend_movies


def diplay_recommended_movies(user_input):
    recommended_movies = recommend_movies(user_input)

    html = "<h2>Recommended Movies</h2> <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; max-width: 1400px; margin: auto;'>"
    for movies in recommended_movies:
        html += f"""
        <div>
            <img src="{movies["Thumbnail"]}" style="max-width: 150px; height: auto;">
            <p><b>{movies["Title"]}</b></p>
        </div>
        """
    html += "</div>"

    return html

# Music Recommendations Function

def recommend_music(prompt, emotion=None, top_k=16):
    # Encode user query
    query_embedding = embedding_model.encode(prompt).tolist()

    # Search ChromaDB for similar songs
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Filter results by emotion if specified
    recommendations = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if emotion is None or meta["Emotion"] == emotion:
            recommendations.append({"Track": meta["Track"], "Artist": meta["Artist"], "Url_spotify": meta["Url_spotify"], "Url_youtube": meta["Url_youtube"], "Thumbnail": meta["Thumbnail"] })

    return recommendations


def display_music_recommend(user_query, user_emotion):
    musics = recommend_music(user_query, user_emotion)

    html = "<h2>Recommended Musics</h2> <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; max-width: 1400px; margin: auto;'>"
    html += """
            <style>
                #links a{
                    font-size: 15px;
                    width : 200px;
                    border-radius: 5px;
                    
                    text-decoration: none;
                    padding: 1px 1px;  
                }

                #links a:hover {
                    color: #FFFFFF;
                    background-color: #F16924;
                }
         </style> """

    for music in musics:
        html += f"""<div>
            <img src="{music['Thumbnail']}" style="max-width: 150px; max-height=220px;">
            <p><b>{music['Track']}</b></p>
            <p><i>Artists: {music['Artist']}</i></p>
            <ul id=links>
                <li><a class="link" href={music['Url_youtube']} target="_blank" rel="noopener noreferrer">Youtube Link </a> </li>
                <li><a class="link" href={music['Url_spotify']} target="_blank" rel="noopener noreferrer"> Spotify Link</a></li>
            </ul>
        </div>
        """
    html += "</div>"

    return html

# Documentation Fuction:
def documentation_display():
    html = """
        <style>
            #docs ul {
                list-style-position: inside; /* Aligns bullets correctly */
                padding-left: 20px;
            }
        
            #docs li {
                margin-bottom: 10px; /* Adds spacing between items */
            }
        
            #docs a {
                display: inline-block; /* Prevents full-width issue */
                font-size: 20px;
                width : 650px;
                border-radius: 10px;
                color: white;
                text-decoration: none;
                padding: 5px 10px;
            }
        
            #docs a:hover {
                color: white;
                background-color: #F16924;
            }
     </style>
    
        <nav id="docs">
            <ul>
                <li><a href="https://www.linkedin.com/in/saurabh-maulekhi-326584241" target="_blank" rel="noopener noreferrer"><b>Linkedin Id</b></a></li> 
                <li><a href="https://www.kaggle.com/code/saurabhmaulekhi/movie-recommendation-system" target="_blank" rel="noopener noreferrer"><b>Movie Recommendation Deployment (Kaggle Notebook)</b></a></li>
                <li><a href="https://www.kaggle.com/code/saurabhmaulekhi/book-recommendations" target="_blank" rel="noopener noreferrer"><b>Books Recommendation Deployment (Kaggle Notebook)</b></a></li> 
                <li><a href="https://www.kaggle.com/code/saurabhmaulekhi/music-recommendation" target="_blank" rel="noopener noreferrer"><b>Music Recommendation Deployment (Kaggle Notebook)</b></a></li>
                <li><a href="https://github.com/Saurabh-Maulekhi/LLM-and-Similarity-Based-Media-Recommendation-System" target="_blank" rel="noopener noreferrer"><b>Github Repo of This Web App</b></a></li> 
            </ul>
        </nav>
    """
    return html

# Lists for the Dropdown

# books List
books_list = pt.index.tolist()

# movies List
movies_list = list(movies['title'])

# emotions List
music  = pd.read_csv('music-recommendation/music_dataset_with_emotions.csv')
emotions_list = [None]
emotions_list += list(music['Emotion'].unique())

# gradio app examples
Music_examples = [["Old Hindi Songs",None], ["Happy Songs", "caring"]]
Books_examples = [["Four Blind Mice"],["Good Omens"]]
Movies_examples = [["Spider-Man 3"],["Pirates of the Caribbean: At World's End"]]

# Gradio App

with gr.Blocks() as app:
    gr.Markdown("# LLM and Similarity Based Media Recommendation System")
    gr.Markdown()
    gr.Markdown("## A Recommender for Music, Movies and Books")
    gr.Markdown()
    gr.Markdown()

    with gr.Tab("Music Recommender"):
        gr.Markdown("### This is Based on LLM")
        user_query = gr.Textbox(placeholder="Your Prompt, e.g. A Party Song")
        user_emotion = gr.Dropdown(emotions_list, label="Emotional Tone of Music (optional)", interactive=True)
        submit_button = gr.Button("Submit")

        output = gr.HTML()
        submit_button.click(fn=display_music_recommend,
                            inputs=[user_query, user_emotion],
                            outputs=output,
                            )
        # Adding examples
        gr.Examples(
            examples=Music_examples,
            inputs=[user_query, user_emotion],
            outputs=output,
            fn=display_music_recommend  # This ensures auto-submit on example click
        )

    with gr.Tab("Book Recommender"):

        with gr.Tab("Top Books"):
            gr.Markdown("### This is Based on Ranking and Ratings")
            gr.HTML(popular_books_display())

        with gr.Tab("Books Recommender"):
            gr.Markdown("### This is Based on Cosine Similarity")
            user_input = gr.Dropdown(books_list, label="Books", interactive=True)

            submit_button = gr.Button("Submit")
            output = gr.HTML()
            submit_button.click(fn=books_recommend_display,
                                inputs=user_input,
                                outputs=output)

            # Adding examples
            gr.Examples(
                examples=Books_examples,
                inputs=user_input,
                outputs=output,
                fn=books_recommend_display  # This ensures auto-submit on example click
            )

    with gr.Tab("Movies Recommender"):
        gr.Markdown("### This is Based on Cosine Similarity")
        user_input = gr.Dropdown(movies_list, label="Movies", interactive=True)
        submit_button = gr.Button("Submit")

        output = gr.HTML()

        submit_button.click(fn=diplay_recommended_movies,
                            inputs=user_input,
                            outputs=output)

        # Adding examples
        gr.Examples(
            examples=Movies_examples,
            inputs=user_input,
            outputs=output,
            fn=diplay_recommended_movies  # This ensures auto-submit on example click
        )

    with gr.Tab("Documentation"):
        gr.HTML(documentation_display())

app.launch(debug=True)