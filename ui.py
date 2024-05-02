import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import base64
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# Load travel posts data from CSV
travel_posts = pd.read_csv("image_dataset.csv", encoding='latin1')

def get_locations():
    return travel_posts["location"].unique()

def get_hashtags():
    hashtags = set()
    for tags in travel_posts["hashtag"]:
        hashtags.update(tags.split(", "))
    return sorted(list(hashtags))

def recommend_posts_hashtag(location, hashtags):
    # Handle empty hashtags
    if not hashtags:
        return pd.DataFrame()

    recommended_posts = []
    for _, post in travel_posts.iterrows():
        if location == post["location"]:
            common_hashtags = set(post["hashtag"].split(", ")) & set(hashtags)
            score = len(common_hashtags)
            if score > 0:
                post['score'] = score  # Add the score to the recommendation
                recommended_posts.append(post)

    return pd.DataFrame(recommended_posts)

def recommend_posts_knn(location, hashtag):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(travel_posts[['location', 'hashtag']])
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(encoded_features)

    user_input_df = pd.DataFrame({
        'location': [location],
        'hashtag': [hashtag]
    })

    encoded_user_input = encoder.transform(user_input_df)

    distances, indices = knn.kneighbors(encoded_user_input)

    recommendations = travel_posts.iloc[indices[0]].reset_index(drop=True)
    recommendations['score'] = 1 / (1 + distances[0])  # Adding score column
    return recommendations

def image_to_base64(image):
    buffered = BytesIO()
    image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    # Encode the bytes object to base64
    encoded_img = base64.b64encode(buffered.getvalue())
    # Convert the encoded bytes to a string
    return encoded_img.decode('utf-8')

st.title("Travel Recommendation App")

# Select recommendation algorithm
algorithm = st.selectbox("Select Recommendation Algorithm:", ["Hashtag-Based", "KNN-Based"])

# Get user input for location and hashtags
location = st.selectbox("Select Location:", options=get_locations())
hashtags = st.multiselect("Select Hashtags:", options=get_hashtags())

# Display the recommendation button
if st.button("Recommend"):
    # Call the recommendation function based on selected algorithm
    if algorithm == "Hashtag-Based":
        recommendations = recommend_posts_hashtag(location, hashtags)
    else:
        recommendations = recommend_posts_knn(location, hashtags[0])  # Using the first hashtag for KNN

    # Check if recommendations are available
    if not recommendations.empty:
        st.subheader("Recommendations:")
        num_recommendations = len(recommendations)
        num_rows = (num_recommendations + 2) // 3  # Calculate number of rows needed
        for i in range(num_rows):
            row_html = "<div style='display:flex; justify-content:center;'>"
            for j in range(3):
                index = i * 3 + j
                if index < num_recommendations:
                    recommendation = recommendations.iloc[index]
                    # Display the image from GitHub repository using the provided URL
                    image_url = recommendation['image_url']
                    # Modify the URL to the correct format
                    full_image_url = f"https://github.com/limwengni/travelpostrecommender/raw/main/{image_url}"

                    try:
                        response = requests.get(full_image_url)
                        if response.status_code == 200:
                            # Display the image with title above
                            st.markdown(f"<div style='text-align:center'><h2>{recommendation['image_title']}</h2></div>", unsafe_allow_html=True)
                            st.image(full_image_url, caption=f"Similarity Score: {recommendation['score']}")

                            # Display location and hashtags in small boxes
                            st.markdown(f"<div style='text-align:center; margin-top: 5px;'>"
                                        f"<div style='background-color: lightblue; padding: 5px; border-radius: 5px; margin-right: 10px; width: 150px; display:inline-block;'>{recommendation['location']}</div>"
                                        f"<div style='background-color: lightgreen; padding: 5px; border-radius: 5px; width: 150px; display:inline-block;'>{' '.join(['#' + tag for tag in recommendation['hashtag'].split(', ')])}</div>"
                                        f"</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.write(f"Error loading image from URL: {full_image_url}")
                        st.write(e)

            row_html += "</div>"
            st.markdown(row_html, unsafe_allow_html=True)

    else:
        st.write("No recommendations found based on your input.")
