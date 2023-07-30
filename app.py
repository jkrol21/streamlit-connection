
import boto3
import io
import numpy as np
import streamlit as st
import toml
import torch
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

from PIL import Image
from qdrant_connection import QdrantConnection

@st.cache_data
def load_embedding_model():

    model_ckpt = "nateraw/vit-base-beans"
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    return model, extractor

@st.cache_resource
def load_transformation_chain():
     
     # Data transformation chain.
     transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * extractor.size["height"])),
                T.CenterCrop(extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
            ]
        )
     
     return transformation_chain

conn = st.experimental_connection('qdrant', type=QdrantConnection)
model, extractor = load_embedding_model()
transformation_chain = load_transformation_chain()

@st.cache_resource
def load_s3_bucket():
    with open("./.streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        s3_params = secrets["connections"]["s3"]
        
    s3_resource = boto3.resource('s3', 
                      aws_access_key_id=s3_params["key"], 
                      aws_secret_access_key=s3_params["secret"], 
                      region_name='us-east-1'
                      )
    # load image from s3 using boto3
    beans_bucket = s3_resource.Bucket("beans-data")
    return beans_bucket


# load image from s3 using boto3
beans_bucket = load_s3_bucket()

@st.cache_data
def load_test_image_paths():
    return [object.key for object in beans_bucket.objects.filter(Prefix="test").all() if object.key.endswith(".jpg")]

@st.cache_data
def get_image_from_s3(path):
    object = beans_bucket.Object(path)
    file_stream = io.BytesIO()
    object.download_fileobj(file_stream)
    img = Image.open(file_stream).convert('RGB')
    
    return img


def get_embeddings(image):
    with torch.no_grad():
            image_transformed = transformation_chain(image).unsqueeze(0)
            embeddings = model(image_transformed).last_hidden_state[:, 0]

    return embeddings


labels_mapping = {
    0: "Angular Leaf Spot",
    1: "Bean Rust",
    2: "Healthy",
}

# callback function to change the random number stored in state
def new_random_image():
    st.session_state["random_image_path"] = np.random.choice(test_image_paths)


#####################################
## Streamlit App
#####################################
st.markdown("""
            # 🫛 Image Similarity Search
            This App is a demonstration of the `st.connection` feature of Streamlit.
            It connects to the Qdrant vector database and allows you to find similar within the beans dataset.
            
            You can either upload your own image or draw a random image from the holdout set.
            """)

test_image_paths = load_test_image_paths()

sample_col, upload_col = st.columns(2)
with sample_col:
    st.markdown('### Draw a Sample Image from Holdout Set')
    if "random_image_path" not in st.session_state:
        st.session_state["random_image_path"] = np.random.choice(test_image_paths)
    
    st.button("New Image", on_click=new_random_image)
    sample_image = get_image_from_s3(st.session_state.random_image_path)
    Image.open("./data/healthy_train.0.jpg")
    st.image(sample_image, caption=['Random Image from Dataset'], width=300)

with upload_col:
    st.markdown('### Upload Your Own Image')
    # form to upload and image
    img_upload = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    st.markdown('If no Image is uploaded the sample image will be used.')



st.markdown("""
            ## 🔎 Find similar Images
            """)

# Button to find similar images
if st.button('Find Similar Images'):

    if img_upload is not None:
        image = Image.open(img_upload).convert('RGB')
        st.markdown('### Your Uplaodes Image')
        st.image(img_upload)
    else:
        image = sample_image.convert('RGB')
        
    img_embeddings = get_embeddings(image).tolist()[0]

    similar_images = conn.find_similars(img_embeddings, limit=3)

    for i, similar_image in enumerate(similar_images):
        st.markdown(f"#### Match {i+1}, Score: {similar_image['score']:0.2f}")
        st.write(f"Bean Class: {labels_mapping[similar_image['label']]}")
        st.image(get_image_from_s3(similar_image['path']), width=300)
        

st.markdown("## 📚 Methodology")
with st.expander("Embedding Methodology", expanded=False):
    st.markdown("""
        This app uses the pretrained Vision Transformer (ViT) model to extract embeddings from images of beans.
        Code adapted from [this tutorial](https://huggingface.co/blog/image-similarity).
        The embeddings are then stored in a Qdrant database.
        The app then allows you to upload an image and find similar images in the database.
                
        Data
        https://github.com/AI-Lab-Makerere/ibean/
        """)
