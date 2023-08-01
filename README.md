# Streamlit Connections Hackathon Submission

Demo Streamlit App which uses `st.connection` with the Qdrant vector database. 

Check out the deployed version: https://qdrant-connection.streamlit.app/

In order to run the app you need to create a Qdrant database and populate it with the embedded beans dataset. Additionally you need to store the beans images for display. Here I used an S3 bucket, but you can use any storage solution you like.
I aimed to use `st.connection` for the S3 connection, but unfortunately currently the provided implementation does not support picture files.
Store the credentials for the Qdrant database and the S3 bucket in `.streamlit/secrets.toml` with the following format:

```
[connections.qdrant]
url = ""
api_key = ""

[connections.s3]
key = ""
secret = ""

``````

For embedding the images I adapted the approach from this blogpost: https://huggingface.co/blog/image-similarity, which featured a trained ViT model (https://huggingface.co/nateraw/vit-base-beans). 

Would be fun to finetune a model on a different dataset, but for me that was unfortunately out of scope for this hackathon.

