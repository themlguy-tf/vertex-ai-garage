# vertex-ai-garage
Code for Vertex AI Garage



import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
embed = hub.load(module_url)
print("module %s loaded" % module_url)


import pandas as pd

def createIndexDF(text_list, page_list, title_list, all_chunks=None, all_pages=None, all_urls=None):
    # Create a new list that concatenates text_list and all_chunks
    text_chunks = text_list.copy()
    if all_chunks is not None:
        text_chunks += all_chunks

    # Create a new list that concatenates page_list and all_pages
    page_nums = page_list.copy()
    if all_pages is not None:
        page_nums += all_pages

    # Create a new list that concatenates title_list and all_urls
    titles = title_list.copy()
    if all_urls is not None:
        titles += all_urls
        titles = [titles.lstrip("/content/corpus/").rstrip(".pdf") for titles in titles]

    # Create a new dataframe with the combined lists
    index_df = pd.DataFrame({'text_chunk': text_chunks,
                             'page_num': page_nums,
                             'title': titles})

    # Calculate embeddings using the given 'embed' function
    embeddings = []
    for row in tqdm(index_df.text_chunk):
        embedding = embed([row])
        embeddings.append(embedding)
    index_df['embedding'] = embeddings

    return index_df
