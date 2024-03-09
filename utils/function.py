def split_context(context):
    docs = re.split(r'(?<!\d)\.\s*(?![.])', context)
    docs = [doc for doc in docs if doc]
    docs = '\n\n'.join(docs)
    return docs.split('\n\n')