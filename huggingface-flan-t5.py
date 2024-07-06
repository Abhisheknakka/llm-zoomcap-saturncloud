#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os


# In[5]:


os.getenv('HF_TOKEN')
os.getenv('GRO_API_KEY')
os.environ['HF_HOME'] = '/run/cache/'


# In[7]:


get_ipython().system('rm -f minsearch.py')
get_ipython().system('wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py')


# In[36]:


import requests 
import minsearch

import requests 
import minsearch

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)


# In[34]:


## section 1: searching query in our database

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results


# In[39]:


## secction 2: building prompt using the query and search results from database

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[40]:


# tokenizer convert text to teken whiuch computer can understand
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")


# In[41]:


## section  3: sending to llm
def llm(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, )
    result = tokenizer.decode(outputs[0])
    return result


# In[42]:


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# In[43]:


get_ipython().system('df -h')


# In[44]:


rag("when is the course starting")


# In[28]:


#test
input_text = "translate English to French: How how you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
input_ids


# In[29]:


outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

