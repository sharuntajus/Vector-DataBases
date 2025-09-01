#!pip install sentence-transformers==4.1.0 | tail -n 1

import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

embeddings = model.encode(documents)
print(embeddings.shape)#4,384

def EuclideanDistance(vector1, vector2):
    squaredSum=sum((x-y) ** 2 for x,y in zip(vector1, vector2))
    return math.sqrt(squaredSum)

EuclideanDistance(embeddings[0], embeddings[1])
EuclideanDistance(embeddings[1], embeddings[0])

l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i,j] = EuclideanDistance(embeddings[i], embeddings[j])

print(l2_dist_manual)
print(l2_dist_manual[0,1])
print(l2_dist_manual[1,0])

l2_dist_manual_improved = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j > i: # Calculate the upper triangle only
            l2_dist_manual_improved[i,j] = EuclideanDistance(embeddings[i], embeddings[j])
        elif i > j: # Copy the uper triangle to the lower triangle
            l2_dist_manual_improved[i,j] = l2_dist_manual[j,i]

print("l2_dist_manual_improved",l2_dist_manual_improved)

l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print(l2_dist_scipy)

np.allclose(l2_dist_manual, l2_dist_scipy)


def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

print(dot_product_fn(embeddings[0], embeddings[1]))

dot_product_manual = np.empty([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

print(dot_product_manual)

# Matrix multiplication operator
dot_product_operator = embeddings @ embeddings.T
print(dot_product_operator)

np.allclose(dot_product_manual, dot_product_operator, atol=1e-05)
np.matmul(embeddings,embeddings.T)

# `np.dot` returns an identical result, but `np.matmul` is recommended if both arrays are 2-D:
np.dot(embeddings,embeddings.T)

dot_product_distance = -dot_product_manual
print("dot_product_distance",dot_product_distance)

# L2 norms
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
print("l2_norms",l2_norms)

# L2 norms reshaped
l2_norms_reshaped = l2_norms.reshape(-1,1)
print("l2_norms_reshaped",l2_norms_reshaped)

normalized_embeddings_manual = embeddings/l2_norms_reshaped
print("normalized_embeddings_manual",normalized_embeddings_manual)


cosine_similarity_manual = np.empty([4,4])
for i in range(normalized_embeddings_manual.shape[0]):
    for j in range(normalized_embeddings_manual.shape[0]):
        cosine_similarity_manual[i,j] = dot_product_fn(
            normalized_embeddings_manual[i],
            normalized_embeddings_manual[j]
        )

print("cosine_similarity_manual",cosine_similarity_manual)

cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T
print("cosine_similarity_operator",cosine_similarity_operator)


query_embedding = model.encode(
    ["Who is responsible for a coding project and fixing others' mistakes?"]
)

# Second, normalize the query embedding:
normalized_query_embedding = torch.nn.functional.normalize(
    torch.from_numpy(query_embedding)
).numpy()

# Third, calculate the cosine similarity between the documents and the query by using the dot product:
cosine_similarity_q3 = normalized_embeddings_manual @ normalized_query_embedding.T

# Fourth, find the position of the vector with the highest cosine similarity:
highest_cossim_position = cosine_similarity_q3.argmax()

# Fifth, find the document in that position in the `documents` array:
print(documents[highest_cossim_position])