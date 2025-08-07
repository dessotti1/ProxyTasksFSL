import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DeiTForFewShot
from args import get_args 


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = DeiTForFewShot(get_args())

# Substitua pelo caminho correto do seu arquivo .pth
checkpoint_path = "proxyfsl/thamiris_FSL_places600_best.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# checkpoint é um dicionário com várias chaves, a parte do modelo fica em 'model'
state_dict = checkpoint.get('model', checkpoint)  # tenta pegar o 'model', senão usa todo

model.load_state_dict(state_dict)
model.eval()

print("Modelo carregado com sucesso!")


def generate_mean_support_embeddings(support_dir="dataset_suporte"):
    """
    Generate normalized mean embeddings for each class in the support dataset.

    Parameters
    ----------
    support_dir : str, optional
        Path to the base directory containing subdirectories for each class. Each subdirectory should contain images.
        Default is "dataset_suporte".

    Returns
    -------
    dict
        A dictionary mapping class names to their corresponding normalized mean image embedding vectors.
    """

    class_embeddings = {}

    # Get all class subdirectories
    class_names = [d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))]

    for class_name in class_names:
        class_path = os.path.join(support_dir, class_name)

        # List image files in the class directory
        image_filenames = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_filenames) == 0:
            print(f"No images found for class '{class_name}', skipping.")
            continue

        image_embeddings = []

        for filename in image_filenames:
            image_path = os.path.join(class_path, filename)

            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0) 

            with torch.no_grad():
                embedding = model.get_features(image)

            image_embeddings.append(embedding.cpu().numpy())

        # Stack and average all embeddings for the class
        stacked_embeddings = np.vstack(image_embeddings)
        mean_embedding = np.mean(stacked_embeddings, axis=0)

        # Normalize the mean embedding
        norm = np.linalg.norm(mean_embedding)
        normalized_embedding = mean_embedding / norm

        class_embeddings[class_name] = normalized_embedding.flatten()

    return class_embeddings


def generate_query_embeddings(query_dir="dataset_consulta"):
    """
    Generate normalized image embeddings for each image in the query directory.

    Parameters
    ----------
    query_dir : str, optional
        Path to the directory containing query images.
        Default is "dataset_consulta".

    Returns
    -------
    dict
        A dictionary mapping image paths to their corresponding normalized image embedding vectors.
    """

    query_embeddings = {}

    # List all image files in the query directory
    image_filenames = [f for f in os.listdir(query_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(query_dir, f) for f in image_filenames]

    for image_path in image_paths:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0) 

        with torch.no_grad():
            embedding = model.get_features(image)

        # Normalize the embedding
        embedding_np = embedding.cpu().numpy()
        norm = np.linalg.norm(embedding_np)
        normalized_embedding = embedding_np / norm

        query_embeddings[image_path] = normalized_embedding

    return query_embeddings


def softmax(x):
    """
    Compute the softmax of a vector.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    ndarray
        Softmax-transformed array.
    """
    # Numerical stability by subtracting max
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict_classes(support_embeddings, query_embeddings):
    """
    Predict the most likely class for each query image based on cosine similarity.

    Parameters
    ----------
    support_embeddings : dict
        Dictionary mapping class names to mean embeddings.
    query_embeddings : dict
        Dictionary mapping image paths to image embeddings.

    Returns
    -------
    dict
        A dictionary mapping each query image path to a prediction result containing:
            - 'predicted_class': the most probable class
            - 'top3': list of tuples (class_name, probability) for the top-3 predictions
    """
    # List of class names
    class_names = list(support_embeddings.keys())

    # Stack all class embeddings into one matrix
    support_matrix = np.vstack([support_embeddings[class_name] for class_name in class_names])

    predictions = {}

    for image_path, query_embedding in query_embeddings.items():
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = support_matrix @ query_embedding.flatten()

        # Convert similarities to probabilities
        probabilities = softmax(similarities)

        # Get top-3 class predictions
        top3_indices = probabilities.argsort()[-3:][::-1]
        top3_class_names = [class_names[i] for i in top3_indices]
        top3_probabilities = probabilities[top3_indices]

        predicted_class = top3_class_names[0]

        predictions[image_path] = {
            "predicted_class": predicted_class,
            "top3": list(zip(top3_class_names, top3_probabilities))
        }

    return predictions
