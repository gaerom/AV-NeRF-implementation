import pickle

# Load the .pkl file
pkl_file_path = "llava_responses/llava_text_embeddings_all.pkl"

with open(pkl_file_path, "rb") as f:
    embeddings_dict = pickle.load(f)

print(embeddings_dict.values())

# # Display the keys (filenames) and a sample of their corresponding embeddings
# print("Loaded keys (filenames):", list(embeddings_dict.keys())[:5])  # Display the first 5 keys
# print("\nSample embedding for a key:")

# # Choose a specific key to inspect its embedding
# sample_key = list(embeddings_dict.keys())[0]  # Get the first key
# print(f"Embedding for {sample_key}:")
# print(embeddings_dict[sample_key])
# print(f"Embedding shape: {embeddings_dict[sample_key].shape}")


# for key, tensor in embeddings_dict.items():
#     print(f"{key}: {tensor.shape}")
