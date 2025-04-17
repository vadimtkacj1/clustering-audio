import os
import json
import argparse
import numpy as np
import torch
import torchaudio
from sklearn.cluster import KMeans
import logging
from transformers import AutoFeatureExtractor, AutoModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioEmbeddingExtractor:
    def __init__(self):
        logger.info("Loading pre-trained audio model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the model and feature extractor
        model_name = "microsoft/wavlm-base-plus" 
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def extract_embedding(self, file_path):
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != self.feature_extractor.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.feature_extractor.sampling_rate
                )
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Prepare input for the model
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(), 
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt"
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Extract embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting embedding from {file_path}: {e}")
            return None

def cluster_songs(features, song_ids, n_clusters):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Group songs by cluster
    playlists = []
    for i in range(n_clusters):
        playlist = {
            "id": i,
            "songs": [song_ids[j] for j in range(len(song_ids)) if cluster_labels[j] == i]
        }
        playlists.append(playlist)
    
    return playlists

def main():
    parser = argparse.ArgumentParser(description='Cluster songs into playlists based on deep learning features')
    parser.add_argument('--path', required=True, help='Path to folder containing music files')
    parser.add_argument('--n', type=int, required=True, help='Number of clusters to generate')
    args = parser.parse_args()
    
    logger.info("Starting audio clustering process")
    torch.set_default_dtype(torch.float32)
    
    # Get all audio files in the directory
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    song_paths = []
    song_ids = []
    
    for file in os.listdir(args.path):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            file_path = os.path.join(args.path, file)
            song_id = os.path.splitext(file)[0]  # Use filename without extension as ID
            song_paths.append(file_path)
            song_ids.append(song_id)
    
    logger.info(f"Found {len(song_paths)} audio files")
    
    # Initialize the embedding extractor
    extractor = AudioEmbeddingExtractor()
    
    # Extract embeddings from each song
    all_embeddings = []
    valid_indices = []
    
    for i, path in enumerate(song_paths):
        logger.info(f"Processing file {i+1}/{len(song_paths)}: {os.path.basename(path)}")
        embedding = extractor.extract_embedding(path)
        
        if embedding is not None:
            all_embeddings.append(embedding)
            valid_indices.append(i)
    

    valid_song_ids = [song_ids[i] for i in valid_indices]
    
    embeddings_array = np.array(all_embeddings)
    
    # Perform clustering
    logger.info(f"Clustering songs into {args.n} playlists")
    playlists = cluster_songs(embeddings_array, valid_song_ids, args.n)
    
    # Output playlists to JSON
    output_data = {"playlists": playlists}
    output_path = "playlists_v2.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Playlists generated and saved to {output_path}")
    
    # Print stats about the playlists
    for playlist in playlists:
        logger.info(f"Playlist {playlist['id']} contains {len(playlist['songs'])} songs")

if __name__ == "__main__":
    main()