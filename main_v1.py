import os
import json
import argparse
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Energy
        energy = np.sum(y**2) / len(y)
        
        # Tempo estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        features = np.concatenate([
            mfcc_mean, 
            chroma_mean, 
            [spectral_centroid_mean], 
            [spectral_rolloff_mean],
            [zcr_mean],
            [energy],
            [tempo]
        ])
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

def cluster_songs(features, song_ids, n_clusters):
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    playlists = []
    for i in range(n_clusters):
        playlist = {
            "id": i,
            "songs": [song_ids[j] for j in range(len(song_ids)) if cluster_labels[j] == i]
        }
        playlists.append(playlist)
    
    return playlists

def main():
    parser = argparse.ArgumentParser(description='Cluster songs into playlists based on audio features')
    parser.add_argument('--path', required=True, help='Path to folder containing music files')
    parser.add_argument('--n', type=int, required=True, help='Number of clusters to generate')
    args = parser.parse_args()
    
    logger.info("Starting audio clustering process")
    
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    song_paths = []
    song_ids = []
    
    for file in os.listdir(args.path):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            file_path = os.path.join(args.path, file)
            song_id = os.path.splitext(file)[0]  #
            song_paths.append(file_path)
            song_ids.append(song_id)
    
    logger.info(f"Found {len(song_paths)} audio files")
    
    all_features = []
    valid_indices = []
    
    for i, path in enumerate(song_paths):
        logger.info(f"Processing file {i+1}/{len(song_paths)}: {os.path.basename(path)}")
        features = extract_features(path)
        
        if features is not None:
            all_features.append(features)
            valid_indices.append(i)
    
    # Filter song_ids to match the valid features
    valid_song_ids = [song_ids[i] for i in valid_indices]
    
    features_array = np.array(all_features)
    
    # Perform clustering
    logger.info(f"Clustering songs into {args.n} playlists")
    playlists = cluster_songs(features_array, valid_song_ids, args.n)
    
    # Output playlists to JSON
    output_data = {"playlists": playlists}
    output_path = "playlists_v1.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Playlists generated and saved to {output_path}")
    
    for playlist in playlists:
        logger.info(f"Playlist {playlist['id']} contains {len(playlist['songs'])} songs")

if __name__ == "__main__":
    main()