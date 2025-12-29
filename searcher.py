import torch
from qdrant_client import QdrantClient
import json
import os

class Searcher:
    def __init__(self, model, processor, collection_name, qdrant_host="localhost", qdrant_port=6333):
        # 1. Receber modelo e processador injetados
        self.model = model
        self.processor = processor
        self.collection_name = collection_name
    

        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # Verifica se collection existe para evitar erros
        if self.qdrant_client.collection_exists(self.collection_name):
            self.collection = self.qdrant_client.get_collection(self.collection_name)
        else:
            print(f"Warning: Collection {self.collection_name} does not exist yet.")

    def search(self, query_text, limit=10, score_threshold=None):
        with torch.no_grad():
            batch_query = self.processor.process_queries([query_text]).to(
                self.model.device
            )
            query_embedding = self.model(**batch_query)

        multivector_query = query_embedding[0].cpu().float().numpy().tolist()
        num_tokens = len(multivector_query)

        actual_threshold = score_threshold * num_tokens if score_threshold is not None else None

        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name, 
                query=multivector_query, 
                limit=limit, 
                timeout=60,
                score_threshold=actual_threshold
            )

            # Normalizar os scores
            for point in search_result.points:
                point.score = point.score / num_tokens

            return search_result.points
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []