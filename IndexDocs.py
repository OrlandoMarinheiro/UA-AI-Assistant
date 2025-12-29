from typing import Self
import torch
from qdrant_client import QdrantClient, models
from pdf2image import convert_from_path, pdfinfo_from_path
import os
import stamina
import hashlib
import time

class PDFIndexer:
    def __init__(self, 
                 model,
                 processor,
                 collection_name,
                 qdrant_host="localhost", 
                 qdrant_port=6333,
                 ):
        
        print("Initializing PDFIndexer with shared model...")
        
        self.model = model
        self.processor = processor
        self.device = self.model.device

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self._initialize_collection()
        
        print("PDFIndexer ready.")

    def _initialize_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=128, 
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        ),
                    ),
                )
            )
        except Exception:
            pass

    @stamina.retry(on=Exception, attempts=3)
    def _upsert_to_qdrant(self, points):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False,
        )

    # --- ALTERAÇÃO AQUI: Adicionado progress_callback ---
    def index_pdf(self, pdf_path, session_id="default_session", batch_size = 4, progress_callback=None):
        if not os.path.exists(pdf_path):
            return False

        filename = os.path.basename(pdf_path)
        
        try:
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
        except Exception as e:
            print(f"Error reading info: {e}")
            return False

        print(f"Indexing file: {filename} ({total_pages} pages)")
        

        start = time.time()
        # Reportar 0% inicial
        if progress_callback: progress_callback(0)

        for i in range(0, total_pages, batch_size):
            end_page = min(i + batch_size, total_pages)
            batch_indices = range(i, end_page)
            
            # --- Lógica de IDs ---
            batch_ids = []
            ids_map = {}
            for global_idx in batch_indices:
                doc_id_str = f"{filename}_{global_idx}"
                point_id = hashlib.md5(doc_id_str.encode()).hexdigest()
                batch_ids.append(point_id)
                ids_map[point_id] = global_idx

            # --- Verificar existentes ---
            try:
                existing_records = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=batch_ids,
                    with_payload=False, with_vectors=False
                )
                existing_ids = {record.id for record in existing_records}
            except Exception:
                existing_ids = set()

            if len(existing_ids) == len(batch_ids):
                # Mesmo saltando, atualizamos o progresso
                if progress_callback:
                    progress = int((end_page / total_pages) * 100)
                    progress_callback(progress)
                continue

            # --- Conversão e Inferência ---
            try:
                batch_pages = convert_from_path(pdf_path, first_page=i+1, last_page=end_page, dpi=300, fmt="jpeg")
                
                images_to_process = []
                metadata_to_process = []
                
                for local_idx, pid in enumerate(batch_ids):
                    if pid not in existing_ids and local_idx < len(batch_pages):
                        images_to_process.append(batch_pages[local_idx])
                        metadata_to_process.append({
                            "point_id": pid,
                            "global_idx": ids_map[pid]
                        })
                
                if images_to_process:
                    with torch.no_grad():
                        batch_images_tensor = self.processor.process_images(images_to_process).to(self.model.device)
                        image_embeddings = self.model(**batch_images_tensor)

                    points = []
                    for j, embedding in enumerate(image_embeddings):
                        meta = metadata_to_process[j]
                        multivector = embedding.cpu().float().numpy().tolist()
                        points.append(models.PointStruct(
                            id=meta["point_id"],
                            vector=multivector,
                            payload={
                                "session_id": session_id,
                                "document": filename,
                                "document_path": pdf_path,
                                "page": meta["global_idx"] + 1,
                            }
                        ))
                    self._upsert_to_qdrant(points)

            except Exception as e:
                print(f"Error processing batch: {e}")
                return False
            
            # --- ATUALIZAR PROGRESSO REAL ---
            if progress_callback:
                progress = int((end_page / total_pages) * 100)
                progress_callback(progress)
        elapsed = time.time() - start
        print(f"Indexing completed in {elapsed:.2f} seconds.")
        return True
    
    def delete_file(self, filename):
        """
        Remove todos os vetores associados a um nome de ficheiro específico.
        """
        print(f"Deleting vectors for file: {filename}")
        try:
            # 1. Definir o filtro: apagar onde o campo "document" é igual ao filename
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
            
            # 2. Executar o delete com FilterSelector
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=filter_condition
                ),
                wait=False
            )
            print(f"Successfully deleted vectors for {filename}")
            return True
            
        except Exception as e:
            print(f"Error deleting file {filename} from Qdrant: {e}")
            return False