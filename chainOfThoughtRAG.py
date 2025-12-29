from groq import Groq
import os
from searcher import Searcher
import sqlite3
from pdf2image import convert_from_path
from io import BytesIO
import base64
from PIL import Image

class ChainOfThoughtRAG:
    def __init__(self, model, processor, collection_name):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.processor = processor
        self.reasoning_steps = []
        self.searcher = Searcher(model, processor, collection_name)

        self.num_retrieved_docs = 3
        self.score_threshold = 0.6

        self.without_RAG_history = []
        self.history_length = 3

    def extract_page_as_base64(self, pdf_path, page_number):
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            pages = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=page_number,
                last_page=page_number
            )
        except Exception as e:
            raise ValueError(f"Unable to read PDF {pdf_path}: {e}") from e

        if not pages:
            raise ValueError("The PDF does not have the requested page.")
        image: Image.Image = pages[0]
        # redimensionar se necessário (poupar tokens de input)
        #image.thumbnail((1024, 1024))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_b64}"
        return image_url
    
    def _save_or_get_description(self, id, doc_path=None, page=None):

        db_path = "descriptions.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Criar tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_descriptions (
                id TEXT PRIMARY KEY,
                doc_path TEXT,
                page INTEGER,
                description TEXT
            )
        """)

        # Verificar se já existe descrição
        cursor.execute(
            "SELECT description FROM image_descriptions WHERE id = ?",
            (id,)
        )
        row = cursor.fetchone()

        if row:
            conn.close()
            print("Using existing description from the database.")
            return row[0]  # existing description
        # If not existing, generate new description
        if not doc_path or not os.path.exists(doc_path):
            print(f"Warning: document path missing or not found for id={id}: {doc_path}")
            return f"(missing file: {doc_path})"

        if page is None:
            print(f"Warning: page number missing for id={id}")
            return "(missing page)"

        try:
            image_URL = self.extract_page_as_base64(doc_path, page)
        except Exception as e:
            print(f"Error extracting page for id={id}: {e}")
            return f"(error reading file: {e})"

        description = self.describe_image(image_URL)
        # Guardar nova descrição
        cursor.execute(
            """
            INSERT INTO image_descriptions (id, doc_path, page, description)
            VALUES (?, ?, ?, ?)
            """,
            (id, doc_path, page, description)
        )

        conn.commit()
        conn.close()
        return description

    def describe_image(self, image_URL):
        print("Describing image...")
        user_content = [
            {
            "type": "text",
            "text": (
                "Extract all the text from the following image and, in place of images or figures, provide a detailed description of their content."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_URL
            }
        }]   
        completions = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        full_response = ""
        for chunk in completions:
            full_response += chunk.choices[0].delta.content or ""
            
        return full_response.strip()
    

    def decompose_question(self, complex_question):
        # Implement decomposition logic here
        
        decomposed_prompt = f"""
        Decompose the following complex user question into a concise list of atomic English sub-questions that can be answered independently. 
        The goal is to use these sub-questions to retrieve relevant information from a document database.

        Guidelines:
        - Each sub-question must focus on a single aspect of the original question.
        - Avoid overlapping information between sub-questions.
        - If the original question is simple, return it as the only sub-question.
        - ALWAYS if you have different languages in the question, decompose it to English.
        - Format: Return ONLY a numbered list.

        Question: {complex_question}

        Sub-questions:
        """

        return self._llm_decompose(decomposed_prompt)

    
    def _llm_decompose(self, prompt):

        completions = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        full_response = ""
        for chunk in completions:
            full_response += chunk.choices[0].delta.content or ""
            
        steps = []
        for line in full_response.split('\n'):
            if line.strip():
                steps.append(line.strip())
                
        return steps


    def reason_step(self, step, retrieved_docs):
        step_reasoning = f"""
        Based on the following retrieved documents, provide a detailed reasoning to answer the question.
        - DO NOT use information NOT present in the retrieved documents.
        - Be precise and concise.
        {step}
        Retrieved Documents:
        {retrieved_docs}
        """
        completions = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": step_reasoning,
                },
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        full_response = ""
        for chunk in completions:
            full_response += chunk.choices[0].delta.content or ""
            
        return full_response.strip()

    def build_final_answer(self, reasoning_chain, original_query):
        
        chain_summary = ""
        # print("REASONING_CHAIN:", reasoning_chain)
        for r in reasoning_chain:
            references = ""
            for p in r['retrieved_docs']:
                references += f"- [{p['payload'].get('document')}]({p['payload'].get('document_path')}/{p['payload'].get('page')})\n"

            chain_summary += "\n".join([f"(Step {r['step']}: {r['reasoning']}) -> REFERENCES ({references})"]) + "\n"
        
        final_prompt = f"""
        Original Question: {original_query}
        Reasoning Chain:
        {chain_summary}
        Based on the above reasoning chain, provide a comprehensive and complete final answer to the original question.
        RULES (MANDATORY):
        - USE bullet points, tables, diagrams, or any other format that improves clarity.
        - USE ONLY information explicitly present in the reasoning chain.
        - Citations MUST use EXACTLY the following MARKDOWN FORMAT: [document_name](doc_path/page_number).
        - Do NOT create a "References", "Sources", or similar section.
        """
        # print("Final answer prompt:", final_prompt)

        completions = self.client.chat.completions.create(
            
            model= "openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )
        
        full_response = completions.choices[0].message.content or ""
        # usage = completions.usage
        # print("\n" + "=" * 50)
        # print("TOKENS UTILIZADOS (TESTE VISION)")
        # print("=" * 50)
        # print(f"Tokens de Input:  {usage.prompt_tokens}")
        # print(f"Tokens de Output: {usage.completion_tokens}")
        # print(f"Total de Tokens:  {usage.total_tokens}")
        # print("=" * 50 + "\n")
        return full_response.strip()
    

    def chain_retrieve_and_reason(self, query, progress_callback=None):

        # 1. Decompose
        if progress_callback:
            progress_callback({"type": "status", "subtype": "decomposing", "text": "Decomposing question..."})
        steps = self.decompose_question(query)
        # print("\n\nDecomposed steps:", steps)
        if progress_callback:
            progress_callback({"type": "status", "subtype": "decomposed", "steps": steps})

        reasoning_chain = []

        for i, step in enumerate(steps):
            # notify step start
            if progress_callback:
                progress_callback({"type": "status", "subtype": "step_start", "step": i+1, "question": step})

            relevant_docs = self.searcher.search(step, limit = self.num_retrieved_docs, score_threshold=self.score_threshold)
            # print("\n\nrelevant_docs:", relevant_docs)

            if not relevant_docs: # []
                if progress_callback:
                    progress_callback({"type": "status", "subtype": "no_docs", "step": i+1})
                continue

            # obtain descriptions for retrieved docs
            description = ""
            for r in relevant_docs:
                doc_id = r.id
                doc_path = r.payload.get('document_path')
                page = r.payload.get('page')
                if progress_callback:
                    progress_callback({"type": "status", "subtype": "fetching_description", "step": i+1, "id": doc_id, "doc_path": doc_path, "page": page})
                desc = self._save_or_get_description(doc_id, doc_path, page)
                # print("\n\nDocs description:", desc)
                if progress_callback:
                    progress_callback({"type": "status", "subtype": "description", "step": i+1, "id": doc_id, "description": desc})
                description += desc + "\n"

            # reason for this step
            if progress_callback:
                progress_callback({"type": "status", "subtype": "reasoning_start", "step": i+1, "question": step})
            step_reasoning = self.reason_step(step, description)
            # print("\n\nStep reasoning:", step_reasoning)
            if progress_callback:
                progress_callback({"type": "status", "subtype": "reasoning", "step": i+1, "reasoning": step_reasoning})

            reasoning_chain.append({
                "step": i + 1,
                "question": step,
                "retrieved_docs": [ {"id": r.id, "score": r.score, "payload": r.payload} for r in relevant_docs ],
                "reasoning": step_reasoning
            })
        
        # final answer
        if progress_callback:
            progress_callback({"type": "status", "subtype": "finalizing", "text": "Building final answer..."})
            
        if not reasoning_chain:
            if progress_callback:
                final_answer = "I'm sorry, but I cant answer your question!"
                progress_callback({"type": "status", "subtype": "final", "answer": final_answer})
                return final_answer
            
        final_answer = self.build_final_answer(reasoning_chain, query)
        # print("\n\nFinal answer:", final_answer)
        if progress_callback:
            progress_callback({"type": "status", "subtype": "final", "answer": final_answer})
        return final_answer, reasoning_chain


    def answer_without_rag(self, query):

        self.without_RAG_history.append({"role": "user", "content": query})
        completions = self.client.chat.completions.create(
            
            model= "openai/gpt-oss-120b",
            messages=self.without_RAG_history[-self.history_length:],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )
        
        full_response = completions.choices[0].message.content or ""
        self.without_RAG_history.append({"role": "assistant", "content": full_response})

        # usage = completions.usage
        # print("\n" + "=" * 50)
        # print("TOKENS UTILIZADOS (TESTE VISION)")
        # print("=" * 50)
        # print(f"Tokens de Input:  {usage.prompt_tokens}")
        # print(f"Tokens de Output: {usage.completion_tokens}")
        # print(f"Total de Tokens:  {usage.total_tokens}")
        # print("=" * 50 + "\n")
        return full_response.strip()
    

