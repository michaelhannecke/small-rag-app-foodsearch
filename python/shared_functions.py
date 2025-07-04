import faiss
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Global variables for FAISS setup
model = None
food_data_store = {}
metadata_store = {}

def initialize_sentence_transformer():
    """Initialize the sentence transformer model"""
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def load_food_data(file_path: str) -> List[Dict]: 
    """Load food data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)
        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])
            
            # Ensure required fields exist
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0

            # Extract taste features from nested food_features if available
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                item['taste_profile'] = ', '.join(taste_features)
            else:
                item['taste_profile'] = ''
        
        print(f"Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data
    
    except Exception as e:
        print(f"Error loading food data: {e}")
        return []


class FAISSFoodCollection:
    """A FAISS-based collection for food similarity search"""
    
    def __init__(self, collection_name: str, collection_metadata: dict = None):
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata or {}
        self.index = None
        self.embeddings = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.model = initialize_sentence_transformer()
        
    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the FAISS collection"""
        # Generate embeddings
        embeddings = self.model.encode(documents)
        
        # Store embeddings and metadata
        self.embeddings = np.array(embeddings).astype('float32')
        self.documents = documents
        self.metadatas = metadatas
        self.ids = ids
        
        # Create FAISS index (using cosine similarity via normalization + inner product)
        dimension = self.embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create index (Inner Product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
    def query(self, query_texts: List[str], n_results: int = 5, where: Dict = None):
        """Query the FAISS collection"""
        if self.index is None or len(self.documents) == 0:
            return {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}
        
        # Generate query embedding
        query_embeddings = self.model.encode(query_texts)
        query_embeddings = np.array(query_embeddings).astype('float32')
        
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, n_results)
        
        # Filter results based on metadata constraints if provided
        filtered_results = self._apply_metadata_filters(indices[0], scores[0], where)
        
        # Convert scores back to distances (1 - cosine_similarity)
        distances = []
        result_indices = []
        result_metadatas = []
        result_ids = []
        
        for idx, score in filtered_results[:n_results]:
            if idx < len(self.metadatas):
                # Convert similarity score to distance
                distance = 1.0 - score
                distances.append(distance)
                result_indices.append(idx)
                result_metadatas.append(self.metadatas[idx])
                result_ids.append(self.ids[idx])
        
        return {
            'ids': [result_ids],
            'distances': [distances], 
            'metadatas': [result_metadatas]
        }
    
    def _apply_metadata_filters(self, indices: np.ndarray, scores: np.ndarray, where: Dict = None):
        """Apply metadata filters to search results"""
        if where is None:
            return list(zip(indices, scores))
        
        filtered_results = []
        
        for idx, score in zip(indices, scores):
            if idx >= len(self.metadatas):
                continue
                
            metadata = self.metadatas[idx]
            
            # Check if metadata matches filter criteria
            if self._matches_filter(metadata, where):
                filtered_results.append((idx, score))
        
        return filtered_results
    
    def _matches_filter(self, metadata: Dict, where: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        if "$and" in where:
            # All conditions must be true
            return all(self._matches_filter(metadata, condition) for condition in where["$and"])
        
        for key, value in where.items():
            if key == "$and":
                continue
                
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle operators like $lte
                for operator, operand in value.items():
                    if operator == "$lte":
                        if metadata[key] > operand:
                            return False
                    elif operator == "$gte":
                        if metadata[key] < operand:
                            return False
                    elif operator == "$eq":
                        if metadata[key] != operand:
                            return False
            else:
                # Direct equality check
                if metadata[key] != value:
                    return False
        
        return True


def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    """Create FAISS collection for similarity search"""
    return FAISSFoodCollection(collection_name, collection_metadata)


def populate_similarity_collection(collection: FAISSFoodCollection, food_items: List[Dict]):
    """Populate collection with food data and generate embeddings"""
    documents = []
    metadatas = []
    ids = []
    
    # Create unique IDs to avoid duplicates
    used_ids = set()
    
    for i, food in enumerate(food_items):
        # Create comprehensive text for embedding using rich JSON structure
        text = f"Name: {food['food_name']}. "
        text += f"Description: {food.get('food_description', '')}. "
        text += f"Ingredients: {', '.join(food.get('food_ingredients', []))}. "
        text += f"Cuisine: {food.get('cuisine_type', 'Unknown')}. "
        text += f"Cooking method: {food.get('cooking_method', '')}. "
        
        # Add taste profile from food_features 
        taste_profile = food.get('taste_profile', '')
        if taste_profile:
            text += f"Taste and features: {taste_profile}. "
            
        # Add health benefits if available
        health_benefits = food.get('food_health_benefits', '') 
        if health_benefits:
            text += f"Health benefits: {health_benefits}. "
            
        # Add nutritional information
        if 'food_nutritional_factors' in food:
            nutrition = food['food_nutritional_factors']
            if isinstance(nutrition, dict):
                nutrition_text = ', '.join([f"{k}: {v}" for k, v in nutrition.items()]) 
                text += f"Nutrition: {nutrition_text}."

        # Generate unique ID to avoid duplicates 
        base_id = str(food.get('food_id', i)) 
        unique_id = base_id
        counter = 1
        while unique_id in used_ids: 
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)
        
        documents.append(text) 
        ids.append(unique_id) 
        metadatas.append({
            "name": food["food_name"],
            "cuisine_type": food.get("cuisine_type", "Unknown"),
            "ingredients": ", ".join(food.get("food_ingredients", [])),
            "calories": food.get("food_calories_per_serving", 0),
            "description": food.get("food_description", ""),
            "cooking_method": food.get("cooking_method", ""),
            "health_benefits": food.get("food_health_benefits", ""),
            "taste_profile": food.get("taste_profile", "")
        })
        
    # Add all data to collection 
    collection.add(
        documents=documents, 
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(food_items)} food items to FAISS collection")


def perform_similarity_search(collection: FAISSFoodCollection, query: str, n_results: int = 5) -> List[Dict]:
    """Perform similarity search and return formatted results"""
    try:
        results = collection.query( 
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance)
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'food_id': results['ids'][0][i],
                'food_name': results['metadatas'][0][i]['name'],
                'food_description': results['metadatas'][0][i]['description'],
                'cuisine_type': results['metadatas'][0][i]['cuisine_type'],
                'food_calories_per_serving': results['metadatas'][0][i]['calories'],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            
            formatted_results.append(result)
        return formatted_results
    
    except Exception as e:
        print(f"Error in similarity search: {e}") 
        return []
    

def perform_filtered_similarity_search(collection: FAISSFoodCollection, query: str, cuisine_filter: str = None,
                                       max_calories: int = None, n_results: int = 5) -> List[Dict]:
    """Perform filtered similarity search with metadata constraints""" 
    where_clause = None

    # Build filters list 
    filters = []
    if cuisine_filter:
        filters.append({"cuisine_type": cuisine_filter})
        
    if max_calories:
        filters.append({"calories": {"$lte": max_calories}})
        
    # Construct where clause based on number of filters
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}
        
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results * 2,  # Get more results to account for filtering
            where=where_clause
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(min(len(results['ids'][0]), n_results)):
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'food_id': results['ids'][0][i],
                'food_name': results['metadatas'][0][i]['name'],
                'food_description': results['metadatas'][0][i]['description'],
                'cuisine_type': results['metadatas'][0][i]['cuisine_type'],
                'food_calories_per_serving': results['metadatas'][0][i]['calories'],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return []