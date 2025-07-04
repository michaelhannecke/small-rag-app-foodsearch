# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a food search and recommendation system built with Python that demonstrates three approaches to similarity search and conversational AI:

1. **Interactive CLI Search** (`interactive_search.py`) - Real-time food search with user interaction
2. **Shared Functions Module** (`shared_functions.py`) - Core functionality for data loading, ChromaDB operations, and similarity search
3. **Main Entry Point** (`main.py`) - Simple entry point (currently minimal)

## Architecture

### Core Components

- **ChromaDB Vector Database**: Uses `all-MiniLM-L6-v2` sentence transformer for embeddings with cosine similarity
- **Food Data**: Rich JSON dataset with nutritional info, ingredients, cooking methods, and taste profiles
- **Similarity Search**: Supports both basic and filtered search with metadata constraints (cuisine type, calories)

### Key Files

- `shared_functions.py` - Core library with data loading, ChromaDB setup, and search functions
- `interactive_search.py` - CLI chatbot interface (incomplete - missing `handle_food_search` function)
- `data/FoodDataSet.json` - Food dataset with comprehensive metadata
- `pyproject.toml` - Project configuration (minimal dependencies specified)

## Development Commands

### Installation
```bash
uv pip install numpy==1.26.4 scipy==1.16.0 faiss-cpu==1.7.4 sentence-transformers==4.1.0 ollama
```

### Running the Application
```bash
cd python
python3 interactive_search.py      # Interactive CLI search
python3 advanced_search.py         # Advanced search with filtering
python3 enhanced_rag_chatbot.py    # RAG chatbot with Llama 3.2
python3 system_comparison.py       # System comparison tool
python3 test_system.py            # Comprehensive testing
```

### Data Setup
The system expects `../data/FoodDataSet.json` to exist relative to the python folder. The dataset contains food items with:
- `food_id`, `food_name`, `food_description`
- `food_calories_per_serving`, `food_nutritional_factors`
- `food_ingredients`, `cuisine_type`, `cooking_method`
- `food_health_benefits`, `food_features` (taste, texture, appearance)

## Key Functions

### Data Management
- `load_food_data()` - Loads and normalizes JSON food data
- `create_similarity_search_collection()` - Sets up FAISS collection with cosine similarity
- `populate_similarity_collection()` - Adds food items with embeddings using SentenceTransformer

### Search Functions
- `perform_similarity_search()` - Basic similarity search using FAISS IndexFlatIP
- `perform_filtered_similarity_search()` - Search with cuisine/calorie filters and metadata matching

## System Status

✅ **Complete Implementation**
- All four systems fully functional and tested
- Interactive CLI search with rich interface
- Advanced search with filtering capabilities  
- RAG chatbot powered by Llama 3.2 via Ollama
- Comprehensive system comparison tool
- All Python files organized in `python/` folder
- **Vector database migrated from ChromaDB to FAISS for improved performance**
- Updated paths and documentation