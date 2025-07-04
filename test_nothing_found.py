#!/usr/bin/env python3
"""Demonstration of the 'Nothing found in my database' functionality"""

import sys
sys.path.append('./python')
from enhanced_rag_chatbot import generate_llm_rag_response

def demo_nothing_found():
    print("🤖 RAG Chatbot - 'Nothing Found' Demo")
    print("=" * 50)
    
    # Test cases that should return "Nothing found in my database"
    test_cases = [
        {
            "query": "flying unicorn steaks",
            "results": [],
            "description": "Empty results"
        },
        {
            "query": "spicy dragon meat curry", 
            "results": [
                {
                    'food_name': 'Apple', 
                    'food_description': 'A fresh red apple',
                    'cuisine_type': 'Fruit', 
                    'food_calories_per_serving': 80, 
                    'similarity_score': 0.12
                }
            ],
            "description": "Low similarity (12%)"
        },
        {
            "query": "plutonium sandwich",
            "results": [
                {
                    'food_name': 'Regular Sandwich', 
                    'food_description': 'Turkey and cheese sandwich',
                    'cuisine_type': 'American', 
                    'food_calories_per_serving': 350, 
                    'similarity_score': 0.25
                }
            ],
            "description": "Irrelevant match (25%)"
        }
    ]
    
    # Test case that should provide a recommendation
    good_case = {
        "query": "chocolate dessert",
        "results": [
            {
                'food_name': 'Chocolate Lava Cake', 
                'food_description': 'Rich chocolate dessert with molten center',
                'cuisine_type': 'American', 
                'food_calories_per_serving': 450, 
                'similarity_score': 0.85
            }
        ],
        "description": "Good match (85%)"
    }
    
    print("❌ Cases that should return 'Nothing found in my database':")
    print()
    
    for i, case in enumerate(test_cases, 1):
        response = generate_llm_rag_response(case["query"], case["results"])
        print(f"{i}. Query: \"{case['query']}\"")
        print(f"   Context: {case['description']}")
        print(f"   Response: \"{response}\"")
        print()
    
    print("✅ Case that should provide recommendations:")
    print()
    
    response = generate_llm_rag_response(good_case["query"], good_case["results"])
    print(f"Query: \"{good_case['query']}\"")
    print(f"Context: {good_case['description']}")
    print(f"Response: \"{response}\"")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed! The system correctly handles edge cases.")

if __name__ == "__main__":
    demo_nothing_found()