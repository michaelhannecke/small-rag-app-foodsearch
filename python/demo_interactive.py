#!/usr/bin/env python3
"""Demo script to show interactive search functionality"""

from shared_functions import *

def demo_interactive_search():
    """Demonstrate the interactive search functionality"""
    print("🍽️  Interactive Food Recommendation System Demo")
    print("=" * 50)
    print("Loading food database...")
    
    # Load food data
    food_items = load_food_data('../data/FoodDataSet.json')
    print(f"✅ Loaded {len(food_items)} food items successfully")
    
    # Create and populate collection
    collection = create_similarity_search_collection(
        "demo_food_search",
        {'description': 'Demo collection for interactive search'}
    )
    populate_similarity_collection(collection, food_items)
    
    print("\n🤖 INTERACTIVE FOOD SEARCH DEMO")
    print("=" * 50)
    print("Running sample searches as mentioned in the PDF...")
    
    # Demo searches as specified in the PDF
    demo_queries = [
        "chocolate dessert",
        "Italian food", 
        "sweet treats",
        "baked goods"
    ]
    
    for query in demo_queries:
        print(f"\n🔍 Searching for '{query}'...")
        print("   Please wait...")
        
        # Perform similarity search
        results = perform_similarity_search(collection, query, 5)
        
        if not results:
            print("❌ No matching foods found.")
            continue
        
        # Display results with rich formatting
        print(f"\n✅ Found {len(results)} recommendations:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            # Calculate percentage score
            percentage_score = result['similarity_score'] * 100
            
            print(f"\n{i}. 🍽️   {result['food_name']}")
            print(f"   📊 Match Score: {percentage_score:.1f}%")
            print(f"   🏷️  Cuisine: {result['cuisine_type']}")
            print(f"   🔥 Calories: {result['food_calories_per_serving']} per serving")
            print(f"   📝 Description: {result['food_description']}")
            
            # Add visual separator
            if i < len(results):
                print("   " + "-" * 50)
        
        print("=" * 60)
        
        # Extract cuisine types from results for suggestions
        cuisines = list(set([r['cuisine_type'] for r in results]))
        print("\n💡 Related searches you might like:")
        for cuisine in cuisines[:3]:
            print(f"   • Try '{cuisine} dishes' for more {cuisine} options")
        
        print("\n" + "🔹" * 50)

if __name__ == "__main__":
    demo_interactive_search()