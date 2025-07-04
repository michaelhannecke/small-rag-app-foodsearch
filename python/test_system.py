#!/usr/bin/env python3
"""Test script for the interactive food search system"""

from shared_functions import *

def test_system():
    """Test the basic functionality of the food search system"""
    print("🧪 Testing Food Search System")
    print("=" * 40)
    
    try:
        # Test 1: Load food data
        print("1. Testing data loading...")
        food_items = load_food_data('../data/FoodDataSet.json')
        print(f"✅ Successfully loaded {len(food_items)} food items")
        
        # Test 2: Create collection
        print("\n2. Testing collection creation...")
        collection = create_similarity_search_collection(
            "test_food_search",
            {'description': 'Test collection'}
        )
        print("✅ Collection created successfully")
        
        # Test 3: Populate collection
        print("\n3. Testing data population...")
        populate_similarity_collection(collection, food_items)
        print("✅ Collection populated successfully")
        
        # Test 4: Test similarity search
        print("\n4. Testing similarity search...")
        test_queries = [
            "chocolate dessert",
            "Italian food", 
            "sweet treats"
        ]
        
        for query in test_queries:
            print(f"\n   🔍 Testing query: '{query}'")
            results = perform_similarity_search(collection, query, 3)
            if results:
                print(f"   ✅ Found {len(results)} results")
                for i, result in enumerate(results[:2], 1):
                    score = result['similarity_score'] * 100
                    print(f"      {i}. {result['food_name']} (Score: {score:.1f}%)")
            else:
                print("   ❌ No results found")
        
        # Test 5: Test filtered search
        print("\n5. Testing filtered search...")
        print("   🔍 Testing filtered query: 'dessert' with max 300 calories")
        filtered_results = perform_filtered_similarity_search(
            collection, "dessert", max_calories=300, n_results=3
        )
        if filtered_results:
            print(f"   ✅ Found {len(filtered_results)} filtered results")
            for i, result in enumerate(filtered_results[:2], 1):
                score = result['similarity_score'] * 100
                print(f"      {i}. {result['food_name']} - {result['food_calories_per_serving']} calories (Score: {score:.1f}%)")
        else:
            print("   ❌ No filtered results found")
        
        print("\n" + "=" * 40)
        print("🎉 All tests completed successfully!")
        print("The system is ready for interactive use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()