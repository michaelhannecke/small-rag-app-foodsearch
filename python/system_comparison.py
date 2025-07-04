#!/usr/bin/env python3
"""
Part 5: Testing and Comparing All Three Systems
Comprehensive comparison tool for Interactive Search, Advanced Search, and RAG Chatbot
"""

from shared_functions import *
import ollama
import time
from typing import List, Dict, Any

def main():
    """Main function for system comparison and testing"""
    print("🔬 FOOD SEARCH SYSTEMS COMPARISON AND TESTING")
    print("=" * 60)
    print("This tool compares all three food search systems:")
    print("1. Interactive CLI Search System")
    print("2. Advanced Search with Filters")
    print("3. Enhanced RAG Chatbot (Llama 3.2)")
    print("=" * 60)
    
    # Initialize all systems
    print("\n🔧 Initializing all systems...")
    
    # Load food data
    food_items = load_food_data('../data/FoodDataSet.json')
    print(f"✅ Loaded {len(food_items)} food items")
    
    # Create collections for each system
    collections = {}
    
    # Interactive search collection
    collections['interactive'] = create_similarity_search_collection(
        "comparison_interactive",
        {'description': 'Collection for interactive search comparison'}
    )
    populate_similarity_collection(collections['interactive'], food_items)
    
    # Advanced search collection
    collections['advanced'] = create_similarity_search_collection(
        "comparison_advanced", 
        {'description': 'Collection for advanced search comparison'}
    )
    populate_similarity_collection(collections['advanced'], food_items)
    
    # RAG chatbot collection
    collections['rag'] = create_similarity_search_collection(
        "comparison_rag",
        {'description': 'Collection for RAG chatbot comparison'}
    )
    populate_similarity_collection(collections['rag'], food_items)
    
    print("✅ All collections initialized")
    
    # Test Ollama connection
    print("🔗 Testing Ollama connection...")
    try:
        test_response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': 'Test'}]
        )
        print("✅ Ollama connection established")
        ollama_available = True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        ollama_available = False
    
    # Start comparison interface
    comparison_interface(collections, ollama_available)


def comparison_interface(collections: Dict, ollama_available: bool):
    """Interactive interface for comparing systems"""
    print("\n" + "="*60)
    print("🔍 SYSTEM COMPARISON INTERFACE")
    print("="*60)
    print("Comparison Options:")
    print("  1. Run predefined test queries on all systems")
    print("  2. Custom query comparison")
    print("  3. Performance benchmarking")
    print("  4. Feature matrix comparison")
    print("  5. Individual system testing")
    print("  6. Export comparison report")
    print("  7. Exit")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\n📋 Select option (1-7): ").strip()
            
            if choice == '1':
                run_predefined_tests(collections, ollama_available)
            elif choice == '2':
                custom_query_comparison(collections, ollama_available)
            elif choice == '3':
                performance_benchmarking(collections, ollama_available)
            elif choice == '4':
                feature_matrix_comparison()
            elif choice == '5':
                individual_system_testing(collections, ollama_available)
            elif choice == '6':
                export_comparison_report(collections, ollama_available)
            elif choice == '7':
                print("👋 Exiting System Comparison. Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_predefined_tests(collections: Dict, ollama_available: bool):
    """Run predefined test queries on all three systems"""
    print("\n🧪 RUNNING PREDEFINED TEST QUERIES")
    print("=" * 50)
    
    test_queries = [
        {
            "query": "chocolate dessert",
            "description": "Simple dessert search"
        },
        {
            "query": "healthy Italian food under 400 calories",
            "description": "Complex query with cuisine and calorie filters"
        },
        {
            "query": "spicy comfort food for dinner",
            "description": "Natural language query with mood context"
        },
        {
            "query": "light breakfast options",
            "description": "Meal-specific search"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print("-" * 40)
        
        # System 1: Interactive Search
        print("\n📱 Interactive Search Results:")
        results_interactive = test_interactive_search(collections['interactive'], test['query'])
        display_comparison_results(results_interactive, "Interactive")
        
        # System 2: Advanced Search (Basic mode for comparison)
        print("\n🔧 Advanced Search Results:")
        results_advanced = test_advanced_search(collections['advanced'], test['query'])
        display_comparison_results(results_advanced, "Advanced")
        
        # System 3: RAG Chatbot
        if ollama_available:
            print("\n🤖 RAG Chatbot Results:")
            results_rag = test_rag_search(collections['rag'], test['query'])
            display_rag_results(results_rag)
        else:
            print("\n🤖 RAG Chatbot: ❌ Ollama not available")
        
        input("\n⏸️  Press Enter to continue to next test...")


def test_interactive_search(collection, query: str) -> List[Dict]:
    """Test the interactive search system"""
    return perform_similarity_search(collection, query, 3)


def test_advanced_search(collection, query: str) -> List[Dict]:
    """Test the advanced search system"""
    return perform_similarity_search(collection, query, 3)


def test_rag_search(collection, query: str) -> Dict:
    """Test the RAG chatbot system"""
    search_results = perform_similarity_search(collection, query, 3)
    
    if not search_results:
        return {"response": "No results found", "search_results": []}
    
    try:
        # Prepare context
        context = prepare_rag_context(query, search_results)
        
        # Build prompt
        prompt = f'''You are a helpful food recommendation assistant. A user asked: "{query}"

Retrieved food options:
{context}

IMPORTANT INSTRUCTIONS:
- If no relevant food options were found OR if the food options don't match the user's query, respond EXACTLY with: "Nothing found in my database"
- Only recommend food items that are actually listed in the retrieved options above
- Do NOT make up or suggest foods that are not in the database results

If you have relevant food options, provide a brief, helpful recommendation (under 100 words) mentioning 1-2 specific food items from the options above.

Response:'''
        
        # Generate response
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        ai_response = response['message']['content'].strip() if response and 'message' in response else "Response generation failed"
        
        return {
            "response": ai_response,
            "search_results": search_results
        }
        
    except Exception as e:
        return {
            "response": f"Error generating response: {e}",
            "search_results": search_results
        }


def prepare_rag_context(query: str, search_results: List[Dict]) -> str:
    """Prepare context for RAG system"""
    context_parts = []
    for i, result in enumerate(search_results[:3], 1):
        context_parts.append(f"{i}. {result['food_name']} - {result['food_description']} ({result['cuisine_type']}, {result['food_calories_per_serving']} calories)")
    return "\n".join(context_parts)


def display_comparison_results(results: List[Dict], system_name: str):
    """Display results for comparison"""
    if not results:
        print(f"   ❌ No results found")
        return
    
    for i, result in enumerate(results, 1):
        score = result['similarity_score'] * 100
        print(f"   {i}. {result['food_name']} ({score:.1f}% match)")
        print(f"      Cuisine: {result['cuisine_type']}, Calories: {result['food_calories_per_serving']}")


def display_rag_results(rag_result: Dict):
    """Display RAG system results"""
    print(f"   AI Response: {rag_result['response']}")
    if rag_result['search_results']:
        print(f"   Based on search results:")
        for i, result in enumerate(rag_result['search_results'], 1):
            score = result['similarity_score'] * 100
            print(f"     {i}. {result['food_name']} ({score:.1f}% match)")


def custom_query_comparison(collections: Dict, ollama_available: bool):
    """Allow user to test custom queries across all systems"""
    print("\n🔍 CUSTOM QUERY COMPARISON")
    print("-" * 30)
    
    query = input("Enter your food search query: ").strip()
    if not query:
        print("❌ Please enter a search query")
        return
    
    print(f"\n🔍 Comparing results for: '{query}'")
    print("=" * 50)
    
    # Test all three systems
    print("\n📱 System 1: Interactive Search")
    results1 = test_interactive_search(collections['interactive'], query)
    display_comparison_results(results1, "Interactive")
    
    print("\n🔧 System 2: Advanced Search")
    results2 = test_advanced_search(collections['advanced'], query)
    display_comparison_results(results2, "Advanced")
    
    if ollama_available:
        print("\n🤖 System 3: RAG Chatbot")
        results3 = test_rag_search(collections['rag'], query)
        display_rag_results(results3)
    else:
        print("\n🤖 System 3: RAG Chatbot - ❌ Ollama not available")


def performance_benchmarking(collections: Dict, ollama_available: bool):
    """Benchmark performance of all systems"""
    print("\n⚡ PERFORMANCE BENCHMARKING")
    print("-" * 30)
    
    test_queries = ["pasta", "dessert", "healthy meal", "spicy food", "Italian cuisine"]
    
    print("Running performance tests on sample queries...")
    
    # Benchmark interactive search
    start_time = time.time()
    for query in test_queries:
        perform_similarity_search(collections['interactive'], query, 5)
    interactive_time = time.time() - start_time
    
    # Benchmark advanced search
    start_time = time.time()
    for query in test_queries:
        perform_similarity_search(collections['advanced'], query, 5)
    advanced_time = time.time() - start_time
    
    # Benchmark RAG system
    rag_time = 0
    if ollama_available:
        start_time = time.time()
        for query in test_queries:
            test_rag_search(collections['rag'], query)
        rag_time = time.time() - start_time
    
    print(f"\n📊 Performance Results (5 queries):")
    print(f"   Interactive Search: {interactive_time:.3f} seconds")
    print(f"   Advanced Search:    {advanced_time:.3f} seconds")
    if ollama_available:
        print(f"   RAG Chatbot:        {rag_time:.3f} seconds")
    else:
        print(f"   RAG Chatbot:        Not available")
    
    print(f"\n📈 Average per query:")
    print(f"   Interactive Search: {interactive_time/5:.3f} seconds")
    print(f"   Advanced Search:    {advanced_time/5:.3f} seconds")
    if ollama_available:
        print(f"   RAG Chatbot:        {rag_time/5:.3f} seconds")


def feature_matrix_comparison():
    """Display feature comparison matrix"""
    print("\n📊 FEATURE COMPARISON MATRIX")
    print("=" * 60)
    
    features = [
        ("Basic Similarity Search", "✅", "✅", "✅"),
        ("Interactive CLI", "✅", "✅", "✅"),
        ("Cuisine Filtering", "❌", "✅", "❌"),
        ("Calorie Filtering", "❌", "✅", "❌"),
        ("Combined Filters", "❌", "✅", "❌"),
        ("Natural Language Processing", "❌", "❌", "✅"),
        ("AI-Generated Responses", "❌", "❌", "✅"),
        ("Contextual Recommendations", "❌", "❌", "✅"),
        ("Conversation Memory", "❌", "❌", "✅"),
        ("Rich Formatting", "✅", "✅", "✅"),
        ("Help System", "✅", "✅", "✅"),
        ("Demo Mode", "❌", "✅", "✅")
    ]
    
    print(f"{'Feature':<30} {'Interactive':<12} {'Advanced':<10} {'RAG Chat':<10}")
    print("-" * 60)
    
    for feature, interactive, advanced, rag in features:
        print(f"{feature:<30} {interactive:<12} {advanced:<10} {rag:<10}")


def individual_system_testing(collections: Dict, ollama_available: bool):
    """Test individual systems separately"""
    print("\n🔧 INDIVIDUAL SYSTEM TESTING")
    print("-" * 30)
    print("1. Test Interactive Search System")
    print("2. Test Advanced Search System")
    print("3. Test RAG Chatbot System")
    print("4. Back to main menu")
    
    choice = input("\nSelect system to test (1-4): ").strip()
    
    if choice == '1':
        test_interactive_system(collections['interactive'])
    elif choice == '2':
        test_advanced_system(collections['advanced'])
    elif choice == '3':
        if ollama_available:
            test_rag_system(collections['rag'])
        else:
            print("❌ RAG system not available - Ollama connection failed")
    elif choice == '4':
        return
    else:
        print("❌ Invalid choice")


def test_interactive_system(collection):
    """Test interactive search system individually"""
    print("\n📱 Testing Interactive Search System")
    query = input("Enter search query: ").strip()
    if query:
        results = perform_similarity_search(collection, query, 5)
        display_comparison_results(results, "Interactive")


def test_advanced_system(collection):
    """Test advanced search system individually"""
    print("\n🔧 Testing Advanced Search System")
    query = input("Enter search query: ").strip()
    if query:
        results = perform_similarity_search(collection, query, 5)
        display_comparison_results(results, "Advanced")


def test_rag_system(collection):
    """Test RAG system individually"""
    print("\n🤖 Testing RAG Chatbot System")
    query = input("Enter your natural language query: ").strip()
    if query:
        results = test_rag_search(collection, query)
        display_rag_results(results)


def export_comparison_report(collections: Dict, ollama_available: bool):
    """Export detailed comparison report"""
    print("\n📄 EXPORTING COMPARISON REPORT")
    print("-" * 30)
    
    report_lines = []
    report_lines.append("# Food Search Systems Comparison Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("## Systems Overview")
    report_lines.append("1. **Interactive CLI Search**: Basic similarity search with user-friendly interface")
    report_lines.append("2. **Advanced Search**: Enhanced search with filtering capabilities")
    report_lines.append("3. **RAG Chatbot**: AI-powered conversational recommendations")
    report_lines.append("")
    report_lines.append("## Feature Comparison")
    report_lines.append("| Feature | Interactive | Advanced | RAG Chat |")
    report_lines.append("|---------|-------------|----------|----------|")
    report_lines.append("| Basic Search | ✅ | ✅ | ✅ |")
    report_lines.append("| Filtering | ❌ | ✅ | ❌ |")
    report_lines.append("| AI Responses | ❌ | ❌ | ✅ |")
    report_lines.append("| Natural Language | ❌ | ❌ | ✅ |")
    report_lines.append("")
    
    # Test sample query
    test_query = "chocolate dessert"
    report_lines.append(f"## Sample Query Test: '{test_query}'")
    
    # Interactive results
    results_interactive = test_interactive_search(collections['interactive'], test_query)
    report_lines.append("### Interactive Search Results:")
    for i, result in enumerate(results_interactive[:3], 1):
        score = result['similarity_score'] * 100
        report_lines.append(f"{i}. {result['food_name']} ({score:.1f}% match)")
    
    # Advanced results
    results_advanced = test_advanced_search(collections['advanced'], test_query)
    report_lines.append("\n### Advanced Search Results:")
    for i, result in enumerate(results_advanced[:3], 1):
        score = result['similarity_score'] * 100
        report_lines.append(f"{i}. {result['food_name']} ({score:.1f}% match)")
    
    # RAG results
    if ollama_available:
        results_rag = test_rag_search(collections['rag'], test_query)
        report_lines.append("\n### RAG Chatbot Results:")
        report_lines.append(f"AI Response: {results_rag['response']}")
    
    report_lines.append("")
    report_lines.append("## Recommendations")
    report_lines.append("- Use **Interactive Search** for simple, quick queries")
    report_lines.append("- Use **Advanced Search** for precise filtering needs")
    report_lines.append("- Use **RAG Chatbot** for natural conversation and explanations")
    
    # Write report to file
    with open('system_comparison_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("✅ Report exported to 'system_comparison_report.md'")
    print("📊 Summary:")
    print(f"   - Interactive Search: {len(results_interactive)} results")
    print(f"   - Advanced Search: {len(results_advanced)} results")
    if ollama_available:
        print(f"   - RAG Chatbot: AI response generated")
    else:
        print(f"   - RAG Chatbot: Not available")


if __name__ == "__main__":
    main()