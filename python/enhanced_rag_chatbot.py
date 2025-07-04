from shared_functions import *
from typing import List, Dict, Any
import json
import ollama

# Global variables
food_items = []

def main():
    """Main function for enhanced RAG chatbot system"""
    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
        print("   Powered by Llama 3.2 & ChromaDB")
        print("=" * 55)
        
        # Load food data
        global food_items
        food_items = load_food_data('../data/FoodDataSet.json')
        print(f"✅ Loaded {len(food_items)} food items")
        
        # Create collection for RAG system
        collection = create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {'description': 'Enhanced RAG chatbot with Ollama Llama 3.2 integration'}
        )
        populate_similarity_collection(collection, food_items)
        print("✅ Vector database ready")
        
        # Test LLM connection
        print("🔗 Testing Ollama connection...")
        try:
            test_response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': 'Hello'}]
            )
            if test_response and 'message' in test_response:
                print("✅ Ollama connection established")
            else:
                print("❌ Ollama connection failed")
                return
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Make sure Ollama is running and llama3.2 model is available")
            return
        
        # Start enhanced RAG chatbot
        enhanced_rag_food_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error: {error}")


def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return "No relevant food items found in the database."
    
    context_parts = []
    context_parts.append("Based on your query, here are the most relevant food options from our database:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:3], 1):
        food_context = []
        food_context.append(f"Option {i}: {result['food_name']}")
        food_context.append(f"  - Description: {result['food_description']}")
        food_context.append(f"  - Cuisine: {result['cuisine_type']}")
        food_context.append(f"  - Calories: {result['food_calories_per_serving']} per serving")
        
        # Get ingredients from metadata if available
        if 'metadatas' in result and result['metadatas']:
            metadata = result['metadatas'][0] if isinstance(result['metadatas'], list) else result['metadatas']
            if 'ingredients' in metadata:
                food_context.append(f"  - Key ingredients: {metadata['ingredients']}")
            if 'health_benefits' in metadata:
                food_context.append(f"  - Health benefits: {metadata['health_benefits']}")
            if 'cooking_method' in metadata:
                food_context.append(f"  - Cooking method: {metadata['cooking_method']}")
            if 'taste_profile' in metadata:
                food_context.append(f"  - Taste profile: {metadata['taste_profile']}")
        
        food_context.append(f"  - Similarity score: {result['similarity_score']*100:.1f}%")
        food_context.append("")
        
        context_parts.extend(food_context)
    
    return "\n".join(context_parts)


def generate_llm_rag_response(query: str, search_results: List[Dict]) -> str:
    """Generate response using Ollama Llama 3.2 with retrieved context"""
    try:
        # Prepare context from search results
        context = prepare_context_for_llm(query, search_results)
        
        # Build the prompt for the LLM
        prompt = f'''You are a food recommendation assistant. Read the database results and follow the rules exactly.

User Query: "{query}"

Retrieved Food Information:
{context}

RULES - You MUST follow these exactly:
1. If the retrieved information says "No relevant food items found in the database" → Answer: "Nothing found in my database"
2. If similarity scores are all under 30% → Answer: "Nothing found in my database"
3. If the food options don't reasonably match the user's query → Answer: "Nothing found in my database"
4. Only if you have good matches (30%+ similarity and relevant to query) → Recommend 1-2 foods from the list

Do NOT explain, do NOT be helpful beyond these rules, do NOT suggest alternatives.

Response:'''
        
        # Generate response using Ollama Llama 3.2
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Extract the generated text
        if response and 'message' in response and 'content' in response['message']:
            response_text = response['message']['content'].strip()
            
            # If response is too short, provide a fallback
            if len(response_text) < 50:
                return generate_fallback_response(query, search_results)
            
            return response_text
        else:
            return generate_fallback_response(query, search_results)
            
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return generate_fallback_response(query, search_results)


def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""
    if not search_results:
        return "Nothing found in my database"
    
    # Check if similarity scores are too low
    if all(result['similarity_score'] < 0.3 for result in search_results):
        return "Nothing found in my database"
    
    top_result = search_results[0]
    response_parts = []
    
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['food_name']}.")
    response_parts.append(f"It's a {top_result['cuisine_type']} dish with {top_result['food_calories_per_serving']} calories per serving.")
    
    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great option would be {second_choice['food_name']}.")
    
    return " ".join(response_parts)


def enhanced_rag_food_chatbot(collection):
    """Enhanced RAG-powered conversational food chatbot with Ollama Llama 3.2"""
    print("\n" + "="*70)
    print("🤖 ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by Llama 3.2 via Ollama")
    print("="*70)
    print("💬 Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please tell me what kind of food you're looking for!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            
            elif user_input.lower() in ['compare']:
                handle_enhanced_comparison_mode(collection)
            
            else:
                # Process the food query with enhanced RAG
                handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(user_input)
                
                # Keep conversation history manageable
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-3:]
                
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find something delicious! 👋")
            break
        except Exception as e:
            print(f"❌ Bot: Sorry, I encountered an error: {e}")


def handle_enhanced_rag_query(collection, query: str, conversation_history: List[str]):
    """Handle user query with enhanced RAG approach using Ollama Llama 3.2"""
    print(f"\n🔍 Searching vector database for: '{query}'...")
    
    # Perform similarity search with more results for better context
    search_results = perform_similarity_search(collection, query, 3)
    
    if not search_results:
        print("🤖 Bot: I couldn't find any matching food items in our database.")
        print("      Try describing what you're looking for with different terms.")
        return
    
    print("📊 Generating AI-powered recommendation...")
    
    # Generate enhanced response using RAG with Ollama
    ai_response = generate_llm_rag_response(query, search_results)
    
    print(f"\n🤖 Bot: {ai_response}")
    
    # Show top search results for reference
    print(f"\n📋 Reference - Top search results:")
    for i, result in enumerate(search_results, 1):
        score = result['similarity_score'] * 100
        print(f"   {i}. {result['food_name']} ({score:.1f}% match)")


def handle_enhanced_comparison_mode(collection):
    """Handle comparison between two different queries"""
    print("\n🔄 COMPARISON MODE")
    print("-" * 30)
    
    query1 = input("Enter first food query: ").strip()
    query2 = input("Enter second food query: ").strip()
    
    if not query1 or not query2:
        print("❌ Please enter both queries for comparison")
        return
    
    print(f"\n📊 Comparing '{query1}' vs '{query2}'...")
    
    # Get results for both queries
    results1 = perform_similarity_search(collection, query1, 3)
    results2 = perform_similarity_search(collection, query2, 3)
    
    # Generate responses for both
    response1 = generate_llm_rag_response(query1, results1)
    response2 = generate_llm_rag_response(query2, results2)
    
    print(f"\n🤖 For '{query1}':")
    print(f"   {response1}")
    
    print(f"\n🤖 For '{query2}':")
    print(f"   {response2}")
    
    # Show comparison summary
    print(f"\n📋 Comparison Summary:")
    print(f"   Query 1 top result: {results1[0]['food_name'] if results1 else 'None'}")
    print(f"   Query 2 top result: {results2[0]['food_name'] if results2 else 'None'}")


def show_enhanced_rag_help():
    """Display help information for enhanced RAG chatbot"""
    print("\n📖 ENHANCED RAG CHATBOT HELP")
    print("=" * 40)
    print("This chatbot uses Retrieval-Augmented Generation (RAG) to provide")
    print("intelligent food recommendations by combining:")
    print("  • Vector similarity search in our food database")
    print("  • AI-powered response generation using Llama 3.2")
    print()
    print("Natural Language Examples:")
    print("  • 'I want something healthy and low calorie'")
    print("  • 'What's good for a romantic dinner?'")
    print("  • 'I need comfort food for a rainy day'")
    print("  • 'Suggest vegetarian options with high protein'")
    print("  • 'What Italian dishes are under 500 calories?'")
    print()
    print("Features:")
    print("  • Natural conversation with AI")
    print("  • Context-aware recommendations")
    print("  • Comparison mode for different queries")
    print("  • Fallback responses when AI is unavailable")
    print()
    print("Commands:")
    print("  • 'compare' - Compare two different food queries")
    print("  • 'help' - Show this help message")
    print("  • 'quit' - Exit the chatbot")


if __name__ == "__main__":
    main()