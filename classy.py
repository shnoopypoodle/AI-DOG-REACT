import os
from openai import OpenAI
from dotenv import load_dotenv

# Step 1: Load environment variables (your API key)
# This command looks for the .env file in your folder
load_dotenv()

# Step 2: Initialize the OpenAI client
# It will automatically read the OPENAI_API_KEY from your .env file
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure your OPENAI_API_KEY is set correctly in the .env file.")
    exit()

def classify_sentence(sentence_to_classify):
    """
    Classifies a Thai sentence using a simple "zero-shot" prompt.
    This is the Day 2 prototype.
    """
    print(f"--- Classifying sentence: '{sentence_to_classify}' ---")
    
    # This is our simple "zero-shot" prompt from Day 2 of the plan.
    # It instructs the AI on what to do without giving it examples.
    system_prompt = (
        "You are an expert text classifier. "
        "Classify the following Thai sentence into one of these three categories: love, sad, or impressed. "
        "Respond with *only* the single category name in English."
    )
    
    try:
        # Step 3: Make the API Call
        # We send the system's instructions and the user's sentence
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a fast and capable model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sentence_to_classify}
            ]
        )
        
        # Step 4: Extract the output from the response object
        # The AI's reply is inside choices[0].message.content
        category = response.choices[0].message.content
        
        # Clean up the output (remove extra spaces, make it lowercase)
        return category.strip().lower()

    except Exception as e:
        # Handle potential errors (e.g., API key is wrong, server is down)
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None

# This special block runs ONLY when you execute the script directly
# (e.g., by running `python classify.py` in your terminal)
if __name__ == "__main__":
    
    # --- This is our test sentence ---
    # You can change this to anything you want!
    test_sentence = "เค้าไปเที่ยวก่อนนะ"
    
    category = classify_sentence(test_sentence)
    
    if category:
        print(f"\n✅ Final Category: {category}")
    else:
        print("\n❌ Classification failed.")

