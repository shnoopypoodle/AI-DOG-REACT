import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

def classify_sentence(sentence_to_classify):
    """
    Classifies a Thai sentence using few-shot prompting for improved accuracy.
    Returns one of three categories: 1-บอกรัก(love), 2-ไปเที่ยว(sad), or 3-ผ้าขนหนู(appreciate).
    """
    print(f"Input: '{sentence_to_classify}'")

    # Enhanced system prompt with clear instructions and examples
    system_prompt = (
        "You are an expert emotion classifier named garden(การ์เด้น) for Thai language text. "
        "Your task is to classify Thai sentences into exactly one of these three emotion categories:\n\n"
        #fine tune
        "1. 'love' - Expressions of romantic feelings, affection, caring, or warmth towards someone\n"
        "2. 'sad' - Expressions of sadness, disappointment, loneliness, separation, or melancholy\n"
        "3. 'appreciate' - Expressions of admiration, amazement, being impressed, appreciation, or feeling proud\n\n"
        "Instructions:\n"
        "- Analyze the emotional tone and context of the Thai sentence carefully\n"
        "- Consider Thai cultural expressions and idioms\n"
        "- Respond with ONLY one word: either 'love', 'sad', or 'appreciate'\n"
        "- Do not include any explanation, punctuation, or additional text"
    )

    examples = [
        #fine tunable
        {"role": "user", "content": "รักเธอมากที่สุดในโลก"},
        {"role": "assistant", "content": "love"},
        {"role": "user", "content": "การ์เด้นน่ารักจังเลย, รักการ์เด้นนะคะ"},
        {"role": "assistant", "content": "love"},
        {"role": "user", "content": "เหงามากเลย ไม่มีใครเข้าใจ"},
        {"role": "assistant", "content": "sad"},
        {"role": "user", "content": "เก่งมากเลย ทำได้ดีจริงๆ"},
        {"role": "assistant", "content": "appreciate"},

    ]
    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(examples)
        messages.append({"role": "user", "content": sentence_to_classify})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=10
        )

        category = response.choices[0].message.content.strip().lower()

        #control response to be only in these 3 emotions
        valid_categories = ['love', 'sad', 'appreciate']
        if category not in valid_categories:
            print(f"Unexpected category '{category}', attempting to match")
            for valid in valid_categories:
                if valid in category:
                    category = valid
                    break
            else:
                print(f"Could not determine valid category from response: '{category}'")
                return None

        return category

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return None

#Section for testing python file in terminal
if __name__ == "__main__":
    print("Starting the system")
    print("Categories: love, sad, appreciate")
    print("Type 'quit' or 'exit' to stop\n")
    

    while True:
        try:
            user_input = input("Enter a sentence to classify emotion: ").strip() #input section will be change to receive text from speech to text model
            if user_input.lower() in ['quit', 'exit']:
                print("\nExiting system")
                break

            if not user_input:
                print("Please enter a sentence.\n")
                continue

            category = classify_sentence(user_input)

            if category:
                print(f"\n Emotion: {category}\n")
            else:
                print("\n Classification failed. Please try again.\n")

        except KeyboardInterrupt:
            print("Keyboard Interrupted")
            break
        except EOFError:
            print("Exiting --")
            break
