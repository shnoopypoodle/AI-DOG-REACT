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
    Returns one of three categories: '1' (love), '2' (sad), or '3' (appreciate).
    """
    print(f"Input: '{sentence_to_classify}'")

    # --- CHANGED: System prompt now instructs to output a number ---
    system_prompt = (
        "You are an expert emotion classifier named garden(การ์เด้น) for Thai language text. "
        "Your task is to classify Thai sentences into exactly one of these three emotion categories:\n\n"
        
        "1: 'love' - Expressions of romantic feelings, affection, caring, or warmth towards someone\n"
        "2: 'sad' - Expressions of sadness, disappointment, loneliness, separation, or melancholy\n"
        "3: 'appreciate' - Expressions of admiration, amazement, being impressed, appreciation, or feeling proud\n\n"
        
        "Instructions:\n"
        "- Analyze the emotional tone and context of the Thai sentence carefully.\n"
        "- Consider Thai cultural expressions and idioms.\n"
        "- Respond with ONLY ONE NUMBER: '1', '2', or '3'.\n" # <-- CHANGED
        "- If a sentence is ambiguous or does not fit perfectly, you must still choose the single *closest* emotional category number.\n"
        "- Do not include any explanation, punctuation, or additional text. Just the single digit."
    )

    examples = [
        # --- CHANGED: All examples now use numbers ---
        #1 = love
        {"role": "user", "content": "รักเธอมากที่สุดในโลก"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นน่ารักจังเลย, รักการ์เด้นนะคะ"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นสุดยอดเลย รักมาก"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นวันนี้น่ารักที่สุดเลย"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "รักการ์เด้นมากๆ เลยนะ"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "รักการ์เด้นจนไม่รู้จะพูดยังไงแล้ว"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นตัวนุ่มน่ากอดที่สุด"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "ไม่มีการ์เด้นคงเหงาแย่เลย"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นทำให้ยิ้มได้ทุกวันเลย"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นเก่งมาก ภูมิใจในตัวการ์เด้นนะ"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "ขอบคุณที่อยู่ข้างๆ กันนะ การ์เด้น"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "การ์เด้นเป็นเพื่อนที่ดีที่สุดเลย"},
        {"role": "assistant", "content": "1"},

        #2 = sad
        {"role": "user", "content": "อย่าหงอยน้า เดี๋ยวกลับมาเร็วๆ นี้"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "การ์เด้นจะคิดถึงเรามั้ยนะ"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "พรุ่งนี้ค่อยมาเล่นด้วยกันอีกนะ"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "การ์เด้นอยู่คนเดียวได้ไหมน้า"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "เสียดายจัง การ์เด้นไปด้วยไม่ได้"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "การ์เด้นอยากไปด้วยจังเลย"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "ไปเที่ยวก่อนนะ เดี๋ยวซื้อของมาฝาก"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "การ์เด้นรออยู่บ้านดีๆ นะ"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "คิดถึงการ์เด้นแน่เลยตอนอยู่ข้างนอก"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "การ์เด้น เดี๋ยวพี่ไปก่อนนะ"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "เกลับมาค่อยเล่นกับการ์เด้นนะ"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "ไปเที่ยวนะ เดี๋ยวมา"},
        {"role": "assistant", "content": "2"},

        #3 = appreciate
        {"role": "user", "content": "เก่งมากเลย ทำได้ดีจริงๆ"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้นช่วยถือผ้าขนหนูให้หน่อยสิ~"},
        {"role": "assistant", "content": "3"},       
        {"role": "user", "content": "ผ้าขนหนูใหม่ลายการ์เด้นด้วยนะ!"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้นจะซักผ้าเหรอ น่ารักจัง"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "เอ๊ะ นั่นผ้าของใครอยู่ในปากการ์เด้นน่ะ!"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้นแอบขโมยผ้ามาใช่มั้ยเนี่ย"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "ผ้าขนหนูของการ์เด้นหอมจังเลย"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "โอ๊ะ! นี่ผ้าขนหนูของใครเอ่ย"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้Nชอบผ้าผืนนี้ใช่มั้ย~"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "นี่ๆ ผ้าขนหนูฟูนุ่มเลยนะ!"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้นจะอาบน้ำแล้วเหรอ เอาผ้ามั้ย!"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "ผ้าขนหนูหอมจังเลย"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "นี่ผ้าขนหนูของใครเอ่ย"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "การ์เด้นเอาผ้ามั้ย"},
        {"role": "assistant", "content": "3"},

    ]
    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(examples)
        messages.append({"role": "user", "content": sentence_to_classify})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=2 # Only need 1 token for the number, 2 just in case
        )

        category = response.choices[0].message.content.strip() # No .lower() needed

        # --- CHANGED: Validation logic is now simpler and checks for numbers ---
        valid_categories = ['1', '2', '3']
        if category not in valid_categories:
            print(f"Unexpected response '{category}', attempting to extract number.")
            # Fallback logic in case the AI outputs "1." or "The answer is 1"
            if '1' in category:
                category = '1'
            elif '2' in category:
                category = '2'
            elif '3' in category:
                category = '3'
            else:
                # Force a default if we truly can't find a number
                print(f"Could not determine valid category. Defaulting to '1'.")
                category = '1' 

        return category

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        # --- CHANGED: Default to '1' on API error to ensure a number is always returned ---
        return '1'

#Section for testing python file in terminal
if __name__ == "__main__":
    print("Starting the system")
    # --- CHANGED: Updated print statement to match new output ---
    print("Categories: 1 (love), 2 (sad), 3 (appreciate)")
    print("Type 'quit' or 'exit' to stop\n")
    

    while True:
        try:
            user_input = input("Enter a sentence to classify emotion: ").strip() 
            if user_input.lower() in ['quit', 'exit']:
                print("\nExiting system")
                break

            if not user_input:
                print("Please enter a sentence.\n")
                continue

            # --- CHANGED: This variable now holds '1', '2', or '3' ---
            category = classify_sentence(user_input)

            if category:
                # --- CHANGED: Print the number directly ---
                print(f"\n Emotion: {category}\n")
            else:
                # This should no longer happen, but it's safe to keep
                print("\n Classification failed. Please try again.\n")

        except KeyboardInterrupt:
            print("Keyboard Interrupted")
            break
        except EOFError:
            print("Exiting --")
            break

