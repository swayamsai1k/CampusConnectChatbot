import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ---------------- FAQ DATA ----------------
faq = {

    "hello":
    "Hello 👋 Welcome to the Smart Enquiry Chatbot. How can I help you today?",

    "thanks":
    "You're welcome 😊 Happy to help!",

    "bye":
    "Goodbye 👋 Thank you for using Smart Enquiry Chatbot. Have a great day!",

    "tell me about the college":
    "Our college provides quality education, modern infrastructure, experienced faculty, placements, hostel facilities, sports, technical events, and industry-oriented learning.",

    "what are the college timings":
    "The college functions from 9:00 AM to 5:00 PM, Monday to Saturday.",

    "what are the working hours of the college":
    "The college working hours are from 9:00 AM to 5:00 PM, Monday to Saturday.",

    "what courses are offered":
    "We offer undergraduate and postgraduate programs in Engineering, Science, and Management.",

    "how can i apply for admission":
    "Admissions are based on entrance exam scores and merit list, followed by counseling.",

    "what is the fee structure":
    "The fee structure varies by course. For example, B.E programs start from ₹XX,XXX per year.",

    "does the college provide hostel facilities":
    "Yes, separate hostels for boys and girls are available with Wi-Fi, mess, and security.",

    "how much attendance is mandatory to write exams":
    "Students must maintain a minimum of 75% attendance to become eligible for semester examinations.",

    "which companies come here for recruitment":
    "Top recruiters include TCS, Infosys, Wipro, Accenture, Capgemini, and many other companies.",

    "what's the highest salary package offered last year":
    "The highest salary package offered last year was ₹XX LPA.",

    "can i get a 50 percent discount on tuition fees if i am a sports player":
    "Eligible sports students may receive fee concessions or scholarships based on their achievements and college policies.",

    "do you have bus services for students":
    "Yes, the college provides bus transportation services covering multiple city routes for students.",

    "can you tell me where the campus is located":
    "The campus is located at [AMC Campus, Bannerghatta Rd, Kalkere, Karnataka 560083 ] with easy transportation facilities available.",

    "is there any place to stay on campus for outstation students":
    "Yes, hostel facilities are available for outstation students with Wi-Fi, food, and security facilities.",

    "what is the eligibility for joining engineering":
    "Students must pass 12th standard with Physics, Chemistry, and Mathematics with the required cutoff marks to join engineering.",

    "tell me about events":
    "The college conducts technical fests, cultural events, sports competitions, workshops, hackathons, and seminars regularly.",

    "tell me about canteen":
    "The college canteen provides hygienic food, snacks, beverages, and seating facilities for students.",

    "tell me about exams":
    "The college conducts internal assessments, lab exams, assignments, and semester-end examinations according to the academic calendar.",

    "what is the aiml fee structure":
    "The AIML department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the cse fee structure":
    "The CSE department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the ise fee structure":
    "The ISE department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the ece fee structure":
    "The ECE department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the eee fee structure":
    "The EEE department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the mechanical fee structure":
    "The Mechanical Engineering department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the civil fee structure":
    "The Civil Engineering department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the mba fee structure":
    "The MBA department fee structure starts from ₹XX,XXX per year depending on the admission category.",

    "what is the mca fee structure":
    "The MCA department fee structure starts from ₹XX,XXX per year depending on the admission category."
}

# ---------------- CLEAN FUNCTION ----------------
def clean(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

# ---------------- LOAD AI MODEL ----------------
print("⏳ Loading AI model... Please wait.\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_questions = list(faq.keys())
faq_answers = list(faq.values())

# Create AI embeddings
faq_embeddings = model.encode(faq_questions)

# ---------------- SMART CHATBOT ----------------
def chatbot_response(question):

    user_question = clean(question)

    # ---------- SPELLING CORRECTION ----------
    corrected_question = user_question

    all_words = []

    for q in faq_questions:
        all_words.extend(clean(q).split())

    all_words = list(set(all_words))

    corrected_words = []

    for word in user_question.split():

        best_word = word
        best_score = 0

        for faq_word in all_words:

            score = fuzz.ratio(word, faq_word)

            if score > best_score:
                best_score = score
                best_word = faq_word

        # Replace misspelled word
        if best_score >= 80:
            corrected_words.append(best_word)
        else:
            corrected_words.append(word)

    corrected_question = " ".join(corrected_words)

    # ---------- AI SEMANTIC SEARCH ----------
    user_embedding = model.encode([corrected_question])

    similarity_scores = cosine_similarity(
        user_embedding,
        faq_embeddings
    )

    best_match_index = np.argmax(similarity_scores)

    best_score = similarity_scores[0][best_match_index]

    # Confidence threshold
    if best_score >= 0.40:
        return f"➜ {faq_answers[best_match_index]}"

    return "❌ Sorry, I don’t understand that question."

# ---------------- MAIN PROGRAM ----------------
print("🎓 AI SMART ENQUIRY CHATBOT")
print("Type 'exit' or 'quit' to stop the chatbot.\n")

while True:

    user_question = input("👨 You: ")

    if user_question.lower() in ["exit", "quit"]:

        print("\n🤖 Bot: Goodbye 👋 Thank you for using Smart Enquiry Chatbot!")
        break

    response = chatbot_response(user_question)

    print(f"\n🤖 Bot:\n{response}\n")