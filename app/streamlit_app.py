import os
import requests
import streamlit as st
from dotenv import load_dotenv

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="CookGPT",
    page_icon="🍳",
    layout="wide"
)

# -------------------------
# CUSTOM CSS (PRODUCT UI)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

/* Header */
.header {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subheader {
    text-align: center;
    color: #aaa;
    margin-bottom: 20px;
}

/* Chat */
.chat-bubble-user {
    background: #2d313a;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.chat-bubble-bot {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 20px;
}

/* Cards */
.card {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

/* Input */
.stTextInput > div > div > input {
    border-radius: 10px;
}

/* Sidebar */
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR (PRODUCT CONTROLS)
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Preferences</div>', unsafe_allow_html=True)

    diet = st.selectbox("Diet Type", ["None", "Vegetarian", "Vegan", "Keto"])
    calories = st.slider("Max Calories", 100, 1000, 500)
    cuisine = st.selectbox("Cuisine", ["Any", "Indian", "Italian", "Chinese", "Mexican"])

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("Try: *'High protein Indian dinner under 500 calories'*")
    st.markdown("---")
    st.markdown("[PROJECT GITHUB LINK](https://github.com/samriddhi153/cookgpt)")

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="header">🍳 CookGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-powered personalized cooking assistant</div>', unsafe_allow_html=True)

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# DISPLAY CHAT
# -------------------------
for chat in st.session_state.history:
    st.markdown(f'<div class="chat-bubble-user"><b>You:</b><br>{chat["user"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-bubble-bot">', unsafe_allow_html=True)

    st.markdown(f"<b>CookGPT:</b><br>{chat['bot']}", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# INPUT (BOTTOM STYLE)
# -------------------------
st.markdown("---")
user_input = st.text_input("Ask CookGPT...", placeholder="e.g. Low calorie pasta")

# -------------------------
# GENERATE
# -------------------------
if st.button("Generate"):

    if not user_input.strip():
        st.warning("Please enter a query")
    else:
        try:
            with st.spinner("Cooking your recipe... 👨‍🍳"):

                enriched_query = f"{user_input} | Diet: {diet}, Calories: {calories}, Cuisine: {cuisine}"

                response = requests.post(
                    f"{BACKEND_URL}/generate",
                    json={"query": enriched_query},
                    timeout=25
                )

            if response.status_code == 200:
                data = response.json()
                recipe = data.get("recipe", "")
                nutrition_data = data.get("nutrition", {})
                total_nut = nutrition_data.get("total_nutrition", {})

                # -------------------------
                # FORMAT OUTPUT (PRODUCT STYLE)
                # -------------------------
                formatted = f"""
### 🍽️ Recipe

{recipe}

---
### 🥗 AI Nutrition Analysis
**Estimated Totals:**
- 🔥 **Calories:** {total_nut.get('calories', 0):.0f} kcal
- 🥩 **Protein:** {total_nut.get('protein', 0):.1f} g
- 🥑 **Fat:** {total_nut.get('fat', 0):.1f} g
- 🍞 **Carbs:** {total_nut.get('carbs', 0):.1f} g

*Note: These are estimates based on analyzed ingredients.*
"""

                st.session_state.history.append({
                    "user": user_input,
                    "bot": formatted
                })

            else:
                st.error(f"Server error {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Backend not running")

        except requests.exceptions.Timeout:
            st.error("Request timed out")

        except Exception as e:
            st.error(str(e))
