import os
import io
import re
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -------- SESSION STATE --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------- USER PROFILE ---------
@dataclass
class UserProfile:
    name: str
    user_type: str      # "Student"/"Professional"
    age: int
    country: str
    monthly_income: float
    risk: str           # "Low"/"Medium"/"High"
    goals: str
    def style_prompt(self):
        if self.user_type.lower().startswith("stud"):
            return "Respond as a friendly mentor to a student. Use clear, simple, supportive language, with practical examples."
        return "Respond as a professional financial advisor for a working adult. Use precise, structured language, include trade-offs."

# --------- DATA & CATEGORIZATION ---------
CATEGORIES = {
    "groceries": ["grocery", "supermarket", "food", "mart"],
    "rent": ["rent", "landlord"],
    "utilities": ["electric", "water", "gas", "utility", "internet"],
    "transport": ["uber", "ola", "fuel", "bus", "metro", "train", "cab", "petrol"],
    "entertainment": ["netflix", "spotify", "movie", "cinema", "concert", "game"],
    "health": ["pharmacy", "doctor", "hospital", "clinic", "medicine"],
    "eating_out": ["restaurant", "cafe", "bar", "eatery", "diner"],
    "shopping": ["amazon", "flipkart", "myntra", "shop", "store"],
    "income": ["salary", "stipend", "bonus", "interest", "dividend"],
}
def categorize(desc: str) -> str:
    desc_l = (desc or "").lower()
    for cat, keys in CATEGORIES.items():
        if any(k in desc_l for k in keys):
            return cat
    return "other"

def load_transactions(uploaded_file: Optional[io.BytesIO]) -> pd.DataFrame:
    # Demo data for new users or failed upload:
    data = {
        "date": pd.date_range("2025-07-01", periods=24, freq="D"),
        "description": [
            "Salary", "Rent", "Grocery Store", "Restaurant", "Metro Card", "Internet Bill",
            "Pharmacy", "Movie", "Amazon", "Fuel", "Bonus", "Electric Bill",
            "Café", "Supermarket", "Hospital", "Netflix", "Ola Ride", "Water Bill",
            "Gym", "Flipkart", "Bus", "Medicine", "Dividend", "Train"
        ],
        "amount": [
            70000, -15000, -2500, -900, -300, -800, -1200, -500, -2200, -1500, 8000, -1200,
            -450, -2100, -5000, -500, -350, -400, -1200, -1800, -200, -600, 1200, -250
        ],
    }
    if uploaded_file is None:
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.DataFrame(data)
    df["category"] = df["description"].apply(categorize)
    return df

def budget_summary(df: pd.DataFrame, monthly_income_hint: Optional[float]=None) -> Dict[str, float]:
    income = df.loc[df["amount"] > 0, "amount"].sum()
    expenses = -df.loc[df["amount"] < 0, "amount"].sum()
    net = income - expenses
    if monthly_income_hint and monthly_income_hint > 0:
        income = max(income, monthly_income_hint)
        net = income - expenses
    savings_rate = (net / income) * 100 if income > 0 else 0.0
    top_spend = (-df[df["amount"] < 0].groupby("category")["amount"].sum()).nlargest(5)
    return {
        "income_total": float(round(income, 2)),
        "expense_total": float(round(expenses, 2)),
        "net_savings": float(round(net, 2)),
        "savings_rate_pct": float(round(savings_rate, 2)),
        "top_spend_json": top_spend.to_json(),
    }

def spending_suggestions(df: pd.DataFrame, profile: UserProfile) -> List[str]:
    tips = []
    summary = budget_summary(df, monthly_income_hint=profile.monthly_income)
    if summary["net_savings"] < profile.monthly_income * 0.1:
        tips.append("Build or maintain a 3–6 month emergency fund; automate a monthly transfer to high‑yield savings.")
    cat_spend = -df[df["amount"] < 0].groupby("category")["amount"].sum()
    for cat, amt in cat_spend.sort_values(ascending=False).head(3).items():
        if amt > profile.monthly_income * 0.15:
            tips.append(f"{cat.capitalize()} spending is high (₹{int(amt)}): Set a spending cap and leverage cash-back offers where possible.")
    eat_out = -df[(df["category"] == "eating_out") & (df["amount"] < 0)]["amount"].sum()
    if eat_out > 0.07 * profile.monthly_income:
        tips.append("You are spending >7% of income on eating out. Consider meal planning and limit eating out to weekends.")
    transport = -df[(df["category"] == "transport") & (df["amount"] < 0)]["amount"].sum()
    if transport > 0.08 * profile.monthly_income:
        tips.append("Transport spend is sizable. Consider monthly passes, rideshares or optimizing travel days.")
    if profile.risk.lower() == "low":
        tips.append("Consider a conservative portfolio: higher allocation to bonds, fixed income, low volatility funds.")
    elif profile.risk.lower() == "high":
        tips.append("For high risk tolerance: diversify, use low-cost index funds with limited exposure to growth sectors.")
    if profile.user_type.lower().startswith("stud"):
        tips.append("As a student, use student discounts, avoid high-interest credit, and keep credit utilization <30%.")
    else:
        tips.append("As a professional, automate investments, optimize tax, and annually review insurance cover.")
    return tips

# --- INTENT FILTER (Optional, for finance/numbers only) ---
FINANCE_KEYWORDS = ["finance", "money", "budget", "expense", "savings", "tax", "investment", "loan", "credit", "debit", "stock", "rate", "income", "emi", "pay", "salary", "roi", "interest", "dividend", "bond", "sip", "fd", "rd", "fixed deposit", "asset", "liability", "capital"]

def is_finance_related(text):
    text_l = text.lower()
    if any(word in text_l for word in FINANCE_KEYWORDS):
        return True
    if any(char.isdigit() for char in text):
        return True
    return False

# ----------- AI PROVIDER WRAPPERS -------------
class HuggingFaceProvider:
    def __init__(self):
        if HF_AVAILABLE:
            try:
                self.gen = pipeline("text2text-generation", model="google/flan-t5-base")
            except Exception:
                self.gen = None
        else:
            self.gen = None
        self.name = "huggingface"
    def generate(self, prompt, max_tokens=512):
        if self.gen is None:
            return ("[Fallback] Unable to answer with LLM. Please try again later.")
        out = self.gen(prompt, max_length=min(1024, max_tokens), do_sample=False)
        return out[0]['generated_text'].strip()

class GraniteWatsonProvider:
    def __init__(self):
        # These env vars are expected to be set on Hugging Face Spaces for secure production
        self.api_key = os.getenv("IBM_WATSON_API_KEY", "")
        self.url = os.getenv("IBM_WATSON_URL", "")
        self.name = "granite_watson"
    def ok(self):
        return bool(self.api_key and self.url)
    def generate(self, prompt, max_tokens=512):
        # NO actual API call for demo/cost reasons – replace with real SDK/API in prod
        return "[Granite/Watson Simulated Response]\n\n" + prompt

# ----------- STREAMLIT UI ----------------------
st.set_page_config(page_title="FinanceBot", page_icon="💸", layout="wide")

with st.sidebar:
    st.title("💸 FinanceBot")
    name = st.text_input("Name", value="Rahul")
    user_type = st.selectbox("You are a", ["Student", "Professional"], index=1)
    age = st.number_input("Age", min_value=16, max_value=90, value=24)
    country = st.text_input("Country", value="India")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=70000, step=1000)
    risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
    goals = st.text_area("Goals (comma-separated)", value="build emergency fund, start SIP, save tax")
    provider_choice = st.selectbox("AI Provider", ["HuggingFace", "Granite/Watson"], index=0)
    uploaded = st.file_uploader("Transaction CSV (date,description,amount)", type=["csv"])

profile = UserProfile(
    name=name, user_type=user_type, age=int(age), country=country,
    monthly_income=float(monthly_income), risk=risk, goals=goals
)
df = load_transactions(uploaded)
summary = budget_summary(df, monthly_income_hint=profile.monthly_income)

# Providers
hf_provider = HuggingFaceProvider()
granite_provider = GraniteWatsonProvider()
provider = hf_provider if provider_choice == "HuggingFace" else granite_provider

# ----------- MAIN UI: Chat and Results -----------
col_chat, col_right = st.columns([0.62, 0.38])

with col_right:
    st.subheader("📊 Budget Summary")
    st.dataframe(df, use_container_width=True, height=240)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Income (₹)", f"{summary['income_total']:.0f}")
    m2.metric("Expenses (₹)", f"{summary['expense_total']:.0f}")
    m3.metric("Net (₹)", f"{summary['net_savings']:.0f}")
    m4.metric("Savings Rate", f"{summary['savings_rate_pct']}%")
    st.markdown("#### 🧠 Spending & Investment Suggestions")
    for tip in spending_suggestions(df, profile):
        st.write("•", tip)

with col_chat:
    st.subheader("🗨️ Ask your finance question")
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    user_msg = st.chat_input("Type your finance/numbers-related question…")
    if user_msg:
        # PREVENT OFF-TOPIC
        if not is_finance_related(user_msg):
            assistant_message = "Sorry, I can only answer questions related to finance or numbers. Please rephrase your query."
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
            with st.chat_message("assistant"):
                st.markdown(assistant_message)
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            # Demographic-aware + context-aware system prompt
            sys_prompt = (
                f"You are a finance-focused AI chatbot, expert in Indian personal finance. "
                f"User: {profile.user_type}, Age {profile.age}, Location {profile.country}, "
                f"Monthly Income ₹{profile.monthly_income:.0f}, Risk Tolerance {profile.risk}, Goals: {profile.goals}. "
                f"{profile.style_prompt()} "
                "Do NOT answer non-finance queries. Always use friendly, supportive, and context-aware explanations."
            )
            context = (
                f"Context: User's Current Budget - Income ₹{summary['income_total']}, "
                f"Expenses ₹{summary['expense_total']}, Net ₹{summary['net_savings']}, "
                f"Savings Rate {summary['savings_rate_pct']}%."
            )
            user_prompt = (
                f"{context}\nUser asked: {user_msg}\n"
                "Split your answer into: 1) Quick answer, 2) Why it matters, 3) Next steps (bullets), 4) Caution notes."
            )
            full_prompt = sys_prompt + "\n\n" + user_prompt
            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {provider.name}…"):
                    try:
                        ai = provider.generate(full_prompt, max_tokens=768)
                    except Exception as e:
                        ai = f"Provider error: {e}\nFallback: Use only rule-based advice."
                st.markdown(ai)
            st.session_state.chat_history.append({"role": "assistant", "content": ai})

st.markdown("""
---
**Disclaimer:** This chatbot provides educational information only and is _not_ financial, tax, or legal advice.
Consult a licensed professional for tailored guidance. Tax laws and investment products change frequently.
""")
