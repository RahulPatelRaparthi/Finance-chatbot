import os
import io
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Personal Finance Chatbot", page_icon="💬", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "provider_inited" not in st.session_state:
    st.session_state.provider_inited = False
if "provider_name" not in st.session_state:
    st.session_state.provider_name = "huggingface"
if "nlp" not in st.session_state:
    st.session_state.nlp = None

class AIProvider:
    def __init__(self):
        self.name = "base"
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

class HuggingFaceProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.name = "huggingface"
        self.gen = None
        if HF_AVAILABLE:
            try:
                self.gen = pipeline("text2text-generation", model="google/flan-t5-base")
            except Exception as e:
                st.warning(f"HuggingFace pipeline failed to load: {e}")
        else:
            st.info("Transformers not installed; responses will be rule‑based only.")
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.gen is None:
            return (
                "[Rule-based fallback] "
                + prompt[:1000]
                + "\n\n(Summarized suggestion) Consider tracking expenses, setting goals, building an emergency fund, "
                  "and using diversified, low-cost index funds aligned with your risk tolerance.)"
            )
        out = self.gen(prompt, max_length=min(1024, max_tokens), do_sample=False)
        return out[0]["generated_text"].strip()

class IBMGraniteWatsonProvider(AIProvider):
    def __init__(self, watson_api_key: Optional[str], watson_url: Optional[str], granite_key: Optional[str]):
        super().__init__()
        self.name = "ibm_granite_watson"
        self.ok = bool(watson_api_key and watson_url) or bool(granite_key)
        self.watson_api_key = watson_api_key
        self.watson_url = watson_url
        self.granite_key = granite_key
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.ok:
            return "[IBM placeholder] Missing credentials — falling back text.\n" + prompt
        return (
            "[IBM Granite/Watson simulated response]\n"
            + "(Replace this with real SDK call)\n\n"
            + prompt
        )

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

@dataclass
class UserProfile:
    name: str
    user_type: str
    age: int
    country: str
    monthly_income: float
    risk_tolerance: str
    goals: str
    def style_prompt(self) -> str:
        if self.user_type.lower().startswith("stud"):
            return (
                "Explain like a friendly mentor to a student. Keep it clear and concise, "
                "use practical examples and low‑jargon."
            )
        return (
            "Explain like a professional financial coach. Be precise, structured, and include "
            "brief rationale with trade‑offs."
        )

def load_transactions(uploaded_file: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded_file is None:
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
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(uploaded_file)
    df["category"] = df["description"].apply(categorize)
    return df


def budget_summary(df: pd.DataFrame, monthly_income_hint: Optional[float] = None) -> Dict[str, float]:
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    income = df.loc[df["amount"] > 0, "amount"].sum()
    expenses = -df.loc[df["amount"] < 0, "amount"].sum()
    net = income - expenses
    if monthly_income_hint and monthly_income_hint > 0:
        income = max(income, monthly_income_hint)
        net = income - expenses
    savings_rate = (net / income) * 100 if income > 0 else 0.0
    by_cat = df.groupby("category")["amount"].sum().sort_values()
    top_spend = (-by_cat[by_cat < 0]).nlargest(5)
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
        tips.append("Build/maintain a 3–6 month emergency fund; automate a monthly transfer to high‑yield savings.")
    cat_spend = -df[df["amount"] < 0].groupby("category")["amount"].sum().sort_values(ascending=False)
    for cat, amt in cat_spend.head(3).items():
        if amt > profile.monthly_income * 0.15:
            tips.append(f"{cat.capitalize()}: spending seems high (₹{amt:.0f}). Set a category cap and use cash‑back offers.")
    eat_out = -df[(df["category"] == "eating_out") & (df["amount"] < 0)]["amount"].sum()
    if eat_out > 0.07 * profile.monthly_income:
        tips.append("Eating out >7% of income. Try a 2‑meal prep plan and limit cafés to weekends.")
    transport = -df[(df["category"] == "transport") & (df["amount"] < 0)]["amount"].sum()
    if transport > 0.08 * profile.monthly_income:
        tips.append("Transport is sizable. Consider monthly passes, ridesharing, or optimizing commute days.")
    if profile.risk_tolerance.lower() == "low":
        tips.append("Consider conservative allocation: larger share in debt/index funds, minimal high‑volatility assets.")
    elif profile.risk_tolerance.lower() == "high":
        tips.append("Higher risk tolerance: maintain diversification; use low‑cost index funds + capped exposure to growth sectors.")
    if profile.user_type.lower().startswith("stud"):
        tips.append("As a student, prioritize fee waivers, student discounts, and keep credit utilization <30%.")
    else:
        tips.append("As a professional, automate investments (SIP), optimize tax‑saving sections, and review insurance coverage.")
    return tips

INTENT_PATTERNS = {
    "savings": r"save|savings|emergency|fd|rd|goal",
    "tax": r"tax|80c|deduction|income tax|regime|tds|refund",
    "invest": r"invest|sip|mutual fund|stock|index|portfolio|asset|401k|pension",
    "budget": r"budget|spend|expense|track|summary|report",
}

def detect_intent(text: str) -> str:
    t = text.lower()
    for k, pat in INTENT_PATTERNS.items():
        if re.search(pat, t):
            return k
    return "general"


def build_system_prompt(profile: UserProfile) -> str:
    return (
        f"You are a personal finance assistant for {profile.user_type.lower()}. "
        f"User profile: age {profile.age}, country {profile.country}, monthly income ₹{profile.monthly_income:.0f}, "
        f"risk {profile.risk_tolerance}. Goals: {profile.goals}. "
        f"{profile.style_prompt()} "
        "Provide educational guidance, not legal or financial advice. Use INR symbols when relevant. "
        "If tax laws are requested, give general pointers and recommend consulting a qualified advisor."
    )


def craft_user_prompt(query: str, intent: str, summary: Dict[str, float]) -> str:
    context = (
        f"\nContext summary: Income ₹{summary['income_total']}, Expenses ₹{summary['expense_total']}, "
        f"Net ₹{summary['net_savings']}, Savings rate {summary['savings_rate_pct']}%.\n"
    )
    return (
        f"Task: Answer the user's question with step‑by‑step, actionable guidance. Intent={intent}."
        f"{context}User question: {query}\n"
        "Structure the answer as: 1) Quick answer, 2) Why it matters, 3) Next steps (bullets), 4) Caution notes."
    )

with st.sidebar:
    st.title("💬 Finance Chatbot")
    st.caption("IBM Granite/Watson with HuggingFace fallback")
    name = st.text_input("Name", value="Rahul")
    user_type = st.selectbox("I am a", ["Student", "Professional"], index=1)
    age = st.number_input("Age", min_value=16, max_value=90, value=24)
    country = st.text_input("Country", value="India")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=70000, step=1000)
    risk = st.selectbox("Risk tolerance", ["Low", "Medium", "High"], index=1)
    goals = st.text_area("Goals (comma‑separated)", value="build emergency fund, start SIP, save tax")
    st.markdown("---")
    provider_choice = st.selectbox("AI Provider", ["Auto (IBM→HF)", "IBM Granite/Watson", "HuggingFace"], index=0)
    uploaded = st.file_uploader("Upload transactions CSV (date,description,amount)", type=["csv"])

profile = UserProfile(
    name=name, user_type=user_type, age=int(age), country=country,
    monthly_income=float(monthly_income), risk_tolerance=risk, goals=goals
)

if not st.session_state.provider_inited:
    ibm_provider = IBMGraniteWatsonProvider(
        watson_api_key=os.getenv("IBM_WATSON_API_KEY"),
        watson_url=os.getenv("IBM_WATSON_URL"),
        granite_key=os.getenv("IBM_GRANITE_API_KEY"),
    )
    hf_provider = HuggingFaceProvider()
    chosen = "huggingface"
    if provider_choice.startswith("IBM") and ibm_provider.ok:
        chosen = ibm_provider.name
    elif provider_choice.startswith("Auto"):
        chosen = ibm_provider.name if ibm_provider.ok else hf_provider.name
    st.session_state.providers = {"ibm": ibm_provider, "hf": hf_provider}
    st.session_state.provider_name = chosen
    st.session_state.provider_inited = True

if provider_choice.startswith("IBM"):
    st.session_state.provider_name = (
        st.session_state.providers["ibm"].name if st.session_state.providers["ibm"].ok else "huggingface"
    )
elif provider_choice.startswith("Auto"):
    st.session_state.provider_name = (
        st.session_state.providers["ibm"].name if st.session_state.providers["ibm"].ok else "huggingface"
    )
else:
    st.session_state.provider_name = "huggingface"

provider = (
    st.session_state.providers["ibm"] if st.session_state.provider_name == "ibm_granite_watson" else st.session_state.providers["hf"]
)

col_chat, col_right = st.columns([0.62, 0.38])

with col_right:
    st.subheader("📊 Budget Summary")
    df = load_transactions(uploaded)
    st.dataframe(df, use_container_width=True, height=250)
    summary = budget_summary(df, monthly_income_hint=profile.monthly_income)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Income (₹)", f"{summary['income_total']:.0f}")
    m2.metric("Expenses (₹)", f"{summary['expense_total']:.0f}")
    m3.metric("Net (₹)", f"{summary['net_savings']:.0f}")
    m4.metric("Savings Rate", f"{summary['savings_rate_pct']:.1f}%")
    st.markdown("### 🧠 AI Spending Suggestions")
    for tip in spending_suggestions(df, profile):
        st.write("• ", tip)

with col_chat:
    st.subheader("🗣️ Ask about savings, taxes, investments, or budgeting")
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    user_msg = st.chat_input("Type your question… (e.g., How much should I invest monthly for a ₹10L goal?)")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        intent = detect_intent(user_msg)
        sys_prompt = build_system_prompt(profile)
        usr_prompt = craft_user_prompt(user_msg, intent, summary)
        final_prompt = sys_prompt + "\n\n" + usr_prompt
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {provider.name}…"):
                try:
                    ai = provider.generate(final_prompt, max_tokens=768)
                except Exception as e:
                    ai = f"Provider error: {e}\nFalling back to heuristic guidance.\n" + usr_prompt
            st.markdown(ai)
        st.session_state.chat_history.append({"role": "assistant", "content": ai})

st.markdown("""
---
**Disclaimers**  
This chatbot provides educational information only and is **not** financial, tax, or legal advice.  
Tax rules change frequently; consult a qualified professional for personalized advice.
""")
