import streamlit as st
from portfolio_optimizer import optimize_portfolio
from train import load_trained_model, recommend_portfolio

# Load trained model
model = load_trained_model()

# Set page layout
st.set_page_config(page_title="AI-Powered Robo-Advisory System", layout="wide")

# Centered title
st.markdown("<h1 style='text-align: center; color: black;'>AI-Powered Robo-Advisory System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Centering using empty columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Middle column for centering
    st.markdown("### 📈 Stock Selection")

    # Stock input
    stocks = st.text_input("Enter stock symbols (comma-separated)", "AAPL,GOOGL,MSFT,AMZN,TSLA")

    # Investment amount
    investment = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=1000)

    # Optimize button
    if st.button("🔍 Optimize Portfolio"):
        stocks_list = [s.strip().upper() for s in stocks.split(",")]

        # Get optimized portfolio
        optimized_allocation = optimize_portfolio(stocks_list)

        # Get AI-recommended allocation
        ai_recommendation = recommend_portfolio(model, optimized_allocation)

        st.session_state["optimized"] = optimized_allocation
        st.session_state["ai_recommendation"] = ai_recommendation

# Display results
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 💹 Optimized Portfolio Allocation", unsafe_allow_html=True)

if "optimized" in st.session_state:
    st.json(st.session_state["optimized"])

st.markdown("### 🤖 AI-Recommended Portfolio", unsafe_allow_html=True)

if "ai_recommendation" in st.session_state:
    st.json(st.session_state["ai_recommendation"])

# Footer
st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Built with ❤️ using Streamlit</h5>", unsafe_allow_html=True)
