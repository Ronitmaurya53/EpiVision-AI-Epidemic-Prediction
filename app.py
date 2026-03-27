# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from project.main import load_data, get_country_data, predict_cases

# st.set_page_config(page_title="EpiVision AI", layout="wide")

# #  Title
# st.title("🧠 EpiVision AI")
# st.subheader("COVID-19 Prediction Dashboard")

# # Sidebar
# df = load_data()

# country = st.sidebar.selectbox("Select Country", df.index.tolist())
# days = st.sidebar.slider("Days to Predict", 5, 30, 10)

# # Data
# data = get_country_data(df, country)
# predictions = predict_cases(data, days)

# # Future dates (FIX)
# last_date = data["Date"].iloc[-1]
# future_dates = pd.date_range(start=last_date, periods=days+1)[1:]

# # Clean Graph
# fig, ax = plt.subplots(figsize=(10,5))

# ax.plot(data["Date"], data["Cases"], label="Actual")
# ax.plot(future_dates, predictions, linestyle="--", label="Predicted")

# ax.set_title(f"{country} COVID Trend")
# ax.legend()

# plt.xticks(rotation=45)

# st.pyplot(fig)

# # Daily graph (clean)
# fig2, ax2 = plt.subplots(figsize=(10,4))

# ax2.bar(data["Date"], data["Daily"])
# ax2.set_title("Daily Cases")

# plt.xticks(rotation=45)

# st.pyplot(fig2)

# #  Metrics
# st.metric("Latest Cases", int(data["Cases"].iloc[-1]))
# st.metric("Predicted Next", int(predictions[-1]))

# #  Data
# st.dataframe(data.tail(10))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from project.main import load_data, get_country_data, predict_cases
from project.main import calculate_accuracy

st.set_page_config(page_title="EpiVision AI", layout="wide")

# Title
st.title("🧠 EpiVision AI")
st.subheader("Smart Epidemic Prediction Dashboard")

# Sidebar
df = load_data()

compare_countries = st.sidebar.multiselect(
    "🌍 Compare Countries", df.index.tolist()
)

country = st.sidebar.selectbox("🌍 Select Country", df.index.tolist())
days = st.sidebar.slider("📅 Days to Predict", 5, 30, 10)

# Data
data = get_country_data(df, country)
predictions = predict_cases(data, days)

# Future dates
last_date = data["Date"].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=days+1)[1:]

# 🔥 RISK SYSTEM
growth = data["Growth"].iloc[-1]

if growth > 0.05:
    risk = "High 🔴"
elif growth > 0.01:
    risk = "Medium 🟠"
else:
    risk = "Low 🟢"

# 📊 GRAPH 1
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data["Date"], data["Cases"], label="Actual", linewidth=2)
ax.plot(future_dates, predictions, linestyle="--", label="Predicted", linewidth=2)

ax.set_title(f"{country} COVID Trend")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# 📊 GRAPH 2
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(data["Date"], data["Daily"])
ax2.set_title("Daily Cases")
plt.xticks(rotation=45)
st.pyplot(fig2)

if compare_countries:
    st.subheader("🌍 Country Comparison")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    for c in compare_countries:
        d = get_country_data(df, c)
        ax3.plot(d["Date"], d["Cases"], label=c)

    ax3.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig3)


st.subheader("🗺️ Global Case Distribution")

map_df = df.copy()
map_df = map_df.sum(axis=1).reset_index()
map_df.columns = ["Country", "Cases"]

fig_map = px.choropleth(
    map_df,
    locations="Country",
    locationmode="country names",
    color="Cases",
    title="COVID Global Spread"
)

st.plotly_chart(fig_map)

# 🔥 METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Latest Cases", int(data["Cases"].iloc[-1]))
col2.metric("Predicted Next", int(predictions[-1]))
col3.metric("Risk Level", risk)

# 🤖 AI SUGGESTIONS
st.subheader("🤖 AI Health Advisory")

if "High" in risk:
    st.error("⚠️ High spread risk! Avoid gatherings, wear masks.")
elif "Medium" in risk:
    st.warning("⚠️ Moderate risk. Stay cautious.")
else:
    st.success("✅ Situation under control.")

# 📋 DATA
st.subheader("📋 Recent Data")
st.dataframe(data.tail(10))


mae = calculate_accuracy(data)

st.metric("Model Error (MAE)", round(mae, 2))