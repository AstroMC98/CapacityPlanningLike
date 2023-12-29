import streamlit as st

def get_lob_filters(activity_log):

    city = st.selectbox("Select a City/Country",
                        list(activity_log['City'].unique()),
                        index = 1)
    
    market = st.selectbox("Select a Market/Language",
                        list(activity_log[activity_log['City'] == city]['Market'].unique()),         
    )
    
    role = st.selectbox("Select a Role",
                        list(activity_log[(activity_log['City'] == city) & (activity_log['Market'] == market)]['Role'].unique()),
    )
        
    return city, market, role