import streamlit as st
import pandas as pd
import numpy as np
import os
import html
import json
import google.generativeai as genai
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_gemini_api_key():
    """
    Retrieve Gemini API key from st.secrets (Streamlit Cloud) or fallback to environment (local).
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Lucas X Bookmarks",
    layout="wide"
)

# Custom CSS for Apple/iOS inspired minimalist theme and hiding Streamlit defaults
custom_css = """
<style>
/* Clean typography & background */
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #F5F5F7 !important;
    color: #1D1D1F !important;
}

/* Keep header transparent so sidebar toggle button is visible */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Hide deploy button and app options hamburger menu but keep expand button */
.stDeployButton {display: none !important;}
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E8E8ED;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #1D1D1F !important;
}

/* Style select filter components */
.stMultiSelect div[data-baseweb="select"] {
    border-radius: 10px !important;
    border: 1px solid #E8E8ED !important;
    background-color: #FFFFFF !important;
}

/* Dropdown menu styling for a finer, rounded Apple look with borders */
div[data-baseweb="menu"] {
    border-radius: 12px !important;
    border: 1px solid #D2D2D7 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
    background-color: #FFFFFF !important;
}

/* Dropdown menu list items with visible separators */
div[data-baseweb="menu"] li {
    border-bottom: 1px solid #E8E8ED !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    color: #1D1D1F !important;
}

/* Remove separator on last item */
div[data-baseweb="menu"] li:last-child {
    border-bottom: none !important;
}

/* Hover state for dropdown items */
div[data-baseweb="menu"] li:hover {
    background-color: #F5F5F7 !important;
}

/* Clean cards for tweets */
div[data-testid="stContainer"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E8E8ED !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 16px !important;
}

/* Custom rounded corners for images */
div[data-testid="stContainer"] img {
    border-radius: 50% !important;
}

/* Link customization */
a {
    color: #0066CC !important;
    text-decoration: none !important;
}
a:hover {
    text-decoration: underline !important;
}

/* Tab styling override */
button[data-baseweb="tab"] {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #86868B !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1D1D1F !important;
    border-bottom-color: #1D1D1F !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Constants
ORIGINAL_CSV = "bookmarks.csv"
CLASSIFIED_CSV = "classified_bookmarks.csv"

def process_raw_data(df):
    """
    Standardize raw columns from bookmarks.csv.
    """
    if df.empty:
        return df

    # 1. Map Author name
    if 'name' in df.columns:
        df['Author'] = df['name'].fillna('Unknown')
    elif 'screen_name' in df.columns:
        df['Author'] = df['screen_name'].fillna('Unknown')
    elif 'Author' in df.columns:
        df['Author'] = df['Author'].fillna('Unknown')
    else:
        df['Author'] = 'Unknown'

    # 2. Map screen_name / username
    if 'screen_name' in df.columns:
        df['Username'] = df['screen_name'].fillna('')
    elif 'Username' in df.columns:
        df['Username'] = df['Username'].fillna('')
    else:
        df['Username'] = ''

    # 3. Map Text (Intelligently combine full_text and note_tweet_text if available)
    if 'note_tweet_text' in df.columns and 'full_text' in df.columns:
        text_series = []
        for _, row in df.iterrows():
            note = str(row['note_tweet_text'])
            full = str(row['full_text'])
            if pd.isna(row['note_tweet_text']) or not note.strip() or note.lower() == 'nan':
                text_series.append(full)
            else:
                text_series.append(note)
        df['Text'] = text_series
    elif 'full_text' in df.columns:
        df['Text'] = df['full_text'].fillna('')
    elif 'Text' in df.columns:
        df['Text'] = df['Text'].fillna('')
    else:
        df['Text'] = ''

    # Clean HTML entities in Text (e.g., &gt; -> >)
    df['Text'] = df['Text'].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else '')

    # 4. Map Date
    date_col = None
    for c in ['tweeted_at', 'Date', 'date', 'created_at', 'timestamp']:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df['Date'] = pd.NaT

    # 5. Map URL
    url_col = None
    for c in ['tweet_url', 'URL', 'url', 'link']:
        if c in df.columns:
            url_col = c
            break
    if url_col:
        df['URL'] = df[url_col].fillna('')
    else:
        df['URL'] = ''

    # 6. Map Avatar URL
    avatar_col = None
    for c in ['profile_image_url_https', 'profile_image', 'avatar']:
        if c in df.columns:
            avatar_col = c
            break
    if avatar_col:
        df['Avatar'] = df[avatar_col].fillna('')
    else:
        df['Avatar'] = ''

    return df

def classify_bookmarks_with_gemini(df):
    """
    Classify tweets in the dataframe using Gemini 3.6 Flash.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in secrets or environment.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.6-flash")
    
    texts = df['Text'].tolist()
    if not texts:
        df['Category'] = []
        return df

    # Prepare structured input data for batching
    input_data = [{"index": idx, "text": text} for idx, text in enumerate(texts)]

    prompt = f"""
    You are an AI classifier specializing in categorizing social media posts (tweets).
    Classify each tweet from the list below into a single, accurate category label.
    
    Prefer using one of these standard categories if they fit:
    - 'AI' (for Artificial Intelligence, LLMs, machine learning, agents, prompts, etc.)
    - 'Tech' (for general software development, engineering, databases, AWS, servers, general tech)
    - 'Programming' (specifically code, software design patterns, backend/frontend development, languages like Clojure/PHP)
    - 'Finance' (for finance, investment banking, VC, search funds, careers in finance, salaries)
    - 'Design' (for UI/UX, product design, front-end visuals)
    - 'Humor' (for funny, jokes, sarcastic remarks)
    
    If none of the above categories fit, you may generate a suitable new single-word category label (e.g., 'Marketing', 'Productivity', etc.).
    
    Return ONLY a JSON object containing a "categories" list, where each item is the category string for the tweet at that corresponding index. The category string MUST be a single word (no quotes, no markdown, no punctuation).
    
    Tweets:
    {json.dumps(input_data, ensure_ascii=False, indent=2)}
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        result = json.loads(response.text)
        categories = result.get("categories", [])
        
        if len(categories) == len(texts):
            # Clean up categories
            cleaned_categories = []
            for cat in categories:
                c = str(cat).replace("`", "").replace("'", "").replace('"', "").strip()
                if c:
                    c = c.split("\n")[0].split()[0]
                cleaned_categories.append(c if c else "Other")
            df['Category'] = cleaned_categories
            return df
        else:
            print("Batch size mismatch. Falling back to individual classification...")
    except Exception as e:
        print(f"Batch classification error: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to individual classification...")

    # Individual Fallback
    categories = []
    for idx, text in enumerate(texts):
        single_prompt = f"""
        Classify the following tweet into exactly one of these categories:
        'AI', 'Tech', 'Programming', 'Finance', 'Design', 'Humor' (or another single-word category if none of these fit).

        RULES:
        - Return ONLY the category name.
        - Do not include any quotes, markdown formatting, or markdown code block markers.
        - Do not include any conversational text or explanations.
        - Your output must be a single word (e.g., Finance).

        Tweet to classify:
        {text}
        """
        try:
            res = model.generate_content(single_prompt)
            cat_val = res.text.strip() if res.text else ""
            if not cat_val:
                raise ValueError("Empty response received from Gemini API.")
            
            # Clean up potential extra words, quotes or punctuation
            cat_val = cat_val.replace("`", "").replace("'", "").replace('"', "").strip()
            if cat_val:
                cat_val = cat_val.split("\n")[0].split()[0]
            else:
                cat_val = "Error"
            categories.append(cat_val)
        except Exception as e:
            print(f"Individual classification failed for tweet index {idx}: {e}")
            import traceback
            traceback.print_exc()
            categories.append("Error")
            
    df['Category'] = categories
    return df

@st.cache_data
def load_classified_data(file_path):
    """
    Load data from the pre-classified CSV.
    """
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()
        
    if df.empty:
        return df

    # Parse Date since CSV writes it as string
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    # Sort descending by date (newest first)
    if not df['Date'].isna().all():
        df = df.sort_values(by='Date', ascending=False)
        
    return df

# Main Title
st.title("Lucas X Bookmarks")

# Check if pre-classified file exists
df = None
if os.path.exists(CLASSIFIED_CSV):
    df = load_classified_data(CLASSIFIED_CSV)
else:
    # If not classified yet, we need to run classification
    if not os.path.exists(ORIGINAL_CSV):
        st.error("Original CSV not found.")
        st.warning("Please ensure there is a bookmarks.csv file in the project directory containing your bookmarks data.")
        st.info("The CSV should ideally contain columns like name, screen_name, full_text, tweeted_at, and tweet_url.")
    else:
        # Read the raw data
        try:
            raw_df = pd.read_csv(ORIGINAL_CSV)
        except Exception as e:
            raw_df = pd.DataFrame()
            st.error(f"Error loading original bookmarks: {e}")
            
        if raw_df.empty:
            st.warning("The bookmarks.csv file is empty.")
            st.info("Please verify the contents of the CSV file and try again.")
        else:
            # Map raw fields to standardized fields
            standardized_df = process_raw_data(raw_df)
            
            # Check API key
            api_key = get_gemini_api_key()
            if not api_key:
                st.error("GEMINI_API_KEY not found in secrets or .env file!")
                st.info("AI Classification is required to generate categories. Please:")
                st.markdown("""
                1. Create a .env file in the root directory (or configure secrets in Streamlit Cloud).
                2. Add your Gemini API key:
                   ```env
                   GEMINI_API_KEY=your_real_api_key
                   ```
                3. Refresh this page to run classification.
                """)
                st.stop()
                
            # Perform classification with Streamlit spinner
            with st.spinner("Classifying bookmarks using Gemini 3.6 Flash..."):
                try:
                    classified_df = classify_bookmarks_with_gemini(standardized_df)
                    classified_df.to_csv(CLASSIFIED_CSV, index=False)
                    st.success("Successfully classified bookmarks and saved to classified_bookmarks.csv!")
                    # Load the newly created file using the cached function
                    df = load_classified_data(CLASSIFIED_CSV)
                except Exception as e:
                    st.error(f"AI Classification failed: {e}")
                    st.stop()

if df is not None and not df.empty:
    # Sidebar Filters Header
    st.sidebar.header("Filters")

    # Author multi-select filter
    unique_authors = sorted([auth for auth in df['Author'].unique() if auth != 'Unknown'])
    # Add 'Unknown' to the list if present
    if 'Unknown' in df['Author'].values:
        unique_authors.append('Unknown')
        
    selected_authors = st.sidebar.multiselect(
        "Filter by Author",
        options=unique_authors,
        placeholder="Choose authors..."
    )

    # Category multi-select filter
    unique_categories = sorted(df['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Filter by Category",
        options=unique_categories,
        placeholder="Choose categories..."
    )

    # Keyword search input
    search_query = st.sidebar.text_input(
        "Keyword Search",
        placeholder="Search text..."
    )

    # Apply filtering
    filtered_df = df.copy()

    if selected_authors:
        filtered_df = filtered_df[filtered_df['Author'].isin(selected_authors)]
        
    if selected_categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
        
    if search_query:
        filtered_df = filtered_df[filtered_df['Text'].str.contains(search_query, case=False, na=False)]

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Bookmarks", value=len(df))
    col2.metric(label="Filtered Results", value=len(filtered_df))
    col3.metric(label="Active Filters", value=sum([bool(selected_authors), bool(selected_categories), bool(search_query)]))

    st.markdown("---")

    # Implement Tab-based navigation so user intentionally chooses the view
    tab_list, tab_analytics = st.tabs(["Bookmarks", "Analytics"])

    with tab_list:
        # Display Filtered Bookmarks List
        if filtered_df.empty:
            st.info("No bookmarks match the selected filters. Try broadening your criteria.")
        else:
            # Display each bookmark
            for idx, row in filtered_df.iterrows():
                with st.container(border=True):
                    # Setup avatar and content columns
                    avatar_url = row['Avatar']
                    # Standard placeholder if no avatar URL is provided
                    if not avatar_url or str(avatar_url).strip() == '':
                        avatar_url = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
                    
                    col_img, col_content = st.columns([1, 15])
                    
                    with col_img:
                        st.image(avatar_url, width=48)
                    
                    with col_content:
                        # Header: Author (@Username) · Date
                        author_name = row['Author']
                        username = row['Username']
                        username_str = f" (@{username})" if username else ""
                        
                        if pd.notna(row['Date']):
                            date_str = row['Date'].strftime('%b %d, %Y · %H:%M')
                        else:
                            date_str = "Unknown Date"
                            
                        st.markdown(f"**{author_name}**{username_str} · *{date_str}*")
                        
                        # Category tag/badge using CSS styled monochrome layout
                        category_name = str(row['Category']).upper()
                        st.markdown(
                            f"<span style='background-color: #E8E8ED; color: #1D1D1F; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; letter-spacing: 0.5px;'>{category_name}</span>", 
                            unsafe_allow_html=True
                        )
                        
                        # Spacer after badge since HTML element won't force markdown newline layout nicely
                        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
                        
                        # Text Content
                        st.markdown(row['Text'])
                        
                        # Spacing and View URL button
                        if row['URL']:
                            st.markdown(f"[View original tweet on X]({row['URL']})")

    with tab_analytics:
        st.subheader("Category Distribution")
        if df.empty:
            st.warning("No data available to display analytics.")
        else:
            # Group by category and count using the loaded data (df)
            category_counts = df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Create Plotly pie chart with shades of gray/silver/black
            grayscale_colors = ["#1D1D1F", "#3A3A3C", "#636366", "#8E8E93", "#AEAEB2", "#C7C7CC", "#D1D1D6", "#E5E5EA"]
            
            fig = px.pie(
                category_counts,
                names='Category',
                values='Count',
                title='All Classified Bookmarks by Category',
                hole=0.4,  # Modern Donut chart look
                color_discrete_sequence=grayscale_colors
            )
            
            # Clean layout styling
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            fig.update_layout(
                showlegend=True,
                margin=dict(t=50, b=20, l=20, r=20),
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color='#1D1D1F')
            )
            
            st.plotly_chart(fig, use_container_width=True)
