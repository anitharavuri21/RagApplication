# main.py
import streamlit as st

def main():
    st.title("Sample Streamlit App")
    st.write("This is a simple Streamlit app deployed from GitHub.")

    # User input example
    user_input = st.text_input("Enter your name:")
    if user_input:
        st.write(f"Hello, {user_input}!")

if __name__ == "__main__":
    main()
 
