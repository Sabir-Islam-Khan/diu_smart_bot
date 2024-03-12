import streamlit as st
import helper as helper
import sitemap as sitemap
def main():
    st.title("Romantic Caption Generator")

    text_input = st.text_input("Enter the text for the caption:")
    
    lang_input = st.text_input("Enter the language for the caption (e.g., en for English):")
    
    if st.button("Generate Caption"):
        if text_input.strip() and lang_input.strip():
            caption = "Your caption will be generated here."
            st.write("Generated Caption:", caption)
        else:
            st.warning("Please enter text and language.")

if __name__ == "__main__":
    main()
