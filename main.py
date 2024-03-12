import streamlit as st
import helper as helper
import sitemap as sitemap
def main():
    st.title("Daffodil Smart Bot")

    text_input = st.text_input("What do you want to know?")
    
    
    if st.button("Get Information"):
        if text_input.strip():
            caption = "Your answers will be here"
            data = helper.get_information(text_input)
            st.write("Answers for query:", data)
        else:
            st.warning("Please enter question to get answer.")

if __name__ == "__main__":
    main()
