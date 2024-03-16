import streamlit as st
import helper as helper

def main():
    st.title("Daffodil Smart Bot")

    text_input = st.text_input("What do you want to know?")
    
    if st.button("Get Information"):
        if text_input.strip():
            data = helper.getInformation(text_input)
          
                
            st.success(data)
                
           
        else:
            st.warning("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
