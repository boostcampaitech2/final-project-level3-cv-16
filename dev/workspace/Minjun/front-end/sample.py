    '''
    sample_title = '<p style="font-family:sans-serif; color: black; font-size: 25px;">Sample</p>'
    st.markdown(sample_title, unsafe_allow_html= True)

    col1 , col2, col3 = st.columns(3)
    with col1:
        col1_text = '<p style="font-family:courier; color:tomato; font-size: 12px;">Input</p>'
        st.markdown(col1_text, unsafe_allow_html=True)
        st.image("input1.png")

    with col2:
        col2_text = '<p style="font-family:courier; color:tomato; font-size: 12px;">Our Model</p>'
        st.markdown(col2_text, unsafe_allow_html=True)
        st.image("output2.png")

    with col3:
        col2_text = '<p style="font-family:courier; color:tomato; font-size: 12px;">Final Result</p>'
        st.markdown(col2_text, unsafe_allow_html=True)
        st.image("csv_result.png")'''