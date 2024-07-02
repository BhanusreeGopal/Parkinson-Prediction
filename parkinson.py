import sklearn
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# loading the saved models
parkinsons_model = joblib.load('h5_model.h5')


# Define custom CSS styles
custom_css = """
<style>
/* Sidebar styles */
.sidebar {
    background-color: #2E3B4E;
    color: #fff;
    padding: 20px;
    border-radius: 15px;
}

/* Navigation link styles */
.sidebar a {
    color: #fff;
    text-decoration: none;
    padding: 10px 0;
    display: block;
    transition: background-color 0.3s;
}

.sidebar a:hover {
    background-color: #4E637E;
    border-radius: 5px;
}

/* Main content styles */
.main-content {
    padding: 20px;
    background-color: #fff;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
"""

# CSS
st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.title('Parkinsons Prediction System')
    
    # Create a selectbox for navigation
    selected = st.selectbox(
        "Pages",
        ['Home', 'Description', 'About', 'Parkinsons Prediction', 'Appointment'],
        index=0  # Set the default index
    )
    
if selected == "Home":
     
     parkinson = pd.read_csv("parkinson.csv")

# Page title
     st.title("DATA VISUALIZATION")
     st .write("")
     #count plot
     st.title("1. COUNT PLOT")
     st .write("")
     a = sns.catplot(x='status',kind='count',data=parkinson)
     st.pyplot(a)

     st.write("Health Status :")
     st.write("1 - Parkinson")
     st.write("0 - Healthy")
     st.write("More number of peoples are suffering from parkinsons.")

     #Box plot

     st.title("2. BOX PLOT")
     st .write("")
     # Create the boxplots using Seaborn
     fig, axes = plt.subplots(2, 3, figsize=(12, 6))

     # Boxplot 1
     plt.subplot(231)
     sns.boxplot(x='status', y='MDVP:Fo(Hz)', data=parkinson)
    
     # Boxplot 2 
     plt.subplot(232)
     sns.boxplot(x='status', y='MDVP:Flo(Hz)', data=parkinson)

     # Boxplot 3
     plt.subplot(233)
     sns.boxplot(x='status', y='MDVP:Jitter(%)', data=parkinson)

     # Boxplot 4
     plt.subplot(234)
     sns.boxplot(x='status', y='MDVP:Jitter(Abs)', data=parkinson)
 
     # Boxplot 5
     plt.subplot(235)
     sns.boxplot(x='status', y='MDVP:RAP', data=parkinson)

     # Boxplot 6
     plt.subplot(236)
     sns.boxplot(x='status', y='MDVP:PPQ', data=parkinson)

     # Adjust layout
     plt.tight_layout()

     # Set the background color to be transparent
     #fig = plt.gcf()
     #fig.patch.set_facecolor('none')

     # Display the boxplots in Streamlit
     st.pyplot(fig)

     # Plot histograms for each variable
     st .title("3. HISTOGRAM")
     st .write("")
     # Set the figure size
     fig, ax = plt.subplots(figsize=(20, 12))

     # Create histograms for all columns in the DataFrame
     parkinson.hist(ax=ax)

     #title for the histograms
     plt.title("Histograms", size=18)

     # Set the background color of the figure to be transparent
     #fig = plt.gcf()
     #fig.patch.set_facecolor('none')

     # Display the histograms in Streamlit
     st.pyplot(fig)

     #bubble plot  
     st.title("4. BUBBLE PLOT") 
     st .write("")
     x=parkinson['MDVP:Fo(Hz)']
     y=parkinson['MDVP:Flo(Hz)']
     N = 195
     colors = np.random.rand(N)
     area = (25 * np.random.rand(N))**2
     parkinson1 = pd.DataFrame({'X': x,'Y': y,'Colors': colors,"bubble_size":area})
     
     # Create a figure
     fig, ax = plt.subplots()

    # Create the scatter plot
     sc = ax.scatter('X', 'Y', s='bubble_size', c='Colors', cmap='viridis', alpha=0.5, data=parkinson1)
     ax.set_xlabel("X", size=16)
     ax.set_ylabel("Y", size=16)
     ax.set_title("", size=18)
    # Display the plot in Streamlit
     st.pyplot(fig)
    

     
  # Description Page
if selected == "Description":

    # Page title
    st.title("Attributes Description")
    st.markdown("")

     # Add a description about the application
    st.info("name - ASCII subject name and recording number .")
    st.info("MDVP:Fo(Hz) - Average vocal fundamental frequency.")
    st.info("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency.") 
    st.info("MDVP:Flo(Hz) - Minimum vocal fundamental frequency.") 
    st.info("MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency.") 
    st.info("MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude.") 
    st.info("NHR,HNR - Two measures of ratio of noise to tonal components in the voice.") 
    st.info("status - Health status of the subject (one) - Parkinson's, (zero) - healthy.")
    st.info("RPDE,D2 - Two nonlinear dynamical complexity measures.")
    st.info("DFA - Signal fractal scaling exponent.") 
    st.info("spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation.") 


    # About Page
if selected == "About":

    # Page title
    st.title("About Parkinson's Disease")
   
    # Add information about the creators or any other relevant details
    st.write("Parkinson's disease is a brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination. Symptoms usually begin gradually and worsen over time. As the disease progresses, people may have difficulty walking and talking. They may also have mental and behavioral changes, sleep problems, depression, memory difficulties, and fatigue.")
    st.write("One clear risk is age: Although most people with Parkinson's first develop the disease after age 60, about 5% to 10% experience onset before the age of 50. Early-onset forms of Parkinsons are often, but not always, inherited, and some forms have been linked to specific alterations in genes.")
    st.write("SYMPTOMS OF PARKINSON'S DISEASE :")
    st.markdown(" Parkinson's has four main symptoms:")
    st.markdown("1. Tremor in hands, arms, legs, jaw, or head.")
    st.markdown("2. Muscle stiffness, where muscle remains contracted for a long time.")

    # Create an expander to hide/show additional content
    with st.expander("Read More"):
        st.markdown("3. Slowness of movement.")
        st.markdown("4. Impaired balance and coordination, sometimes leading to falls.")
        st.markdown("")
        st.markdown("OTHER SYMPTOMS MAY INCLUDE :")
        st.markdown("1. Depression and other emotional changes.")
        st.markdown("2. Difficulty swallowing, chewing, and speaking.")
        st.markdown("3. Urinary problems or constipation.")
        st.markdown("4. Skin problems.")

fo=""                
    # Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction")
    st.write("")
    st.write("There are currently no blood or laboratory tests to diagnose non-genetic cases of Parkinson's. Doctors usually diagnose the disease by taking a person's medical history and performing a neurological examination. ")
    st.write("")
    #st.write("Enter The Values In The Given Attributes to predict whether the person has been affected by Parkinson's Or Not")
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo += st.text_input('MDVP:Fo(Hz)') 
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    #convert string to float
    fo = float(fo)
    fhi = float(fhi)
    flo = float(flo)
    Jitter_percent = float(Jitter_percent)
    Jitter_Abs = float(Jitter_Abs)
    RAP = float(RAP)
    PPQ = float(PPQ)
    DDP = float(DDP)
    Shimmer = float(Shimmer)
    Shimmer_dB = float(Shimmer_dB)
    APQ3 = float(APQ3)
    APQ5 = float(APQ5)
    APQ = float(APQ)
    DDA = float(DDA)
    NHR = float(NHR)
    HNR = float(HNR)
    RPDE = float(RPDE)
    DFA = float(DFA)
    spread1 = float(spread1)
    spread2 = float(spread2)
    D2 = float(D2)
    PPE = float(PPE)

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        # Define the features as a 2D list (list of lists)
        features = [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]

        # Use the 'predict' method with the features
        parkinsons_prediction = parkinsons_model.predict(features)                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

if selected == "Appointment":

    # Page title
    st.title("Make an Appointment")

    # Input fields for appointment details
    name = st.text_input("Name")
    phone_number = st.text_input("Phone Number")
    hospital_name = st.text_input("Hospital Name")
    appointment_date = st.date_input("Appointment Date")
    appointment_time = st.time_input("Appointment Time")
    reason = st.text_area("Reason for Appointment")

    # Button to schedule the appointment
    if st.button("Schedule Appointment"):
        # Assuming you have a database, you can save the appointment details there
        appointment_data = pd.DataFrame({
            "Name Of The Patient": [name],
            "Phone Number": [phone_number],
            "Hospital Name": [hospital_name],
            "Appointment Date": [appointment_date],
            "Appointment Time": [appointment_time],
            "Reason": [reason]
        })

        # Display the appointment details
        st.success("Appointment scheduled successfully!")
        st.write("Appointment Details:")
        st.write(appointment_data)

