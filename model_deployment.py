# Import any library used while writing streamlit code
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
# Import also libraries used in the pipeline you will load
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

select_page = st.sidebar.radio('Select page', ['Analysis', 'Model Classification'])

if select_page == 'Analysis':

    st.title('Malaria Clinical Diagnosis Analysis')

    def main():
        cleaned_df = pd.read_csv('cleaned_malaria_clinical_data.csv')
        st.image('https://ppcexpo.com/blog/wp-content/uploads/2022/01/exploratory-data-analysis.jpg')
        st.write('### Head of Dataframe')
        st.dataframe(cleaned_df.head(10))
        
        # Create 3 tabs in analysis page
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
        
        # Univariate Analysis
        tab1.write('### Univariate Analysis with Histogram for each Feature')
        for col in cleaned_df.columns:
            tab1.plotly_chart(px.histogram(cleaned_df, x= col))
         
        # Bivariate: Plot Numerical vs Target feature
        
        tab2.write('### Numerical Features vs Target Variable')
        
        select_plot = tab2.selectbox('Select PLot Type', ['Boxplot', 'Violinplot', 'Stripplot'])

        select_feature = tab2.selectbox('Select Feature', ['temperature','parasite_density', 'wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb','mean_cell_hb_conc','platelet_count','platelet_distr_width','mean_platelet_vl','neutrophils_percent',
 'lymphocytes_percent','mixed_cells_percent','neutrophils_count','lymphocytes_count','mixed_cells_count','RBC_dist_width_Percent'])

        if select_plot == 'Boxplot':
            tab2.plotly_chart(px.box(cleaned_df, 'Clinical_Diagnosis', select_feature))

        elif select_plot == 'Violinplot':
            tab2.plotly_chart(px.violin(cleaned_df, 'Clinical_Diagnosis', select_feature))

        else:
            tab2.plotly_chart(px.strip(cleaned_df, 'Clinical_Diagnosis', select_feature))
        
        # Multivaariate
        tab3.write('### Heatmap for Numerical Features')
        
        num_cols = cleaned_df.select_dtypes(exclude= 'O').columns.to_list()[:-1]
        num_cols_df = cleaned_df.select_dtypes(exclude= 'O').drop('Clinical_Diagnosis', axis= 1)
        tab3.plotly_chart(px.imshow(num_cols_df.corr(), x= num_cols))

    if __name__=='__main__':
        main()  

elif select_page == 'Model Classification':
    
    # Classification Model
    def main():
        st.title('Model Classification')
        st.image('https://th.bing.com/th/id/OIP.Ar7K9D0zbN-MISnUoFYk-QHaFj?rs=1&pid=ImgDetMain')
        
        # First step: load pkl file
        pipeline = joblib.load('malaria_clinical_diagnosis.pkl')

        # Second step: Create dataframe from input data
        def Prediction(location, Enrollment_Year, bednet, fever_symptom, temperature, Suspected_Organism,
         Suspected_infection, RDT, parasite_density, Microscopy, Laboratory_Results,
          wbc_count, rbc_count, hb_level, hematocrit, mean_cell_volume, mean_corp_hb, mean_cell_hb_conc,
           platelet_count, platelet_distr_width, mean_platelet_vl, neutrophils_percent, lymphocytes_percent,
            mixed_cells_percent, neutrophils_count, lymphocytes_count, mixed_cells_count, RBC_dist_width_Percent
):

            df = pd.DataFrame(columns= ['location', 'Enrollment_Year', 'bednet', 'fever_symptom', 'temperature',
                    'Suspected_Organism', 'Suspected_infection', 'RDT', 'parasite_density',
                    'Microscopy', 'Laboratory_Results', 'wbc_count',
                    'rbc_count', 'hb_level', 'hematocrit', 'mean_cell_volume',
                    'mean_corp_hb', 'mean_cell_hb_conc', 'platelet_count',
                    'platelet_distr_width', 'mean_platelet_vl', 'neutrophils_percent',
                    'lymphocytes_percent', 'mixed_cells_percent', 'neutrophils_count',
                    'lymphocytes_count', 'mixed_cells_count', 'RBC_dist_width_Percent' ])

            df.at[0, 'location'] = location
            df.at[0, 'Enrollment_Year'] = Enrollment_Year
            df.at[0, 'bednet'] = bednet
            df.at[0, 'fever_symptom'] = fever_symptom
            df.at[0, 'temperature'] = temperature
            df.at[0, 'Suspected_Organism'] = Suspected_Organism
            df.at[0, 'Suspected_infection'] = Suspected_infection
            df.at[0, 'RDT'] = RDT
            df.at[0, 'parasite_density'] = parasite_density
            df.at[0, 'Microscopy'] = Microscopy
            df.at[0, 'Laboratory_Results'] = Laboratory_Results
            df.at[0, 'wbc_count'] = wbc_count
            df.at[0, 'rbc_count'] = rbc_count
            df.at[0, 'hb_level'] = hb_level
            df.at[0, 'hematocrit'] = hematocrit
            df.at[0, 'mean_cell_volume'] = mean_cell_volume
            df.at[0, 'mean_corp_hb'] = mean_corp_hb
            df.at[0, 'mean_cell_hb_conc'] = mean_cell_hb_conc
            df.at[0, 'platelet_count'] = platelet_count
            df.at[0, 'platelet_distr_width'] = platelet_distr_width
            df.at[0, 'mean_platelet_vl'] = mean_platelet_vl
            df.at[0, 'neutrophils_percent'] = neutrophils_percent
            df.at[0, 'lymphocytes_percent'] = lymphocytes_percent
            df.at[0, 'mixed_cells_percent'] = mixed_cells_percent
            df.at[0, 'neutrophils_count'] = neutrophils_count
            df.at[0, 'lymphocytes_count'] = lymphocytes_count
            df.at[0, 'mixed_cells_count'] = mixed_cells_count
            df.at[0, 'RBC_dist_width_Percent'] = RBC_dist_width_Percent

            # Third step Make prediction of dataframe using pipeline
            result = pipeline.predict(df)[0]
            return result

        # Now we will decide how user can select each feature
        location = st.selectbox('Please select your location', ['Accra', 'Kintampo', 'Navrongo'])
        Enrollment_Year = st.selectbox('Please select your enrollment year', [2017, 2004, 2012, 2002, 2003, 2016, 2011, 2010])
        bednet = st.selectbox('Do you use a bednet?', ['yes', 'no'])
        fever_symptom = st.selectbox('Do you have fever symptoms?', ['Yes', 'No'])
        suspected_organism = st.selectbox('Please select the suspected organism', ['Not Known', 'Protozoan', 'Bacteria', 'Mixed Organisms', 'Viral', 'Fungi'])
        RDT = st.selectbox('Rapid Diagnostic Test (RDT) result', ['Positive', 'Negative'])
        microscopy = st.selectbox('Microscopy result', ['Positive', 'Negative'])
        suspected_infection = st.selectbox('Please select the suspected infection', [
                'Malaria', 'URTI', 'Gastroenteritis', 'Sepsis', 'Otitis media', 
                'Other - Specify'
            ])
        temperature = st.sidebar.slider('Enter your body temperature in Celsius', 34.2, 41.1)
        parasite_density = st.sidebar.slider('Enter your parasite density per microliter', 0, 92004)
        wbc_count = st.sidebar.slider('Enter your white blood cell count (x10^9/L)', 0.5, 22.4)
        rbc_count = st.sidebar.slider('Enter your red blood cell count (x10^12/L)', 1.275, 6.67)
        hb_level = st.sidebar.slider('Enter your hemoglobin level (g/dL)', 2.3, 16.7)
        hematocrit = st.sidebar.slider('Enter your hematocrit (%)', 4.5875, 52.7)
        mean_cell_volume = st.sidebar.slider('Enter your mean cell volume (fL)', 55, 95)
        mean_corp_hb = st.sidebar.slider('Enter your mean corpuscular hemoglobin (pg)', 16.2, 32.2)
        mean_cell_hb_conc = st.sidebar.slider('Enter your mean cell hemoglobin concentration (g/dL)', 26, 38)
        platelet_count = st.sidebar.slider('Enter your platelet count (x10^9/L)', 3, 599)
        platelet_distr_width = st.sidebar.slider('Enter your platelet distribution width (%)', 8.85, 19.65)
        mean_platelet_vl = st.sidebar.slider('Enter your mean platelet volume (fL)', 4.95, 10.95)
        neutrophils_percent = st.sidebar.slider('Enter your neutrophils (%)', 9.3, 93.3)
        lymphocytes_percent = st.sidebar.slider('Enter your lymphocytes (%)', 3.8, 79.6)
        mixed_cells_percent = st.sidebar.slider('Enter your mixed cells (%)', 0.3, 17.5875)
        neutrophils_count = st.sidebar.slider('Enter your neutrophils count (x10^9/L)', 0.1, 14.85)
        lymphocytes_count = st.sidebar.slider('Enter your lymphocytes count (x10^9/L)', 0.3, 8.3)
        mixed_cells_count = st.sidebar.slider('Enter your mixed cells count (x10^9/L)', 0.0, 2.15)
        RBC_dist_width_Percent = st.sidebar.slider('Enter your RBC distribution width (%)', 11.5, 22.35)

        


        if st.button('Predict'):
            result = Prediction(location, Enrollment_Year, bednet, fever_symptom, temperature, suspected_organism,
                        suspected_infection, RDT, parasite_density, microscopy, Laboratory_Results,
                        wbc_count, rbc_count, hb_level, hematocrit, mean_cell_volume, mean_corp_hb, mean_cell_hb_conc,
                        platelet_count, platelet_distr_width, mean_platelet_vl, neutrophils_percent, lymphocytes_percent,
                        mixed_cells_percent, neutrophils_count, lymphocytes_count, mixed_cells_count, RBC_dist_width_Percent)

            if result == 1:
                st.write('The diagnosis is likely Uncomplicated Malaria.')
            elif result == 0:
                st.write('The diagnosis is likely a Non-malaria Infection.')
            elif result == 2:
                st.write('The diagnosis is likely Severe Malaria.')

        
    if __name__=='__main__':
        main()
