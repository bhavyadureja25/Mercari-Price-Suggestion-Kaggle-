#!/usr/bin/env python
# coding: utf-8


# importing all the required libraries 
import pandas as pd
import numpy as np 
from tqdm import tqdm
import streamlit as st
from math import log
import scipy
import re
import string
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from scipy.sparse import vstack, hstack, csr_matrix
import joblib

## to import all the raw materials for prediction 

# Importing the exported pickle files
transformer_main_cat = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\sklearn_ohe1.pkl")
transformer_sub_cat = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\sklearn_ohe2.pkl")
transformer_sub_sub_cat = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\sklearn_ohe3.pkl")
transformer_brandname = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\sklearn_ohe4.pkl")
tf_idf_vector_item_description = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\tf_idf_vect_item_item_decsrp.pickle")
tf_idf_vector_name = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\tf_idf_vect_item_name.pickle")
std_scalar = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\standard_scaler_tfidf.pickle")
ridge_regression_transformer = joblib.load(r"C:\Users\User\Downloads\uoh_scripts\ridge_model_tfidf.pkl")


def featurization(df):
    featurization_df = pd.DataFrame() 
    # number of words
    featurization_df['word_count'] = df['item_description'].apply(lambda x: len(str(x).split(" ")))
    # number of characters
    featurization_df['char_count'] = df['item_description'].str.len() ## this also includes spaces
    #average word length
    def avg_word(sentence):
        try:
            words = sentence.split()
            return (sum(len(word) for word in words)/len(words))
        except:
            return np.nan
    featurization_df['avg_word'] = df['item_description'].apply(lambda x: avg_word(x))
    #number of stopwords 
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    featurization_df['stopwords'] = df['item_description'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
    #number of numerics
    featurization_df['numerics'] = df['item_description'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
    #number of upper case words
    featurization_df['upper'] = df['item_description'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))

    return featurization_df


## CLEANING THE TEXT DATA

#using stop words to clean data - 'No' has been removed since removing it has been changes the meaning of the sentance 
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]) 

def dropping_item_decsription_nans(df):
    """
    removing products without item description 
    """
    df = df.dropna(axis=0, subset=['item_description'])
    return df 

# def price_less_than_zero(df):
#     """
#     removing products with price 0 or negative
#     """
#     df = df[df['price'] > 0]
#     return df 

def remove_emoji(sentence):
    """
    Remove emojis from the string
    """
    pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return pattern.sub(r'', sentence)

def decontracted(phrase):
    """
    replacing short of strings with full strigs
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def applying_porter_stemmer(df, colname):
    """
    using porter stemmer
    """
    porter = PorterStemmer()
    df[colname] = df[colname].apply(porter.stem)
    return df 

## Compiler function

def clean_data_func(df, col_name):
#     df = price_less_than_zero(df)
    df = dropping_item_decsription_nans(df)

    preprocessed_text = []

    p_punct1 = re.compile(f"[{string.punctuation}]")
    for sentance in tqdm(df[col_name].values):
        sentance = sentance.replace('[rm]', '')
        sentance = sentance.replace('&','_')
        sentance = re.sub('[^A-Za-z0-9_]+', ' ', sentance)
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub(p_punct1, "", sentance)
        sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
        sentance = remove_emoji(sentance)
        preprocessed_text.append(sentance.strip())
    df = df.copy()
    df[col_name] = preprocessed_text
    df = applying_porter_stemmer(df, col_name)
    
    return df

## to create a function which will clean the data/inputs from the user 

def input_clean_func(input_df):
    
    #1. replacing empty cells of brand name and catgegory column with "missing"
    input_df['category_name'] = input_df['category_name'].replace([np.nan], ['missing/missing/missing'])
    input_df['brand_name'] = input_df['brand_name'].replace([np.nan], ['missing'])
    
    # 1. breaking catgeory column into 3 parts
    categories = [i.split('/') for i in input_df['category_name']]
    input_df['main_category'] = categories[0][0]
    input_df['sub_category'] = categories[0][1]
    input_df['sub_sub_category'] = categories[0][2]
    
    #3. adding the numerical features to the dataframe 
    num_features = featurization(input_df)
    
    #4. cleaning the text data of product name and item description 
    df_cleaned = clean_data_func(df=input_df, col_name = 'item_description')
    df_cleaned = clean_data_func(df=df_cleaned, col_name = 'name')
    
    # concatenating numerical features with train data 
    df_cleaned = pd.concat([df_cleaned,num_features], axis=1)
    
    df_cleaned = df_cleaned[['item_condition_id','shipping','item_description',
       'name', 'main_category', 'sub_category',
       'sub_sub_category', 'brand_name','word_count', 'char_count', 'avg_word', 'stopwords',
       'numerics', 'upper']]

    #5. applying one hot encoder to brand name and category column 
    tr_main_cat = transformer_main_cat.transform(df_cleaned['main_category'].values)
    tr_sub_cat = transformer_sub_cat.transform(df_cleaned['sub_category'].values)  
    tr_sub_sub_cat = transformer_sub_sub_cat.transform(df_cleaned['sub_sub_category'].values)
    tr_brandname = transformer_brandname.transform(df_cleaned['brand_name'].values)  
    
    #6. converting to vector format
    tfidf_name = tf_idf_vector_item_description.transform(df_cleaned['item_description'].values)
    tfidf_item_desc = tf_idf_vector_name.transform(df_cleaned['name'].values)

    #7. putting the input in the final model for prediction 
    X_train_num = df_cleaned[['word_count', 'char_count','avg_word', 'stopwords', 'numerics', 'upper',
                           'item_condition_id', 'shipping']].apply(pd.to_numeric)
    
    X_train_num = csr_matrix(X_train_num.values)
    
    final_test_df = hstack((tr_main_cat, tr_sub_cat, tr_sub_sub_cat, tr_brandname, X_train_num,\
                   tfidf_item_desc, tfidf_name)).tocsr().astype('float32')
    
    scaled_tf_idf = std_scalar.transform(final_test_df)
    
    return scaled_tf_idf
        


# ##### inputs from the user 
# 
# 1. product name - text box
# 2. brand name  - text box 
# 3. item condition - drop down
# 4. category - dowpdown # with added catgegory of missing
# 5. shipping - dropdown 
# 6. item description - text box 


## APP LAYOUT
# Heading and Sub-headings
st.title('MERCARI PRICE SUGGESTION')
st.subheader('This web app helps to suggest the appropriate product price you would like to list on the website for sale.')
st.text('')
st.text('To get a price prediction, simply enter the details required below and click')
st.text('on APPLY to get the expected price of your product')
st.text('') 

if (st.button('Click for Instructions')):
    st.markdown('**Product Title**: Enter a short, crisp and catchy title for thr product that would be visible to customers')
    st.markdown('**Brand Name**: Enter the brand to which this product belongs to. Leave empty if unknown.')
    st.markdown('**Item Condition**: Enter the existing condition of the item as best, good, normal, poor, bad.')        
    st.markdown('**Category**: Enter the appropriate category to which this item bleongs in the format main catgeory/sub category/sub-subcategory). Leave blank if you are not sure.')        
    st.markdown('**Shipping**: Select the appropriate option from the dropdown according to who will be paying the shipping fees for the given product.') 
    st.markdown('**Product Desciption**: Enter a detailed but crisp. description of the prducts explaining the condition, how the product works and why should the buyer go for it.')
st.text('')
product_name = st.text_input('Product Title', 'Enter a short, crisp and catchy title')
item_description = st.text_area('Product Decsription', 'Decsribe your product in 2-3 lines')
brand_name = st.text_input('Brand Name', 'Enter the Brand Name pf the product')
item_condition = st.selectbox('Item Condition',('Best', 'Good', 'Normal', 'Poor', 'Bad'))
category = st.text_input('Item Condition',('Enter Item consition in the format main catgeory/sub category/sub-sub category'))
shipping = st.selectbox('Shipping Fees',('Shipping to be paid by the buyer', 'Shipping to be paid by the seller'))
apply = st.button(' Apply ')


# pritning the final prediction on web app
# pritning the final prediction on web app
if apply:
    if product_name is not None:
        if item_condition is not None:
            if shipping is not None:
                
                # item condition id 
                if item_condition == 'Best':
                    item_condition_id = 1
                elif item_condition == 'Good':
                    item_condition_id = 2 
                elif item_condition == 'Normal':
                    item_condition_id = 3
                elif item_condition == 'Poor':
                    item_condition_id = 4
                else:
                    item_condition_id = 5
                    
                #shipping 
                if shipping == "Shipping to be paid by the buyer":
                    shipping_id = 1
                else:
                    shipping_id = 2
                    
                input_df = pd.DataFrame(index = [0], columns=['name','item_condition_id', 'category_name', 'brand_name','shipping', 'item_description'])
                input_df['name'][0] = product_name
                input_df['item_description'][0] = item_description
                input_df['category_name'][0] = category 
                input_df['item_condition_id'][0]  = item_condition_id
                input_df['brand_name'][0] = brand_name
                input_df['shipping'][0] = shipping_id
                
                final_tfidf_test_df = input_clean_func(input_df)
                
                #predicting price using ridge regression
                predicted_price = ridge_regression_transformer.predict(final_tfidf_test_df)

                # converting the price from log to normal 
                final_predict = np.exp(predicted_price)
                
                st.text('')
                st.write('Predicted Price is ${}'.format(final_price[0]))
                print(final_price[0])
                
            else: st.write('Please enter shipping details')
        else:st.write('Please enter Item condition')
    else:st.write('Please enter Product Name')
else: st.write('Please make sure all the information is filled before you click APPLY!!')        

########################################################################################