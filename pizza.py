import pandas as pd
import numpy as np

#import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#import matplotlib.pyplot as plt
import streamlit as st

#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords,wordnet
#from nltk.stem import WordNetLemmatizer



@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv("https://raw.githubusercontent.com/tunthunchawin/pizza/main/pizza_data.csv")
	return df

st.set_page_config(layout='wide')

df = load_data()

df.columns = df.columns.str.lower()

df['ingredients']= df['ingredients'].str.replace(' and','')

df['ingredients_new'] = df.ingredients.str.split(',')


df2=df['ingredients'].str.get_dummies(',')
df3=pd.get_dummies(df.type)

df4=pd.concat([df,df2,df3],axis=1)

indices = pd.Series(df.index, index=df['fullname']).drop_duplicates()

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['ingredients'])


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

st.title("***ARE YOU BORED CHOOSING PIZZA?***")

st.write("This wep-app is created to help you find the least similarty of ingredients on pizza in order to make you more enjoyable with your meals."" "
	"To acquire the result, we apply some of the mathametic formula which is so-called Cosine similarity to the model.")

st.latex(r'''similarity = \sum_{i=1}^{n} A_{i}B_{i}(\frac{1}{\sum_{i=1}^{n}\sqrt{A_{i}^2}{\sum_{i=1}^{n}\sqrt{B_{i}^2}}})
''')

st.write("Anyway, don't worry about it. Just go scroll down!!!")

st.title('**MENU**')
st.write('***PIZZA***')

st.sidebar.image("pizza logo.jpg", use_column_width=True,caption="SLICES PIZZARIA")

#st.sidebar.write('**MENU**')

#add_selectbox = st.sidebar.selectbox(
    #'PIZZA',
    #(df.fullname.tolist()))

#add_selectbox2 = st.sidebar.selectbox(
    #'CRUST',
    #('PAN','CRISPY','THIN'),key = "<uniquevalueofsomesort>"
#)

#add_selectbox3 = st.sidebar.selectbox(
    #'SIZE',
    #('S','M','L'),key = "<uniquevalueofsomesort>"
#)

age = st.slider('Select a page', 1, 3,1)
st.write("Page", age)

def slider_pizza():
	if age ==1:

		col1, col2, col3 ,col4, col5= st.beta_columns([5,5,5,5,5])

		with col1:
			st.image("Cheesy Bacon Overload.png", use_column_width=True)
			st.write("Cheesy Bacon Overload")
			st.image("Bacon & Pineapple.png", use_column_width=True)
			st.write("Bacon & Pineapple")
			
			
		with col2:
			st.image("Cocktail Sausage & Crab Sticks.png", use_column_width=True)
			st.write("Cocktail Sausage & Crab Sticks")
			st.image("Crab Sticks & Bacon Bit.png", use_column_width=True)
			st.write("Crab Sticks & Bacon Bit")
			
			
		with col3:
			st.image("BBQ Chicken.png", use_column_width=True)
			st.write("BBQ Chicken")
			st.image("Mighty Meat.png", use_column_width=True)
			st.write("Mighty Meat")
			
		with col4:
			st.image("Pizza Dipper Classic.png", use_column_width=True)
			st.write("Pizza Dipper Classic")
			st.image("Chicken Deluxe.png", use_column_width=True)
			st.write("Chicken Deluxe")
			
		with col5:
			st.image("Chili Chicken.png", use_column_width=True)
			st.write("Chili Chicken")
			st.image("Double Pepperoni.png", use_column_width=True)
			st.write("Double Pepperoni")
	if age ==2:

		col1, col2, col3 ,col4, col5= st.beta_columns([5,5,5,5,5])

		with col1:
			st.image("Super Deluxe.png", use_column_width=True)
			st.write("Super Deluxe")
			st.image("Shrimp Cocktail.png", use_column_width=True)
			st.write("Shrimp Cocktail")
			
		with col2:
			st.image("Seafood Deluxe.png", use_column_width=True)
			st.write("Seafood Deluxe")
			st.image("Classic Mushroom & Tomato.png", use_column_width=True)
			st.write("Classic Mushroom & Tomato")
			
		with col3:
			st.image("Seafood Cocktail.png", use_column_width=True)
			st.write("Seafood Cocktail")
			st.image("Tom Yum Kung.png", use_column_width=True)
			st.write("Tom Yum Kung")
			
		with col4:
			st.image("Super Seafood.png", use_column_width=True)
			st.write("Super Seafood")
			st.image("Ham&Crab Sticks.png", use_column_width=True)
			st.write("Ham&Crab Sticks")
			
		with col5:
			st.image("Meat Deluxe.png", use_column_width=True)
			st.write("Meat Deluxe")
			st.image("Hawaiian.png", use_column_width=True)
			st.write("Hawaiian")

	if age ==3:

		col1, col2, col3 ,col4, col5= st.beta_columns([5,5,5,5,5])

		with col1:
			
			st.image("Chicken Trio.png", use_column_width=True)
			st.write("Chicken Trio")
			
		with col2:
			
			st.image("Double Cheese.png", use_column_width=True)
			st.write("Double Cheese")
			
		with col3:
			
			st.image("Freestyle.png", use_column_width=True)
			st.write("Freestyle")
slider_pizza()


st.write("**NEW YORK PIZZA**")
age2 = st.slider('Select a page', 1, 2,1, key = "<uniquevalueofsomesort>")
st.write("Page", age2)


def slider_pizza2():
	if age2 ==1:

		col1, col2, col3 ,col4, col5= st.beta_columns([5,5,5,5,5])

		with col1:
			st.image("Parma Ham New York Pizza.png", use_column_width=True)
			st.write("Parma Ham New York Pizza")
			st.image("Super Deluxe New York Pizza.png", use_column_width=True)
			st.write("Super Deluxe New York Pizza")
			
			
		with col2:
			st.image("Carbonara New York Pizza.png", use_column_width=True)
			st.write("Carbonara New York Pizza")
			st.image("Seafood Deluxe New York Pizza.png", use_column_width=True)
			st.write("Seafood Deluxe New York Pizza")
			
			
		with col3:
			st.image("Seafood Mania New York Pizza.png", use_column_width=True)
			st.write("Seafood Mania New York Pizza")
			st.image("Seafood Cocktail New York Pizza.png", use_column_width=True)
			st.write("Seafood Cocktail New York Pizza")
			
		with col4:
			st.image("Mighty Meat New York Pizza.png", use_column_width=True)
			st.write("Mighty Meat New York Pizza")
			st.image("Meat Deluxe New York Pizza.png", use_column_width=True)
			st.write("Meat Deluxe New York Pizza")
			
		with col5:
			st.image("Double Pepperoni New York Pizza.png", use_column_width=True)
			st.write("Double Pepperoni New York Pizza")
			st.image("Shrimp Cocktail New York Pizza.png", use_column_width=True)
			st.write("Shrimp Cocktail New York Pizza")
	if age2 ==2:

		col1, col2, col3 ,col4, col5= st.beta_columns([5,5,5,5,5])

		with col1:
			st.image("Tom Yum Kung New York Pizza.png", use_column_width=True)
			st.write("Tom Yum Kung New York Pizza")
			
			
		with col2:
			st.image("Ham&Crab Sticks New York Pizza.png", use_column_width=True)
			st.write("Ham&Crab Sticks New York Pizza")
			
			
		with col3:
			st.image("Hawaiian New York Pizza.png", use_column_width=True)
			st.write("Hawaiian New York Pizza")
			
			
		with col4:
			st.image("Chicken Trio New York Pizza.png", use_column_width=True)
			st.write("Chicken Trio New York Pizza")
			
			
		with col5:
			st.image("Double Cheese New York Pizza.png", use_column_width=True)
			st.write("Double Cheese New York Pizza")
			
	
slider_pizza2()





def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=False)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['fullname'].iloc[movie_indices]


get_recommendations('Shrimp Cocktail PIZZA')




st.title('***Let we help you find out the least similarity of the pizza taste***')

pick1 = st.selectbox('Pick one',df.fullname.tolist())

col1, col2, col3 = st.beta_columns([5,5,5])

with col2:
	st.image(f'{pick1}.png',width=None,caption=f'{pick1}')


ingred =df[df['fullname']==pick1].ingredients.tolist()

st.selectbox(f'Ingredients of {pick1}',ingred)

st.write('**We recoomend these!!! :sunglasses:**')
st.table(pd.DataFrame(get_recommendations(f'{pick1}')).reset_index().drop(columns=['index']))

x = pd.DataFrame(get_recommendations(f'{pick1}')).reset_index().drop(columns=['index'])
y = x.fullname.tolist()

def slider_pizza3():
	col1, col2, col3 ,col4, col5 = st.beta_columns([5,5,5,5,5])

	with col1:
		st.image(f"{y[0]}.png", use_column_width=True)
		st.write(f"{y[0]}")
		st.image(f"{y[5]}.png", use_column_width=True)
		st.write(f"{y[5]}")
	with col2:
		st.image(f"{y[1]}.png", use_column_width=True)
		st.write(f"{y[1]}")
		st.image(f"{y[6]}.png", use_column_width=True)
		st.write(f"{y[6]}")
	with col3:
		st.image(f"{y[2]}.png", use_column_width=True)
		st.write(f"{y[2]}")
		st.image(f"{y[7]}.png", use_column_width=True)
		st.write(f"{y[7]}")
	with col4:
		st.image(f"{y[3]}.png", use_column_width=True)
		st.write(f"{y[3]}")
		st.image(f"{y[8]}.png", use_column_width=True)
		st.write(f"{y[8]}")
	with col5:
		st.image(f"{y[4]}.png", use_column_width=True)
		st.write(f"{y[4]}")
		st.image(f"{y[9]}.png", use_column_width=True)
		st.write(f"{y[9]}")
		
	
slider_pizza3()



st.title('***If you are in two minds, just random it***')
st.write('Push the random button')
if st.button('RANDOM'):
	pick2 = df.fullname.sample().tolist()[0]

	st.write(pick2)

	col1, col2, col3 = st.beta_columns([5,5,5])
	with col2:
		st.image(f'{pick2}.png',width=None,caption=f'{pick2}')

	

	ingred2 =df[df['fullname']==pick2].ingredients.tolist()

	st.selectbox(f'Ingredients of {pick2}',ingred)
	st.write('**We recoomend these!!!** :sunglasses:')



	st.table(pd.DataFrame(get_recommendations(f'{pick2}')).reset_index().drop(columns=['index']))

	d = pd.DataFrame(get_recommendations(f'{pick2}')).reset_index().drop(columns=['index'])
	z = d.fullname.tolist()


	col1, col2, col3 ,col4, col5 = st.beta_columns([5,5,5,5,5])

	with col1:
		st.image(f"{z[0]}.png", use_column_width=True)
		st.write(f"{z[0]}")
		st.image(f"{z[5]}.png", use_column_width=True)
		st.write(f"{z[5]}")
	with col2:
		st.image(f"{z[1]}.png", use_column_width=True)
		st.write(f"{z[1]}")
		st.image(f"{z[6]}.png", use_column_width=True)
		st.write(f"{z[6]}")
	with col3:
		st.image(f"{z[2]}.png", use_column_width=True)
		st.write(f"{z[2]}")
		st.image(f"{z[7]}.png", use_column_width=True)
		st.write(f"{z[7]}")
	with col4:
		st.image(f"{z[3]}.png", use_column_width=True)
		st.write(f"{z[3]}")
		st.image(f"{z[8]}.png", use_column_width=True)
		st.write(f"{z[8]}")
	with col5:
		st.image(f"{z[4]}.png", use_column_width=True)
		st.write(f"{z[4]}")
		st.image(f"{z[9]}.png", use_column_width=True)
		st.write(f"{z[9]}")

st.sidebar.write("Data source:https://www.1112.com")
st.sidebar.write("Coder:tun thunchawin")








