import flask
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

app = flask.Flask(__name__, template_folder='templates')

books = pd.read_csv('recommender/model/books.csv', encoding='utf-8-sig')
ratings = pd.read_csv('recommender/model/ratings.csv', encoding='utf-8-sig')
to_read = pd.read_csv('recommender/model/to_read.csv', encoding='utf-8-sig')

books = books[['id','title','authors']]
books.rename(columns = {'id':'book_id'}, inplace = True) 
#Removing the subtitle-title inside brackets from the 'title' column
books['title'] = books.title.str.replace(r"\(.*\)",'')
#Remove white space at the end of string
books.title = books.title.str.rstrip()

#Removing duplicate records
ratings = ratings.drop_duplicates(subset=['book_id', 'user_id'], keep='last')

ratings = ratings.reset_index()
indices = pd.Series(ratings.index, index=ratings['user_id'])
all_users = [ratings['user_id'][i] for i in range(len(ratings['user_id']))]

books = books.reset_index()
indices = pd.Series(books.index, index=books['title'])
all_books = [books['title'][i] for i in range(len(books['title']))]


books_info = pd.DataFrame()
#Add book_id to the inserted user info
def get_books_info(user_id):
    #Get past ratings history of inputed user
    books_info = ratings.loc[ratings['user_id'] == user_id]

    #Merging to get the book Id.
    books_info = pd.merge(books_info, books, on='book_id', how='inner')
    books_info = books_info[['book_id', 'title', 'rating']].sort_values(by='book_id')

    return books_info
    
#let's select the subgroup of users. 
user_subset = pd.DataFrame()

def get_top_users(books_info):
    #Obtaining a list of users who have read the same books
    user_subset = ratings[ratings['book_id'].isin(books_info['book_id'].tolist())]
    
    #Group up the rows by user id
    user_subset = user_subset.groupby(['user_id'])
    
    #Let's sort these groups too, so users who read common books with input have a higher priority
    user_subset = sorted(user_subset,  key=lambda x: len(x[1]), reverse=True)
    
    #This limit(0-100) is set because we do not want to waste too much time on every user
    top_users = user_subset[1:101]
    
    return top_users

pearson_correlation = {}
def get_pearson_correlation(top_users, books_info):
    for user_id, group in top_users:
        #Let's start by sorting the input and current user group
        group = group.sort_values(by='book_id')
        books_info = books_info.sort_values(by='book_id')
        
        nratings = len(group)

        #Get the review scores for the movies that they both have in common
        temp = books_info[books_info['book_id'].isin(group['book_id'].tolist())]
        
        if nratings<2:
            continue

        #Store them in a temporary variable
        new_user_ratings = temp['rating'].tolist()
        #Store the current user group ratings
        user_ratings = group['rating'].tolist()

        corr = pearsonr(new_user_ratings, user_ratings)
        pearson_correlation[user_id] = corr[0]
            
    return pearson_correlation

#Let's get weighted rating for recommendation dataframe
def get_recommendation_df(pearson_correlation):
    #Converting the pearson_correlation output to a dataframe
    pearson_df = pd.DataFrame(columns=['user_id', 'similarity_value'], data=pearson_correlation.items())
    
    #Getting the top 50 users based on similarity value
    top_users = pearson_df.sort_values(by='similarity_value', ascending=False)[0:50]
    
    #Getting the book and rating of top users
    top_users_rating = top_users.merge(ratings, left_on='user_id', right_on='user_id', how='inner')
    
    #Multiplies the similarity by the user's ratings to get weighted rating
    top_users_rating['weighted_rating'] = top_users_rating['similarity_value']*top_users_rating['rating']

    #Getting the sum of similarity value and weighted rating by book id
    top_users_rating = top_users_rating.groupby('book_id').sum()[['similarity_value','weighted_rating']]
    top_users_rating.columns = ['sum_similarity_value','sum_weighted_rating']
    
    recommendation_df = pd.DataFrame()
    #Now we take the weighted average by book id
    recommendation_df['weighted_average_score'] = top_users_rating['sum_weighted_rating']/top_users_rating['sum_similarity_value']
    recommendation_df['book_id'] = top_users_rating.index
    
    #Ordering the book by weighted average score
    recommendation_df = recommendation_df[recommendation_df['weighted_average_score'] >= 3.0]
    recommendation = recommendation_df.sort_values(by='weighted_average_score', ascending=False).reset_index(drop=True)
    
    return recommendation

def get_recommend_books(user_id):

    books_info = get_books_info(user_id)
    top_users = get_top_users(books_info)
    pearson_correlation = get_pearson_correlation(top_users, books_info)
    recommendation_df = get_recommendation_df(pearson_correlation)

    #Finally recommended books for the inputted user
    recommend_books = books.loc[books['book_id'].isin(recommendation_df['book_id'].tolist())]
    recommend_books = pd.merge(recommend_books, recommendation_df, on='book_id', how='inner')
    recommend_books = recommend_books.sort_values(by='weighted_average_score', ascending=False).reset_index(drop=True)

    return recommend_books



# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('home.html'))
            
    if flask.request.method == 'POST':
        user_id = flask.request.form['user_name']
        user_id = pd.to_numeric(user_id, errors='coerce')

        if user_id not in all_users:
            return(flask.render_template('negative.html',name=user_id))
        else:
            result_final = get_recommend_books(user_id)
            names = []
            authors = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][2])
                authors.append(result_final.iloc[i][3])

        return flask.render_template('positive.html',book_names=names,author_names=authors,search_user=user_id)

@app.route('/rate', methods=['GET', 'POST'])

def rate():        
    if flask.request.method == 'POST':
        book_name = flask.request.form['book_name']
        user_id = flask.request.form['user_id']
        rating = flask.request.form['rating']

        rated_book = books[books['title'] == book_name]

        app.logger.info('testing info log')
        app.logger.info(rated_book['book_id'])
        app.logger.info(user_id)
        app.logger.info(rating)

        if book_name not in all_books:
            return(flask.render_template('negative_book.html',name=book_name))
        else:
            #fields=[last_row,,'second','third']
            #with open(r'name', 'a') as f:
                #writer = csv.writer(f)
                #writer.writerow(fields)
            
            result_final = get_recommend_books(user_id)
            names = []
            authors = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][2])
                authors.append(result_final.iloc[i][3])

            return flask.render_template('positive.html',book_names=names,author_names=authors,search_user=user_id)

if __name__ == '__main__':
    app.debug = True
    app.run()
