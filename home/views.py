from base64 import encode
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.views import View
from django.utils.safestring import mark_safe
import random

# library Keras
import numpy as np
import pandas as pd
import json

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from home.models import User







model_RNN = load_model('E:/Do An/Source code/FoodRecomendation/home/models/food_model.h5')
model_collaborative = load_model('E:/Do An/Source code/FoodRecomendation/home/models/food_datatest.h5')

foods = pd.read_csv('E:/Do An/Source code/FoodRecomendation/home/models/food_recipe.csv')

list_food = list(foods.recipe_id.unique())
list_user = list(foods.user_id.unique())

list_user = random.sample(list_user, 50)

foods_arr = np.array(list_food)

reviews = pd.DataFrame()
reviews['review'] = foods['review']
reviews['rating'] = foods['rating']

X_train = reviews['review'][:40000]

vocab_size = 1000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def convertDataFrameToObject(data):
    
    data_json_raw = json.loads(data.to_json(orient='records'))
    data_json_dump = json.dumps(data_json_raw,indent=4)
    result = json.loads(data_json_dump)
    
    return result

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def decode_pred(pred):
    preds = np.array(pred)
    pred_classes = []
    for pred in preds:
        if pred[0] >=0 and pred[0] <= 0.3:
            pred_classes.append(1)
        elif pred[0] >0.3 and pred[0] <= 0.5:
            pred_classes.append(2)
        elif pred[0] >0.5 and pred[0] <= 0.7:
            pred_classes.append(3)
        elif pred[0] >0.7 and pred[0] <= 0.9:
            pred_classes.append(4)
        else:
            pred_classes.append(5)
    return pred_classes



# Create your views here.
class HomePageView(View):
    def get(self,request):
        print(request.session['userId'])
        if request.session['userId'] is None:
            return HttpResponseRedirect('/login/')

        if request.session['userId']:
            userId = request.session.get('userId')
            user = np.array([userId for i in range(len(list_food))])
            pred = model_collaborative.predict([foods_arr,user])

            pred = pred.reshape(-1) 
            pred_ids = np.random.choice(pred.shape[0], 6, replace=False)  
            pred_ids

            top_foods_dataframe = foods.iloc[pred_ids]
   
            
            top_foods = convertDataFrameToObject(top_foods_dataframe)

            for food in top_foods:
                food['ingredients'] = food['ingredients'].replace("'", "\"")
                if is_json( food['ingredients']):
                    food['ingredients'] = json.loads(food['ingredients'])
                else:
                    food['ingredients'] = ''

            param = {
                'top_foods': top_foods,
                
            }
            return render(request,'index.html',param)
       
    

class LoginView(View):
    def get(self, request):
        

        param={
            'users': list_user
        }
        return render(request, 'login.html',param)

    def post(self, request):
        username = int(request.POST.get('searchValue'))

        request.session['userId'] = username

        return HttpResponseRedirect('/homePage/')


class FoodDetail(View):
    def get(self,request,pk):
        if request.session['userId'] is None:
            return HttpResponseRedirect('/login/')
        
        food__list_current = convertDataFrameToObject(foods.loc[foods['recipe_id'] == pk])
        user_review_list = []

        for food in food__list_current:
            # Thành phần món ắn
            food['ingredients'] = food['ingredients'].replace("'", "\"")
            if is_json( food['ingredients']):
                food['ingredients'] = json.loads(food['ingredients'])
            else:
                food['ingredients'] = ''

            # Các bước thực hiện
            food['steps'] = food['steps'].replace("'", "\"")
            if is_json( food['steps']):
                food['steps'] = json.loads(food['steps'])
            else:
                food['steps'] = ''

            # Lấy các các đánh giá của người dùng về món ăn
            user_review = {
                'user_id': food['user_id'],
                'created_date': food['date'],
                'review': food['review']
            }
            user_review_list.append(user_review)

        food_current = food__list_current[0]

        user_review_list_limit = []

        if len(user_review_list) >= 3:
            user_review_list_limit = random.sample(user_review_list,3)
        else:
            user_review_list_limit = user_review_list

        print(user_review_list_limit)
        param = {
            'food': food_current,
            'user_review': user_review_list_limit

        }
        
        return render(request,'foodDetail.html',param)

    def post(self,request,pk):
        if request.POST.get('reviewByUser'):
            commentValue = request.POST.get('commentValue')
            reviews =[
                commentValue
            ]

            X_new = tokenizer.texts_to_sequences(reviews)

            X_new_padded = pad_sequences(X_new, padding='post')

            X_new_padded.resize(1,50)

            pred = model_RNN.predict(X_new_padded)
            pred_classes = decode_pred(pred)
            print(pred_classes)
            param = {
                'result': pred_classes
            }

            return JsonResponse(param)
        
