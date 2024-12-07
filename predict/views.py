from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pandas as pd
model = joblib.load('C:\Users\ilakk\Desktop\voice_recognition\predict\random_forest_model.pkl')  
LR_model = joblib.load('C:\Users\ilakk\Desktop\voice_recognition\predict\logistic_regression_model.pkl')  
def predict(request):
    features = {
        'meanfreq': float(request.GET.get('meanfreq')),
        'sd': float(request.GET.get('sd')),
        'median': float(request.GET.get('median')),
        'Q25': float(request.GET.get('Q25')),
        'Q75': float(request.GET.get('Q75')),
        'IQR': float(request.GET.get('IQR')),
        'skew': float(request.GET.get('skew')),
        'kurt': float(request.GET.get('kurt')),
        'sp.ent': float(request.GET.get('sp.ent')),
        'sfm': float(request.GET.get('sfm')),
        'mode': float(request.GET.get('mode')),
        'centroid': float(request.GET.get('centroid')),
        'meanfun': float(request.GET.get('meanfun')),
        'minfun': float(request.GET.get('minfun')),
        'maxfun': float(request.GET.get('maxfun')),
        'meandom': float(request.GET.get('meandom')),
        'mindom': float(request.GET.get('mindom')),
        'maxdom': float(request.GET.get('maxdom')),
        'dfrange': float(request.GET.get('dfrange')),
        'modindx': float(request.GET.get('modindx'))
        
    }

    features_df = pd.DataFrame([features])

    prediction = LR_model.predict(features_df)
    prediction_prob = LR_model.predict_proba(features_df)  

    # Return the prediction as a JSON response
    return JsonResponse({'prediction': prediction[0], 'probabilities': prediction_prob.tolist()})
