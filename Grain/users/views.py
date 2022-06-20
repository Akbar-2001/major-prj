from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

from .algorithms.ProcessAlgorithm import Algorithms
algo = Algorithms()
# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def ViewData(request):
    import os
    import pandas as pd
    from django.conf import settings
    df = pd.read_csv('media/grain_dataset.csv')
    print(df)
    df = df.to_html()

    return render(request, 'users/ViewData.html',{'data':df})


def SVR(request):
    mae, mse, r2 = algo.processSVR()

    return render(request, 'users./Svr.html', {'mae':mae, 'mse':mse, 'r2':r2})


def RandomForest(request):
    mae, mse, r2 = algo.RandomForestRegressor()

    return render(request, 'users/Rf.html',{'mae':mae, 'mse':mse, 'r2':r2})

def GradientBoosting(request):
    mae, mse, r2 = algo.GradientBoosting()
    return render(request, 'users/Gdbt.html',{'mae':mae, 'mse':mse, 'r2':r2})

def GRSVR(request):
    mae, mse = algo.GRSVR()
    return render(request, 'users/GRSVR.html', {'mae':mae, 'mse':mse})