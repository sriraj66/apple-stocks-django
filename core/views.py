from django.shortcuts import render
from .classifier import ModelClassifier
from django.contrib.messages import success

model = ModelClassifier()
model.TrainModel()

def index(request):
    fields = [
        'Day','Closing Price','Opening Price','One Day High','One Day Low'
    ]
    data = []
    days = [i for i in range(1,32)]
    if request.method == 'POST':
        for i in fields:
            try:
                data.append(int(request.POST.get(i,0)))
            except:
                data.append(float(request.POST.get(i,0)))
        print(data)
        pridict_data = model.PredictModel(data)
        success(request, f"The Amount of Stocks On day {data[0]} : {pridict_data[0]}")

    return render(request, 'index.html',context={'field':fields,'days':days})