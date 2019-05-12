from django.contrib import admin

from .models import Stock, Prediction, Simulation, ChangePointPriorScale

admin.site.register(Stock)
admin.site.register(Prediction)
admin.site.register(Simulation)
admin.site.register(ChangePointPriorScale)
