from django.db import models


class Stock(models.Model):
    ticker_name = models.CharField(max_length=5)
    exchange = models.CharField(max_length=10)

    def __str__(self):
        return self.ticker_name + ":" + self.exchange


class ChangePointPriorScale(models.Model):
    value = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.value


class Prediction(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    number_of_days = models.IntegerField()
    change_point_prior_scale = models.ManyToManyField(ChangePointPriorScale)
    seasonality = models.BooleanField()
    current_price = models.DecimalField(max_digits=20, decimal_places=2)
    predicted_price = models.DecimalField(max_digits=20, decimal_places=2)
    previous_actual_price = models.DecimalField(max_digits=20, decimal_places=2)
    previous_predicted_price = models.DecimalField(max_digits=20, decimal_places=2)
    previous_actual_date = models.DateField()
    previous_predicted_date = models.DateField()

    def __str__(self):
        return self.predicted_price


class Simulation(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)
    shares = models.IntegerField()
    start_date = models.DateField()
    end_date = models.DateField()

    def __str__(self):
        return 'This a simulation boy!'
