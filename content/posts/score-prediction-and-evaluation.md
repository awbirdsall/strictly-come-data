---
title: "Dance Score Predictions"
date: 2018-10-18T21:48:30-04:00
draft: false
---

*It's time to predict some scores!*

![Oti and Danny scoring a 40](/strictly-come-data/images/strictly-perfect-score.gif)

[Last time](../predicting-dance-scores-with-ml), I walked through how I trained three optimized machine learning models to predict Strictly Come Dancing scores, using all available data through Week 3 of the current series, Series 16. All three models performed decently, but the gradient boosting regressor tended to achieve the best accuracy scores in the cross-validation. So, I used that model as the "official" prediction of the week's scores, though I was interested to see how the other models' predictions compared.

With the trained models in hand, all that's necessary is to feed them the inputs for week 4, and ask them to predict the total score out of 40. I assembled the prediction inputs using the information available on [Ultimate Strictly](http://www.ultimatestrictly.com/series-16-week-4/). For example, Kate and Aljaž were scheduled to dance a Samba (to [Toto's](https://www.mcsweeneys.net/articles/totos-africa-by-ernest-hemingway) ["Africa"](https://www.newyorker.com/culture/rabbit-holes/the-overwhelming-emotion-of-hearing-totos-africa-remixed-to-sound-like-its-playing-in-an-empty-mall), though the model doesn't know what to do with that information—[poor model!](https://www.youtube.com/watch?v=MKpjThxxHOI)). I also added the information that all of the dances will take place during Series 16 and take place 0.23 of the way through the series. I imported the inputs into a Python session as a DataFrame called `predict_inputs`, et voilà!—

```python
predict_inputs['gbr'] = gbr.predict(predict_inputs).round()
predict_inputs['xgbr'] = xgbr.predict(predict_inputs).round()
predict_inputs['rfr'] = rfr.predict(predict_inputs).round()
```

where `gbr`, `xgbr`, and `rfr` denote the gradient boosting, XGBoost, and random forest regressors, respectively. I rounded the predictions to the nearest integer since judges can't award fractional scores.

The predictions were as follows:

| celebrity | professional     | dance                | gbr | xgbr | rfr  |
|-----------|------------------|-------------------------------|-----|------|------|
| Ashley Roberts   | Pasha Kovalev     | Tango           | 32 | 33 | 28 |
| Charles Venn     | Karen Clifton     | Salsa           | 24 | 24 | 25 |
| Danny John-Jules | Amy Dowden        | Viennese Waltz  | 29 | 30 | 27 |
| Faye Tozer       | Giovanni Pernice  | Rumba           | 30 | 28 | 29 |
| Graeme Swann     | Oti Mabuse        | Jive            | 22 | 25 | 23 |
| Joe Sugg         | Dianne Buswell    | Cha-cha-cha     | 26 | 28 | 25 |
| Kate Silverton   | Aljaž Skorjanec   | Samba           | 25 | 26 | 26 |
| Katie Piper      | Gorka Márquez     | Jive            | 17 | 17 | 20 |
| Lauren Steadman  | AJ Pritchard      | Quickstep       | 23 | 24 | 26 |
| Dr. Ranj Singh   | Janette Manrara   | Paso Doble      | 24 | 26 | 25 |
| Seann Walsh      | Katya Jones       | Charleston      | 23 | 23 | 23 |
| Stacey Dooley    | Kevin Clifton     | Foxtrot         | 27 | 27 | 25 |
| Vick Hope        | Graziano Di Prima | Quickstep      | 25 | 26 | 25 |

None of the predictions seemed absolutely bonkers, which was great! The random forest predictions did seem all be smushed together in the middling 20s, which seemed less plausible based on experience—I would agree with the other two models to expect a few totals in the thirties for standout dances. A quick seaborn `distplot` makes the difference in distributions clear:

{{< figure src="/strictly-come-data/images/week-4-model-predictions-dist.png" title="Plot: histogram comparing predicted model score distributions for Week 4">}}

I was then lucky enough to have the assistance of the biggest fan of Strictly in the U.S., who was willing to contribute fan-predicted scores for the upcoming show (without having seen the model predictions). A comparison of the gradient boosting model and the fan predictions:

| celebrity        | professional      | dance          | fan | gbr  |
|------------------|-------------------|----------------|-----|------|
| Ashley Roberts   | Pasha Kovalev     | Tango          | 36  | 32 |
| Charles Venn     | Karen Clifton     | Salsa          | 25  | 24 |
| Danny John-Jules | Amy Dowden        | Viennese Waltz | 29  | 29 |
| Faye Tozer       | Giovanni Pernice  | Rumba          | 36  | 30 |
| Graeme Swann     | Oti Mabuse        | Jive           | 30  | 22 |
| Joe Sugg         | Dianne Buswell    | Cha-cha-cha    | 28  | 26 |
| Kate Silverton   | Aljaž Skorjanec   | Samba          | 28  | 25 |
| Katie Piper      | Gorka Márquez     | Jive           | 21  | 17 |
| Lauren Steadman  | AJ Pritchard      | Quickstep      | 25  | 23 |
| Dr. Ranj Singh   | Janette Manrara   | Paso Doble     | 25  | 24 |
| Seann Walsh      | Katya Jones       | Charleston     | 30  | 23 |
| Stacey Dooley    | Kevin Clifton     | Foxtrot        | 30  | 27 |
| Vick Hope        | Graziano Di Prima | Quickstep      | 28  | 25 |

Taken as a whole, the fan predictions seemed to tend a little higher. The largest discrepancy was for Seann and Katya's Charleston, with a seven-point spread between the two predictions. Could it be the model was overweighting the celebrity results from the early weeks of the series, when scores can be very low, and not sufficiently accounting for week-to-week improvement?  Despite this, the fan and model agreed that Danny and Amy's Viennese waltz was likely to score a 29, and that Katie and Gorka were at high risk of finding themselves at the bottom of the leader board.

All left to do was wait until last Saturday, when the scores were in. When the judges revealed their scores, how would the results compare?

Find out next time, and keeeeeeeeeeeeeeeeeeeeep data-ing!
