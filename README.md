Pour tester l'application vous pouvez lancer la commande
```bash
docker-compose up -d

Vous trouverez l'application en fonctionnement sur l'addresse http://locahost.

# Machine Learning
L'application fonctionne avec une API en Flask qui fonctionne avec une application en Machine Learning qui repose sur un Random Forest Classifier.
Pour comprendre comment fonctionne les models de classification vous pouvez consulter cette article https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3
Les données utiliser sont stocker en JSON, mais sont des Dummy Data. L'idée était quand même de coller à la réalité.
Cette application est fonctionnels mais cela demande quand même de faire des tests avec des données réeals.
