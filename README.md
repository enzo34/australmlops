# But de l'application
Cette application est à destination des assurances afin de déterminer à partir des données rentrer par un utilisateurs de trouver l'offre la plus adapter. Par exemple:
```json
{
        "age": 35,
        "sexe": "M",
        "situation_familiale": "Divorcé(e)",
        "enfants_a_charge": 1,
        "type_de_vehicule": "Moto",
        "experience_de_conduite": 10,
        "historique_accidents": "Aucun",
        "usage_vehicule": "Les deux",
        "couverture_souhaitee": "Totale",
    },
```
L'application est un simple formulaires qui sera plus tard convertis en Web Components pour être utiliser sur n'importe quelle application via un CDN.
Pour tester l'application vous pouvez lancer la commande.
```bash
docker-compose up -d
```
Bien sure assurez vous d'avoir Docker d'installer sur votre machine
Vous trouverez l'application en fonctionnement sur l'addresse http://locahost.

# Machine Learning
L'application fonctionne avec une API en Flask qui fonctionne avec une application en Machine Learning qui repose sur un Random Forest Classifier.
Pour comprendre comment fonctionne les models de classification vous pouvez consulter cette article https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3
Les données utiliser sont stocker en JSON, mais sont des Dummy Data. L'idée était quand même de coller à la réalité.
Cette application est fonctionnels mais cela demande quand même de faire des tests avec des données réeals et de pousser plus loins l'algorithme de base qui est utiliser.
