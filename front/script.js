document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const data = {
        age: document.getElementById('age').value,
        sexe: document.getElementById('sexe').value,
        situation_familiale: document.getElementById('situation_familiale').value,
        enfants_a_charge: document.getElementById('enfants_a_charge').value,
        type_de_vehicule: document.getElementById('type_de_vehicule').value,
        experience_de_conduite: document.getElementById('experience_de_conduite').value,
        historique_accidents: document.getElementById('historique_accidents').value,
        usage_vehicule: document.getElementById('usage_vehicule').value,
        couverture_souhaitee: document.getElementById('couverture_souhaitee').value,
    };

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Offre recommandée : ' + data.prediction;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
