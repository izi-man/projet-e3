let selectedAnswer = null;

const answers = document.querySelectorAll(".answer");
const submitButton = document.getElementById("submit");
const result = document.getElementById("result");

answers.forEach((answer) => {

    answer.addEventListener("click", () => {

        answers.forEach((item) => {
            item.style.fontWeight = "normal";
        });

        answer.style.fontWeight = "bold";

        selectedAnswer = answer;

        submitButton.disabled = false;
    });

});


submitButton.addEventListener("click", async () => {

    if (!selectedAnswer) {
        return;
    }

    const correct = selectedAnswer.dataset.correct === "true";

    // Exemple de données envoyées au modèle
    const data = {
        temps_reponse: 2.3,
        nb_erreurs: correct ? 0 : 1,
        score: correct ? 100 : 50
    };

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );

        if (!response.ok) {
            throw new Error("Erreur API");
        }

        const prediction = await response.json();

        result.textContent =
            `Réponse enregistrée. Prédiction IA : ${prediction.satisfait}`;

    } catch (error) {

        console.error(error);

        result.textContent =
            "Impossible de contacter l'API.";

    }

});