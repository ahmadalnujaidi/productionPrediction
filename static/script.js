document
  .getElementById("predictionForm")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = {
      weight: document.getElementById("weight").value,
      temperature: document.getElementById("temperature").value,
      speed: document.getElementById("speed").value,
      processTime: document.getElementById("processTime").value,
      components: document.getElementById("components").value,
      efficiency: document.getElementById("efficiency").value,
      quantity: document.getElementById("quantity").value,
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (result.error) {
        alert("Error: " + result.error);
        return;
      }

      document.getElementById("delay-prediction").textContent =
        result.delay_prediction;
      document.getElementById("defect-prediction").textContent = (
        result.defect_prediction * 100
      ).toFixed(2);
      document.getElementById("result").classList.remove("hidden");
    } catch (error) {
      alert("Error: " + error.message);
    }
  });
