const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");
const preview = document.getElementById("preview");

let selectedFile = null;

fileInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];

    if (selectedFile) {
        const reader = new FileReader();

        reader.onload = function(event) {
            preview.innerHTML = `<img src="${event.target.result}" />`;
        };

        reader.readAsDataURL(selectedFile);
    }
});

predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please upload an image!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    result.textContent = "Predicting...";

    const res = await fetch("http://localhost:9020/predict", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    result.textContent = "Prediction: " + data.prediction;
});
