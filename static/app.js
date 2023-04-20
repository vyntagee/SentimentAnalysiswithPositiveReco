document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const searchQuery = document.getElementById("search-query").value;
    const response = await fetch("/analyze", {
        method: "POST",
        body: new FormData(event.target),
    });
    const results = await response.json();
    updateResultsTable(results);
});

function updateResultsTable(results) {
    const resultsTableBody = document.getElementById("results");
    resultsTableBody.innerHTML = "";
    results.forEach((result) => {
        const row = document.createElement("tr");
        ["user_name", "date", "text", "sentiment", "category", "recommendation"].forEach((key) => {
            const cell = document.createElement("td");
            cell.textContent = result[key];
            row.appendChild(cell);
        });
        resultsTableBody.appendChild(row);
    });
}
