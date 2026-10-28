// =======================================
// PHÂN TÍCH BÌNH LUẬN
// =======================================

async function predict() {

    const text = document.getElementById("comment").value;

    if (text.trim() === "") {
        alert("Vui lòng nhập phản hồi.");
        return;
    }

    document.getElementById("result").innerHTML = `
        <div class="text-center mt-4">

            <div class="spinner-border text-primary"></div>

            <p class="mt-3">
                Đang phân tích...
            </p>

        </div>
    `;

    const response = await fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            text: text
        })

    });

    const data = await response.json();

    let sentimentColor = "secondary";

    if (data.sentiment === "Tích cực")
        sentimentColor = "success";

    else if (data.sentiment === "Tiêu cực")
        sentimentColor = "danger";

    else
        sentimentColor = "warning";

    document.getElementById("result").innerHTML = `

<div class="card shadow mt-4">

<div class="card-body">

<h3 class="mb-4">

📊 Kết quả phân tích

</h3>

<p>

<b>Chủ đề:</b>

<span class="badge bg-primary">

${data.topic}

</span>

</p>

<div class="progress mb-4">

<div class="progress-bar bg-primary"

style="width:${data.topic_confidence}%">

${data.topic_confidence}%

</div>

</div>

<p>

<b>Cảm xúc:</b>

<span class="badge bg-${sentimentColor}">

${data.sentiment}

</span>

</p>

<div class="progress">

<div class="progress-bar bg-${sentimentColor}"

style="width:${data.sentiment_confidence}%">

${data.sentiment_confidence}%

</div>

</div>

</div>

</div>

`;

    document.getElementById("comment").value = "";

    loadFeedbacks();

}



// =======================================
// LOAD LỊCH SỬ
// =======================================

async function loadFeedbacks() {

    const response = await fetch("/feedbacks");

    const data = await response.json();

    const tbody = document.getElementById("feedbackTable");

    if (!tbody) return;

    tbody.innerHTML = "";

    data.forEach(item => {

        let color = "#f59e0b";

        if (item.sentiment === "Tích cực")
            color = "#22c55e";

        if (item.sentiment === "Tiêu cực")
            color = "#ef4444";

        tbody.innerHTML += `

<tr>

<td>${item.id}</td>

<td>${item.comment}</td>

<td style="font-weight:bold;color:${color}">

${item.sentiment}

</td>

<td>

${item.emotion}

</td>

<td>

${item.topic}

</td>

<td>

${item.created_at}

</td>

</tr>

`;

    });

}



// =======================================
// TỰ LOAD KHI MỞ TRANG
// =======================================

window.onload = function () {

    loadFeedbacks();

};