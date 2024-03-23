/*JS CODE, Connection between html frontend and python backend*/
document.addEventListener('DOMContentLoaded', function() {
    initializePage(); // Clear some text
});

function initializePage() {

    var loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'none';
    loadingMessage.textContent = ''; 
    
    var reportSection = document.getElementById('report-section');
    reportSection.innerHTML = '';

}

/*The js code for regression or classification*/
document.getElementById('classification-btn').addEventListener('click', function() {
    handleChoice('c', this);
    var messageElement = document.getElementById('upload-success-message');
    var container = document.querySelector('.background-container');
    messageElement.textContent = "You have chosen classification";
    messageElement.style.color = "green";
    messageElement.style.textShadow = '1px 1px 1px rgba(255, 255, 255, 0.2)';
    messageElement.style.display = 'block';
    messageElement.style.fontSize = '24px';
    setTimeout(function() {
        messageElement.style.display = 'none'; // Hide the message after 3 seconds
    }, 3000);
});

document.getElementById('regression-btn').addEventListener('click', function() {
    handleChoice('r', this);
    var messageElement = document.getElementById('upload-success-message');
    var container = document.querySelector('.background-container');
    messageElement.textContent = "You have chosen regression";
    messageElement.style.color = "green";
    messageElement.style.textShadow = '1px 1px 1px rgba(255, 255, 255, 0.2)';
    messageElement.style.display = 'block';
    messageElement.style.fontSize = '24px';
    setTimeout(function() {
        messageElement.style.display = 'none'; // Hide the message after 3 seconds
    }, 3000);
});

function handleChoice(choice, clickedButton) {
    const otherButtonId = choice === 'c' ? 'regression-btn' : 'classification-btn';
    const otherButton = document.getElementById(otherButtonId);

    // Change color to green
    clickedButton.style.backgroundColor = 'green';
    otherButton.style.backgroundColor = 'rgb(0, 36, 63)';

    fetch('/choose-between-r-or-c', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({choice: choice}),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

/* FIND CSV FILE and LOAD CSV */
document.getElementById('csv-upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    var formData = new FormData(this);
    var fileInput = document.getElementById('csvFile');
    var file = fileInput.files[0]; // Get the uploaded file

    fetch('/upload-csv', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        var messageElement = document.getElementById('upload-success-message');
        var container = document.querySelector('.background-container');
        if(data.success) {
            messageElement.textContent = "CSV file uploaded successfully!";
            messageElement.style.display = 'block';
            // Hide the message after 3 seconds:
            setTimeout(() => messageElement.style.display = 'none', 3000);

            var fileNameDisplay = document.getElementById('filename-display');

            // length of filename
            if (file.name.length > 21) {

                var part1 = file.name.substring(0, 21);
                var part2 = file.name.substring(21);
                fileNameDisplay.innerHTML = "Loaded CSV: <br>" + part1 + "-" + "<br>" + part2 + data.df_info;
            } else {

                fileNameDisplay.innerHTML = "Loaded CSV: <br>" + file.name + data.df_info;
            }
        } else {
            messageElement.textContent = "Error: " + data.message;
            messageElement.style.display = 'block';
            // Red error text
            messageElement.style.color = 'red';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('upload-success-message').textContent = "Error: An unexpected error occurred.";
        document.getElementById('upload-success-message').style.display = 'block';
        document.getElementById('upload-success-message').style.color = 'red';
    });
})

/* COLUMN NAME */
document.getElementById('show-button').addEventListener('click', function() {
    fetch('/show-csv')
    .then(response => response.json())
    .then(data => {
        if(data.success) {
            document.getElementById('df-show').textContent = data.df_show;
        } else {
            document.getElementById('df-show').textContent = "Error: Unable to retrieve DataFrame info.";
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('df-show').textContent = "Error: An unexpected error occurred.";
    });
});

/* Pick up the column number */
let storedReportText = '';

document.getElementById('choose-target-button').addEventListener('click', function() {
    
    document.getElementById('df-show').innerHTML = '';
    var reportSection = document.getElementById('report-section');
    reportSection.innerHTML = '';

    
    var loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'block';
    loadingMessage.style.bottom = '300px';
    loadingMessage.style.fontSize = '40px'; 
    loadingMessage.style.textShadow = '1px 1px 1px rgba(0, 36, 63, 1), 5px 5px 5px rgba(0, 36, 63, 1)';
    loadingMessage.style.color = "yellow"; 
    // Loading.. code
    var dots = 0;
        function updateLoadingMessage() {
            var baseText = 'Loading';
            var dotText = '.'.repeat(dots);
            var waitText = 'Please Wait';
            loadingMessage.innerHTML = `${baseText}${dotText}<br>${waitText}`;

            dots = (dots + 1) % 4;
        }

    // Update the message every 500 milliseconds
    var loadingInterval = setInterval(updateLoadingMessage, 500);

    var targetNumber = document.getElementById('target-number').value;
    fetch('/choose-target', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({targetNumber: targetNumber}),
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(loadingInterval); 
        if(data.success) {
            loadingMessage.textContent = 'Loading Complete! Click SEE REPORT!';
            loadingMessage.style.color = "green";
            loadingMessage.style.fontSize = '40px'; 
            loadingMessage.style.textShadow = '1px 1px 1px rgba(255, 255, 255, 0.2)'; 

            document.getElementById('see-report-button').disabled = false;
        } else {
            console.error("Error choosing target:", data.message);
            loadingMessage.style.display = 'none';
            reportSection.innerHTML = "Error: An unexpected error occurred."; // THIS,WHEN YOU LOAD ERROR CSV FILE, LIKE MOVIES
            reportSection.style.color = "red";
            reportSection.style.position = 'relative'; 
            reportSection.style.bottom = '300px';
            reportSection.style.fontSize = '40px'; 
        }
    })
    .catch(error => {
        clearInterval(loadingInterval);
        console.error('Error:', error);
        loadingMessage.style.display = 'none';
        reportSection.innerHTML = "Error: unexpected error occurred.";
        reportSection.style.color = "red";
        reportSection.style.position = 'relative'; 
        reportSection.style.bottom = '300px';
        reportSection.style.fontSize = '40px'; 
    });
});

document.getElementById('see-report-button').addEventListener('click', function() {
    var reportSection = document.getElementById('report-section');
    var loadingMessage = document.getElementById('loading-message'); 

    reportSection.innerHTML = '';
    loadingMessage.style.display = 'none'; 
    fetch('/see-report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if(data.success) {
            document.getElementById('df-show').textContent = data.report
            document.getElementById('best-model-r2-score').innerHTML = `Best Model and Score: ${data.best_model_score}`;
        } else {
            reportSection.innerHTML = "Error: " + data.message;
            reportSection.style.color = "red"; 
            reportSection.style.bottom = '300px';
            reportSection.style.fontSize = '40px'; 
        }
    })
    .catch(error => {
        console.error('Error:', error);
        reportSection.innerHTML = "Error: An unexpected error occurred.";
        reportSection.style.color = "red"; 
        reportSection.style.bottom = '300px';
        reportSection.style.fontSize = '40px'; 
    });
});

document.getElementById('dump-best-model-button').addEventListener('click', function() {
    // Start displaying the loading message with dynamic dots
    var loadingMessage = document.getElementById('loading-message');
    loadingMessage.style.display = 'block';
    loadingMessage.style.bottom = '300px';
    loadingMessage.style.fontSize = '40px'; 
    loadingMessage.style.textShadow = '1px 1px 1px rgba(0, 36, 63, 1), 5px 5px 5px rgba(0, 36, 63, 1)';
    loadingMessage.style.color = "yellow"; 

    var dots = 0; // Start with no dots
    function updateLoadingMessage() {
        var baseText = 'Saving Model';
        var dotText = '.'.repeat(dots);
        var waitText = 'Please Wait';
        loadingMessage.innerHTML = `${baseText}${dotText}<br>${waitText}`;

        dots = (dots + 1) % 4;
    }

    var loadingInterval = setInterval(updateLoadingMessage, 500);

    fetch('/dump-model-final', {
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(loadingInterval); 
        if(data.success) {
            loadingMessage.textContent = "Congratulations! Your model has been saved successfully";
            loadingMessage.style.color = "green";
            loadingMessage.style.fontSize = '30px'; 
            loadingMessage.style.textShadow = '1px 1px 1px rgba(255, 255, 255, 0.2)';

            setTimeout(() => loadingMessage.style.display = 'none', 3000);
        } else {
            loadingMessage.style.display = 'none';
            console.error('Error:', data.message);
        }
    })
    .catch(error => {
        clearInterval(loadingInterval); 
        console.error('Error:', error);
        loadingMessage.style.display = 'none';
    });
});