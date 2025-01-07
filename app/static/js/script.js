$(document).ready(function () {
    $("#predictionForm").submit(function (event) {
      event.preventDefault();
  
      var formData = $(this).serialize();
  
      $.ajax({
        url: "/predict",
        type: "POST",
        data: formData,
        success: function (response) {
          console.log(response);
          $("#predictionResult").text(response.prediction_text);
          $(".result").show();
        },
        error: function (error) {
          console.log("Error:", error);
          $("#predictionResult").text(
            "Error occurred while processing the prediction."
          );
          $(".result").show();
        },
      });
    });
  });
  