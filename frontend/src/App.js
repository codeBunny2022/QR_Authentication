import React, { useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  // Function to determine the final decision
  const getFinalDecision = () => {
    if (!prediction) return null;
    
    const results = [prediction.RandomForest, prediction.SVM, prediction.CNN];
    const realCount = results.filter((res) => res === 0).length;
    const fakeCount = results.filter((res) => res === 1).length;

    return fakeCount > realCount ? "Fake" : "Real";
  };

  const finalDecision = getFinalDecision();
  const finalDecisionStyle = finalDecision === "Fake" ? "danger" : "success";

  return (
    <div className="container mt-5">
      <div className="card p-4 shadow-lg">
        <h2 className="text-center">QR Code Authentication by @chirag</h2>

        <div className="mb-3">
          <input type="file" onChange={handleFileChange} className="form-control" />
        </div>

        <button onClick={handleUpload} className="btn btn-primary w-100">
          Upload & Predict
        </button>

        {prediction && (
          <div className="mt-4">
            <h4 className="text-center">Predictions</h4>

            <div className="d-flex justify-content-around">
              <div className={`card p-3 text-white bg-${prediction.RandomForest === 0 ? "success" : "danger"}`}>
                <h5>Random Forest</h5>
                <p>{prediction.RandomForest === 0 ? "First Print" : "Second Print"}</p>
              </div>

              <div className={`card p-3 text-white bg-${prediction.SVM === 0 ? "success" : "danger"}`}>
                <h5>SVM</h5>
                <p>{prediction.SVM === 0 ? "First Print" : "Second Print"}</p>
              </div>

              <div className={`card p-3 text-white bg-${prediction.CNN === 0 ? "success" : "danger"}`}>
                <h5>CNN</h5>
                <p>{prediction.CNN === 0 ? "First Print" : "Second Print"}</p>
              </div>
            </div>

            {/* Final Decision Box */}
            <div className={`alert alert-${finalDecisionStyle} text-center mt-4`}>
              <h4>Final Decision: {finalDecision}</h4>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
