from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import onnxruntime as ort
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the ONNX model
model_path = r"full_model_5.onnx"
onnx_session = ort.InferenceSession(model_path)

# Function to preprocess the input data
def preprocess_csv(file: UploadFile):
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file.file, header=0)
        
        # Extract I and Q components and reshape them for the model
        I = df.iloc[:, 5:2048+5].values.reshape(-1, 64, 32, 1).astype(np.float32)
        Q = df.iloc[:, 2048+5:4096+5].values.reshape(-1, 64, 32, 1).astype(np.float32)
        return I, Q
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

# Define prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Preprocess the uploaded CSV file
        I_data, Q_data = preprocess_csv(file)
        
        # Prepare input for ONNX model
        input_name_I = onnx_session.get_inputs()[0].name
        input_name_Q = onnx_session.get_inputs()[1].name
        output_name = onnx_session.get_outputs()[0].name

        # Run inference
        predictions = onnx_session.run([output_name], {input_name_I: I_data, input_name_Q: Q_data})[0]
        predicted_classes = np.argmax(predictions, axis=1) + 1  # Adjust index to start from 1
        
        # Return predictions as JSON
        return JSONResponse(content={
            "predictions": predictions.tolist(),
            "predicted_classes": predicted_classes.tolist()
        })
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Unexpected error: {e}"}, status_code=500)

# Run the FastAPI app
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000)
