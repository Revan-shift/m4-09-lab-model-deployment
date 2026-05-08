{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07cd0688-9288-4273-93da-440350c40aee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [08/May/2026 10:29:51] \"GET /health HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [08/May/2026 10:29:53] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [08/May/2026 10:29:55] \"POST /predict HTTP/1.1\" 400 -\n",
      "127.0.0.1 - - [08/May/2026 10:29:57] \"POST /predict HTTP/1.1\" 400 -\n",
      "127.0.0.1 - - [08/May/2026 10:29:59] \"POST /predict HTTP/1.1\" 400 -\n",
      "127.0.0.1 - - [08/May/2026 10:30:01] \"POST /predict_batch HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the model and target names upon server startup\n",
    "model = joblib.load(\"model.joblib\")\n",
    "target_names = joblib.load(\"target_names.joblib\")\n",
    "\n",
    "@app.route(\"/health\", methods=[\"GET\"])\n",
    "def health():\n",
    "    \"\"\"Health check endpoint for load balancers and monitoring.\"\"\"\n",
    "    return jsonify({\"status\": \"healthy\"}), 200\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    \"\"\"Endpoint for single predictions with input validation.\"\"\"\n",
    "    data = request.get_json(force=True, silent=True)\n",
    "    \n",
    "    if not data or \"features\" not in data:\n",
    "        return jsonify({\"error\": \"Missing 'features' key in JSON request body.\"}), 400\n",
    "    \n",
    "    features = data[\"features\"]\n",
    "    \n",
    "    if not isinstance(features, list) or len(features) != 4:\n",
    "        return jsonify({\"error\": \"The 'features' key must contain a list of exactly 4 values.\"}), 400\n",
    "    \n",
    "    try:\n",
    "        # Convert all features to floats to ensure they are numeric\n",
    "        features_float = [float(x) for x in features]\n",
    "    except (ValueError, TypeError):\n",
    "        return jsonify({\"error\": \"All values in the 'features' list must be numeric.\"}), 400\n",
    "    \n",
    "    # Reshape for a single sample prediction\n",
    "    features_array = np.array([features_float])\n",
    "    \n",
    "    # Make predictions\n",
    "    pred_idx = model.predict(features_array)[0]\n",
    "    pred_probs = model.predict_proba(features_array)[0]\n",
    "    \n",
    "    predicted_class = target_names[pred_idx]\n",
    "    probabilities = {target_names[i]: float(prob) for i, prob in enumerate(pred_probs)}\n",
    "    \n",
    "    return jsonify({\n",
    "        \"predicted_class\": predicted_class,\n",
    "        \"probabilities\": probabilities\n",
    "    }), 200\n",
    "\n",
    "@app.route(\"/predict_batch\", methods=[\"POST\"])\n",
    "def predict_batch():\n",
    "    \"\"\"Endpoint for processing multiple samples at once.\"\"\"\n",
    "    data = request.get_json(force=True, silent=True)\n",
    "    \n",
    "    if not data or \"samples\" not in data:\n",
    "        return jsonify({\"error\": \"Missing 'samples' key in JSON request body.\"}), 400\n",
    "    \n",
    "    samples = data[\"samples\"]\n",
    "    \n",
    "    try:\n",
    "        samples_array = np.array(samples, dtype=float)\n",
    "        if samples_array.ndim != 2 or samples_array.shape[1] != 4:\n",
    "            raise ValueError\n",
    "    except (ValueError, TypeError, IndexError):\n",
    "        return jsonify({\"error\": \"The 'samples' key must contain a list of lists, where each sub-list has exactly 4 numeric values.\"}), 400\n",
    "        \n",
    "    # Make predictions for the batch\n",
    "    predictions_idx = model.predict(samples_array)\n",
    "    predictions_probs = model.predict_proba(samples_array)\n",
    "    \n",
    "    results = []\n",
    "    for i in range(len(predictions_idx)):\n",
    "        predicted_class = target_names[predictions_idx[i]]\n",
    "        probabilities = {target_names[j]: float(prob) for j, prob in enumerate(predictions_probs[i])}\n",
    "        results.append({\n",
    "            \"predicted_class\": predicted_class,\n",
    "            \"probabilities\": probabilities\n",
    "        })\n",
    "        \n",
    "    return jsonify({\"predictions\": results}), 200\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "       app.run(debug=True, use_reloader=False, port=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3baebee9-aee5-45e2-b6b7-6c2ec1ce7670",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
