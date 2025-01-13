from operator import is_not

from accelerate.commands.config.config import description
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from transformers import pipeline
import torch
import re
import numpy as np
from flask import jsonify
from model import RN

print(torch.cuda.is_available())

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device = "cuda"
# replace with "mps" to run on a Mac device
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


class UserInput(FlaskForm):
    input = StringField("Enter description")
    submit = SubmitField("Submit")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict_nn(attributes):
    """
    Perform inference using the trained neural network.
    :param attributes: A dictionary with the input features for the neural network.
    :return: Neural network predictions.
    """
    print("fac predictie")
    rn = RN()
    rn.prepare(attributes)
    rn.train_dynamic(epochs=50)
    rn.save_model()
    
    weights_input_hidden = np.load('Catology/weights_input_hidden.npy')
    weights_hidden_output = np.load('Catology/weights_hidden_output.npy')
    b1 = np.load('Catology/b1.npy')
    b2 = np.load('Catology/b2.npy')

    # Convert the attributes to the correct input format

    input_data = np.array([[
        attributes["Number"],
        attributes["Ext"],
        attributes["Shy"],
        attributes["Calm"],
        attributes["Scared"],
        attributes["Vigilant"],
        attributes["Affectionate"],
        attributes["Friendly"],
        attributes["Solitary"],
        attributes["Aggressive"],
        attributes["PredatorMammal"],
        attributes["Coat"],
        attributes["Intelligence_Score"]
    ]])

    print(f"input data: {input_data}")
    input_data2 = input_data[input_data != -1]
    print(f"input data2: {input_data2}")

    
    hidden_layer, output_layer = rn.forward(input_data2)
    predicted_class = np.argmax(output_layer, axis=1)[0]
    return predicted_class




def validate_output(generated_text):
    """
    Validate the model's output to ensure it matches the expected format.
    """
    pattern = r"-([\w_]+):\s*(-?\d)"
    matches = re.findall(pattern, generated_text)
    if len(matches) == 13:  # Ensure all attributes are present
        return {attr: int(score) for attr, score in matches}
    else:
        print("Validation failed for output:", generated_text)
        return None


def evaluate_attributes(input_text):
    """
    Generate and validate the attributes using Gemma.
    """

    prompt = f"""
    Evaluate the following characteristics of the animal and assign a score each attribute. 
    Respond ONLY with the scores in the following format and only if you find details about that attribute in text, where you don t find details about attributes put -1.
    The attributes to evaluate are:
    
    -Number: [between 1 and 5]
    -Ext: [between 0 and 4]
    -Shy: [between 1 and 5]
    -Calm: [between 1 and 5]
    -Scared: [between 1 and 5]
    -Vigilant: [between 1 and 5]
    -Affectionate: [between 1 and 5]
    -Friendly: [between 1 and 5]
    -Solitary: [between 1 and 5]
    -Aggressive: [between 1 and 5]
    -PredatorMammal: [between 0 and 4]
    -Coat: [between 0 and 4]
    -Intelligence_Score: [between 0 and 4]

    where coat refers to the length of the hair
    and number refers to how many cats are in the same household
    Answear with a number in the interval for each attribute you can find in the text, for those that are not sure, you put nothing not event the attribute name. If there is no mention of an attribute, assign a score of 0. No description, 
    Text: "{input_text}"

        """

    # Generate response using Gemma
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    # Debug output
    print(f"Generated raw output: {generated_text}")

    # Validate output
    attributes_dict = validate_output(generated_text)

    print(f"Parsed attributes: {attributes_dict}")
    return attributes_dict

def return_cat_description(predicted_class):
    prompt = f"""
        You will be given a number. Based on this number i want you to return the full name of the breed.
        The breed will be one of these based on the number:
    "Bengal": 1,
    "Birman": 2,
    "British Shorthair": 3,
    "Chartreux": 4,
    "European Shorthair": 5,
    "Maine Coon": 6,
    "Persian": 7,
    "Ragdoll": 8,
    "Sphynx": 9,
    "Oriental Shorthair": 10,
    "Turkish Van": 11,
    "Other": 12,
    "Unknown": 13 
    Only say the breed, nothing else!!
    Here is the number: {predicted_class}
            """
    # Generate response using Gemma
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    # Debug output
    print(f"Generated raw output: {generated_text}")

    return generated_text

def return_human_description(predicted_class, human_description):
    prompt = f"""
           You will be given a number and a description of a human. Based on this number i want you to return the full name of the breed.
           The breed will be one of these based on the number:
       "Bengal": 1,
       "Birman": 2,
       "British Shorthair": 3,
       "Chartreux": 4,
       "European Shorthair": 5,
       "Maine Coon": 6,
       "Persian": 7,
       "Ragdoll": 8,
       "Sphynx": 9,
       "Oriental Shorthair": 10,
       "Turkish Van": 11,
       "Other": 12,
       "Unknown": 13 
       Based on the description of the human, i want you to say why this breed would be a good match for this person
       Here is the number: {predicted_class}
       Here is the description of the human: {human_description}
       format the text a bit so it's not just a block of text and don't ask the user anything, just give the breed and the reasoning!!!
               """
    # Generate response using Gemma
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    # Debug output
    print(f"Generated raw output: {generated_text}")

    return generated_text

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/identify', methods=["GET", "POST"])
def identify():
    if request.method == "POST":
        try:
            # Parse JSON data from the request
            data = request.get_json()
            user_input = data.get('input', '').strip()
            print(f"User input: {user_input}")

            # Initialize response variables
            attributes = None
            prediction = None
            description = None

            if user_input:
                # Process the input
                attributes = evaluate_attributes(user_input)
                if attributes:
                    prediction = predict_nn(attributes)
                    print("prediction: ")
                    print(prediction)
                if prediction:
                    description = return_cat_description(prediction)

                # Send the description as a JSON response

                print("here")
                return jsonify({'description': description or "No description available"})
            else:
                return jsonify({'error': "No input provided"}), 400

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500

    # For GET requests (if any), return the form template
    form = UserInput()
    return render_template('identify.html', user_input=form)

@app.route('/match', methods=["GET", "POST"])
def match():
    if request.method == "POST":
        try:
            # Parse JSON data from the request
            data = request.get_json()
            user_input = data.get('input', '').strip()
            print(f"User input: {user_input}")

            # Initialize response variables
            attributes = None
            prediction = None
            description = None

            if user_input:
                # Process the input
                attributes = evaluate_attributes(user_input)
                if attributes:
                    prediction = predict_nn(attributes)
                    print("prediction: ")
                    print(prediction)
                if prediction:
                    description = return_human_description(prediction, user_input)

                print("here")
                return jsonify({'description': description or "No description available"})
            else:
                return jsonify({'error': "No input provided"}), 400

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500

    return render_template('match.html', user_description = UserInput())

@app.route('/compare', methods=["GET", "POST"])
def compare():
    if request.method == "POST":
        try:
            # Parse JSON data from the request
            data = request.get_json()
            user_input = data.get('input', '').strip()
            print(f"User input: {user_input}")

            # Initialize response variables
            attributes = None
            prediction = None
            description = None

            if user_input:
                # Process the input
                attributes = get_breed_id(user_input)
                if attributes:
                    prediction = predict_nn(attributes)
                    print("prediction: ")
                    print(prediction)
                if prediction:
                    description = return_human_description(prediction, user_input)

                print("here")
                return jsonify({'description': description or "No description available"})
            else:
                return jsonify({'error': "No input provided"}), 400

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500
    return render_template('compare.html')

@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(_,_,debug=True)
